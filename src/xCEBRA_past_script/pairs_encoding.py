import os, argparse
from pathlib import Path
from itertools import combinations_with_replacement
import numpy as np
import torch, mat73, scipy.io
import cebra
from cebra.data import DatasetxCEBRA, ContrastiveMultiObjectiveLoader
from cebra.models import init as init_model

from src.config import (
    MODEL_DIR, NEURAL_PATH, EMOTION_PATH, N_LATENTS,
    BEHAVIOR_INDICES, ELECTRODE_NAMES, NODE_MAP, PATIENT_CONFIG,
    FULL_NEURAL_PATH
)
from src.utils_training import (
    feat_indices_for_electrodes,
    electrode_is_missing,
    build_cebra_config,
    train_and_save
)

# ---- constants ----
N_ELECTRODES, N_BANDS, N_LAGS = len(ELECTRODE_NAMES), 5, 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_steps", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pairs_limit", type=int, default=-1)
    args = ap.parse_args()
    device = torch.device(args.device)

    # ---- setup ----
    pid = int(float(os.environ["PATIENT_ID"]))
    _, patient_id = PATIENT_CONFIG[pid]
    out_root = Path(f"./output_xCEBRA_lags/{patient_id}/pairs_encoding")
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- load data ----
    train_idx = np.load(MODEL_DIR / "train_idx.npy")
    neural_array = mat73.loadmat(NEURAL_PATH)["stim"].T
    emotion_array = scipy.io.loadmat(EMOTION_PATH)["resp"].flatten()

    X_full = torch.tensor(neural_array, dtype=torch.float32)
    y_full = torch.tensor(emotion_array, dtype=torch.float32).unsqueeze(1)
    full_neural = torch.load(FULL_NEURAL_PATH, map_location="cpu").float().contiguous()

    name_to_idx = {nm: i for i, nm in enumerate(ELECTRODE_NAMES)}

    # ---- enumerate all node combinations (including self-pairs) ----
    pairs = list(combinations_with_replacement(NODE_MAP.keys(), 2))
    if args.pairs_limit > 0:
        pairs = pairs[:args.pairs_limit]

    # ---- training loop ----
    for A, B in pairs:
        # skip missing electrodes
        if any(electrode_is_missing(neural_array, name_to_idx[nm], N_ELECTRODES, N_BANDS, N_LAGS)
               for nm in NODE_MAP[A]) or any(electrode_is_missing(neural_array, name_to_idx[nm], N_ELECTRODES, N_BANDS, N_LAGS)
               for nm in NODE_MAP[B]):
            print(f"[SKIP] {A}__{B}: missing electrode(s)")
            continue

        # ---- collect unique electrode indices ----
        elec_idx = np.unique([name_to_idx[nm] for nm in (NODE_MAP[A] + NODE_MAP[B])])
        feat_idx = feat_indices_for_electrodes(elec_idx, N_ELECTRODES, N_BANDS, N_LAGS)

        # ---- subset data ----
        X_train = X_full[train_idx][:, feat_idx]
        y_train = y_full[train_idx]

        # ---- dataset + config ----
        ds = DatasetxCEBRA(neural=X_train, position=y_train)
        loader = ContrastiveMultiObjectiveLoader(ds, batch_size=args.batch_size, num_steps=args.num_steps)
        config = build_cebra_config(loader, BEHAVIOR_INDICES)

        # ---- train model from scratch ----
        model = init_model("offset10-model", num_neurons=X_train.shape[1],
                           num_units=256, num_output=N_LATENTS).to(device)
        ds.configure_for(model)

        # ---- metadata + output dir ----
        meta = {
            "patient_id": patient_id,
            "mode": "single" if A == B else "pair",
            "pair": [A] if A == B else [A, B],
            "electrodes": NODE_MAP[A] if A == B else (NODE_MAP[A] + NODE_MAP[B]),
            "feat_idx": feat_idx.tolist()
        }

        out_dir = out_root / (f"{A}" if A == B else f"{A}__{B}")

        # ---- train and save ----
        train_and_save(model, loader, config, out_dir,
                       full_neural_tensor=full_neural,
                       meta=meta, device=device, num_steps=args.num_steps)
    
    # --- Null shuffled baseline ---
    print("[NULL] training shuffled baseline...")
    y_shuffled = y_full[torch.randperm(len(y_full))]
    ds_null = DatasetxCEBRA(neural=X_full[train_idx], position=y_shuffled[train_idx])
    loader_null = ContrastiveMultiObjectiveLoader(ds_null, batch_size=args.batch_size, num_steps=args.num_steps)
    config_null = build_cebra_config(loader_null, BEHAVIOR_INDICES)
    model_null = init_model("offset10-model", num_neurons=X_train.shape[1], num_units=256, num_output=N_LATENTS).to(device)
    ds_null.configure_for(model_null)
    null_dir = out_root.parent / "null_model"
    train_and_save(model_null, loader_null, config_null, null_dir,
                full_neural_tensor=full_neural,
                meta={"patient_id": patient_id, "type": "null"},
                device=device, num_steps=args.num_steps)


    print(f"[DONE] Combined single + pair + null training for patient {patient_id}")


if __name__ == "__main__":
    main()
