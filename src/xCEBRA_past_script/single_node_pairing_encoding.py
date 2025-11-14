import os, argparse, json
from pathlib import Path
from itertools import combinations
import numpy as np
import torch, mat73, scipy.io
import cebra
from cebra.data import DatasetxCEBRA, ContrastiveMultiObjectiveLoader
from cebra.models import init as init_model

from src.config import (
    MODEL_DIR, NEURAL_PATH, EMOTION_PATH, N_LATENTS,
    BEHAVIOR_INDICES, ELECTRODE_NAMES, NODE_MAP, PATIENT_CONFIG,
    FULL_NEURAL_PATH, FULL_EMOTION_PATH
)
from src.utils import load_fixed_cebra_model
from src.utils_training import feat_indices_for_electrodes, electrode_is_missing, FeatureMaskWrapper, freeze_backbone_unfreeze_head, build_cebra_config, train_and_save

N_ELECTRODES, N_BANDS, N_LAGS = len(ELECTRODE_NAMES), 5, 5
N_FEATS = N_ELECTRODES * N_BANDS * N_LAGS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_steps", type=int, default=1000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pairs_limit", type=int, default=-1)
    args = ap.parse_args()
    device = torch.device(args.device)

    pid = int(float(os.environ["PATIENT_ID"]))
    _, patient_id = PATIENT_CONFIG[pid]
    out_root = Path(f"./output_xCEBRA_lags/{patient_id}/pair_nodes")
    out_root.mkdir(parents=True, exist_ok=True)

    train_idx = np.load(MODEL_DIR / "train_idx.npy")
    neural_array = mat73.loadmat(NEURAL_PATH)["stim"].T
    emotion_array = scipy.io.loadmat(EMOTION_PATH)["resp"].flatten()
    print(emotion_array.shape)
    X_full = torch.tensor(neural_array, dtype=torch.float32)
    y_full = torch.tensor(emotion_array, dtype=torch.float32).unsqueeze(1)

    full_neural = torch.load(FULL_NEURAL_PATH, map_location="cpu").float().contiguous()

    # --- Global model ---
    model_global = load_fixed_cebra_model(MODEL_DIR / "xcebra_weights.pt",
                                          name="offset10-model",
                                          num_units=256, num_output=N_LATENTS,
                                          num_neurons=N_FEATS)

    # --- Null shuffled baseline ---
    print("[NULL] training shuffled baseline...")
    y_shuffled = y_full[torch.randperm(len(y_full))]
    ds_null = cebra.data.DatasetxCEBRA(neural=X_full[train_idx], position=y_shuffled[train_idx])
    loader_null = ContrastiveMultiObjectiveLoader(ds_null, batch_size=args.batch_size, num_steps=args.num_steps)
    config_null = build_cebra_config(loader_null, BEHAVIOR_INDICES)
    model_null = init_model("offset10-model", num_neurons=N_FEATS, num_units=256, num_output=N_LATENTS).to(device)
    ds_null.configure_for(model_null) #dataset reads the model’s parameters and adjusts to be compatible with that model’s input/output shapes.
    null_dir = out_root.parent / "null_model"
    train_and_save(model_null, loader_null, config_null, null_dir,
                   full_neural_tensor=full_neural,
                   meta={"patient_id": patient_id, "type": "null"},
                   device=device, num_steps=args.num_steps)

    # --- Pairwise loop ---
    name_to_idx = {nm: i for i, nm in enumerate(ELECTRODE_NAMES)}
    pairs = list(combinations(NODE_MAP.keys(), 2))
    if args.pairs_limit > 0:
        pairs = pairs[:args.pairs_limit]

    for A, B in pairs:
        # skip if either node missing
        if any(electrode_is_missing(neural_array, name_to_idx[nm], N_ELECTRODES, N_BANDS, N_LAGS) for nm in NODE_MAP[A]) \
           or any(electrode_is_missing(neural_array, name_to_idx[nm], N_ELECTRODES, N_BANDS, N_LAGS) for nm in NODE_MAP[B]):
            print(f"[SKIP] {A}__{B}: one or both nodes missing → skipped")
            continue

        elec_idx = [name_to_idx[nm] for nm in (NODE_MAP[A] + NODE_MAP[B])]
        feat_idx = feat_indices_for_electrodes(elec_idx, N_ELECTRODES, N_BANDS, N_LAGS)
        mask = torch.zeros(N_FEATS, dtype=torch.bool); mask[feat_idx] = True

        model_pair = init_model("offset10-model", num_neurons=N_FEATS, num_units=256, num_output=N_LATENTS).to(device)
        model_pair.load_state_dict(model_global.state_dict())
        freeze_backbone_unfreeze_head(model_pair)
        masked_model = FeatureMaskWrapper(model_pair, mask.to(device))

        ds_pair = cebra.data.DatasetxCEBRA(neural=X_full[train_idx], position=y_full[train_idx])
        loader_pair = ContrastiveMultiObjectiveLoader(ds_pair, batch_size=args.batch_size, num_steps=150)  # hardcoded
        config_pair = build_cebra_config(loader_pair, BEHAVIOR_INDICES)
        ds_pair.configure_for(masked_model)

        meta = {"patient_id": patient_id, "pair": [A, B], "feature_indices": feat_idx.tolist()}
        pair_dir = out_root / f"{A}__{B}"
        train_and_save(masked_model, loader_pair, config_pair, pair_dir,
                       full_neural_tensor=full_neural,
                       meta=meta, device=device, num_steps=args.num_steps)

    print(f"[DONE] Pairwise training for patient {patient_id}")


if __name__ == "__main__":
    main()
