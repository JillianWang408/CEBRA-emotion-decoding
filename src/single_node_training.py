import os, argparse
from pathlib import Path
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
from src.utils_training import feat_indices_for_electrodes, electrode_is_missing, build_cebra_config, train_and_save

N_ELECTRODES, N_BANDS, N_LAGS = len(ELECTRODE_NAMES), 5, 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_steps", type=int, default=1000)
    args = ap.parse_args()

    pid = int(float(os.environ["PATIENT_ID"]))
    _, patient_id = PATIENT_CONFIG[pid]
    out_root = Path(f"./output_xCEBRA_lags/{patient_id}/single_node")
    out_root.mkdir(parents=True, exist_ok=True)

    train_idx = np.load(MODEL_DIR / "train_idx.npy")
    neural_array = mat73.loadmat(NEURAL_PATH)["stim"].T
    emotion_array = scipy.io.loadmat(EMOTION_PATH)["resp"].flatten()

    name_to_idx = {nm: i for i, nm in enumerate(ELECTRODE_NAMES)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for node_name, name_list in NODE_MAP.items():
        elec_indices = [name_to_idx[nm] for nm in name_list]
        # full coverage required
        if any(electrode_is_missing(neural_array, e, N_ELECTRODES, N_BANDS, N_LAGS) for e in elec_indices):
            print(f"[SKIP] {node_name}: missing electrodes → skipped")
            continue
        print(f"[TRAIN] Starting training for node: {node_name} (electrodes={name_list})")

        feat_idx = feat_indices_for_electrodes(elec_indices, N_ELECTRODES, N_BANDS, N_LAGS)
        X_train = torch.tensor(neural_array[train_idx][:, feat_idx], dtype=torch.float32)
        y_train = torch.tensor(emotion_array[train_idx], dtype=torch.float32).unsqueeze(1)

        ds = cebra.data.DatasetxCEBRA(neural=X_train, position=y_train)
        loader = ContrastiveMultiObjectiveLoader(ds, batch_size=args.batch_size, num_steps=args.num_steps)
        config = build_cebra_config(loader, BEHAVIOR_INDICES)

        model = init_model("offset10-model", num_neurons=X_train.shape[1],
                           num_units=256, num_output=N_LATENTS).to(device)
        ds.configure_for(model)

        full_neural = torch.load(FULL_NEURAL_PATH, map_location="cpu").float().contiguous()
        meta = {
            "patient_id": patient_id,
            "node": node_name,
            "electrodes": name_list,
            "feat_idx": feat_idx.tolist()
        }
        node_dir = out_root / node_name
        train_and_save(model, loader, config, node_dir,
                       full_neural_tensor=full_neural,
                       meta=meta, device=device, num_steps=args.num_steps)

    print(f"[DONE] Single-node training for patient {patient_id}")


if __name__ == "__main__":
    main()