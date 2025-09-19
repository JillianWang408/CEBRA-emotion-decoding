import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import mat73
import scipy.io
import cebra
from cebra.data import DatasetxCEBRA, ContrastiveMultiObjectiveLoader
from cebra.solver import MultiObjectiveConfig
from cebra.solver.schedulers import LinearRampUp
from cebra.models import init as init_model
from cebra.models.jacobian_regularizer import JacobianReg

from src.utils import load_fixed_cebra_model
from src.config import (
    MODEL_DIR, NEURAL_PATH, EMOTION_PATH,
    N_LATENTS, BEHAVIOR_INDICES,
    ELECTRODE_NAMES, NODE_MAP, PATIENT_CONFIG, FULL_NEURAL_PATH, FULL_EMOTION_PATH
)

# ---- DC5 / lags layout constants ----
N_ELECTRODES = len(ELECTRODE_NAMES)  # 40
N_BANDS = 5
N_LAGS  = 5
F_PER_LAG = N_ELECTRODES * N_BANDS   # 200
N_FEATS   = N_LAGS * F_PER_LAG       # 1000

def feat_indices_for_electrodes(elec_indices):
    """Return DC5 column indices for given electrode indices across all lags & bands."""
    idxs = []
    for lag in range(N_LAGS):
        base_lag = lag * F_PER_LAG
        for e in elec_indices:
            base_e = base_lag + e * N_BANDS
            for b in range(N_BANDS):
                idxs.append(base_e + b)
    return np.array(idxs, dtype=int)

def main():
    ap = argparse.ArgumentParser(description="Train xCEBRA per node (lags/DC5), full-node coverage required.")
    ap.add_argument("--feature-mode", choices=["lags"], default="lags",
                    help="Only 'lags' (DC5) is supported here.")
    # You can tweak these defaults as needed
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_steps", type=int, default=1000)
    args = ap.parse_args()

    pid = int(float(os.environ["PATIENT_ID"]))  # handles "27" or "27.0"
    ec_code, numeric_code = PATIENT_CONFIG[pid]  # e.g. ("EC301", "301")
    patient_id = numeric_code
    out_root = Path(f"./output_xCEBRA_lags/{patient_id}/single_node")
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Patient={patient_id} | Saving to {out_root}")

    # --- Load data ---
    train_idx = np.load(MODEL_DIR / "train_idx.npy")
    neural_array = mat73.loadmat(NEURAL_PATH)["stim"].T                 # (T, 1000)
    emotion_array = scipy.io.loadmat(EMOTION_PATH)["resp"].flatten()    # (T,)

    assert neural_array.ndim == 2 and neural_array.shape[1] == N_FEATS, \
        f"Expected DC5 shape (*,{N_FEATS}), got {neural_array.shape}"

    # --- Helpers ---
    name_to_idx = {nm: i for i, nm in enumerate(ELECTRODE_NAMES)}

    def electrode_is_missing(e_idx: int) -> bool:
        """Full-coverage test: electrode considered missing if all its features are zero across time."""
        col_idx = feat_indices_for_electrodes([e_idx])
        sub = neural_array[:, col_idx]
        return np.allclose(sub, 0.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Iterate nodes (full coverage REQUIRED) ---
    for node_name, name_list in NODE_MAP.items():
        # Map names → indices
        try:
            elec_indices = [name_to_idx[nm] for nm in name_list]
        except KeyError as e:
            raise SystemExit(f"[{node_name}] Unknown electrode in NODE_MAP: {e}")

        # Full-node coverage gate
        missing = [e for e in elec_indices if electrode_is_missing(e)]
        if missing:
            miss_names = [ELECTRODE_NAMES[e] for e in missing]
            print(f"[SKIP] {node_name}: missing electrodes {miss_names} (full-node coverage required).")
            continue

        feat_idx = feat_indices_for_electrodes(elec_indices)  # len == 4*5*5 == 100
        X_full = neural_array[:, feat_idx]                    # (T, 100)
        y_full = emotion_array                                 # (T,)

        # Train tensors (train split only)
        X_train = torch.tensor(X_full[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y_full[train_idx], dtype=torch.float32).unsqueeze(1)

        # === Dataset & Loader ===
        datasets = DatasetxCEBRA(neural=X_train, position=y_train)
        loader = ContrastiveMultiObjectiveLoader(
            dataset=datasets, batch_size=args.batch_size, num_steps=args.num_steps
        )

        # === Config ===
        config = MultiObjectiveConfig(loader)
        config.set_slice(*BEHAVIOR_INDICES)
        config.set_loss("FixedCosineInfoNCE", temperature=1.0)
        config.set_distribution("time_delta", time_delta=1, label_name="position")
        config.push()
        config.finalize()

        criterion = config.criterion
        feature_ranges = config.feature_ranges

        # === Model sized to node ===
        model = init_model(
            name="offset10-model",
            num_neurons=datasets.neural.shape[1],  # should be 100
            num_units=256,
            num_output=N_LATENTS
        ).to(device)

        datasets.configure_for(model)

        # Optimizer / Solver / Scheduler
        opt = torch.optim.Adam(
            list(model.parameters()) + list(criterion.parameters()),
            lr=3e-4, weight_decay=0.0
        )
        solver = cebra.solver.init(
            name="multiobjective-solver",
            model=model,
            feature_ranges=feature_ranges,
            regularizer=JacobianReg(),
            renormalize=True,
            use_sam=False,
            criterion=criterion,
            optimizer=opt,
            tqdm_on=True
        ).to(device)

        weight_scheduler = LinearRampUp(
            n_splits=1,
            step_to_switch_on_reg=args.num_steps // 4,
            step_to_switch_off_reg=args.num_steps // 2,
            start_weight=0.0,
            end_weight=0.1
        )

        # === Train ===
        print(f"[TRAIN] {node_name}: electrodes={name_list} -> idx={elec_indices} -> feats={len(feat_idx)}")
        solver.fit(loader=loader, valid_loader=None, scheduler_regularizer=weight_scheduler)

        # === Save per-node artifacts ===
        node_dir = out_root / node_name
        node_dir.mkdir(parents=True, exist_ok=True)

        # Weights
        torch.save(solver.model.state_dict(), node_dir / "model_weights.pt")

        # ------------ Embedding on FULL dataset using the SINGLE-NODE model --------------

        # Load cached full tensors
        neural_tensor_full  = torch.load(FULL_NEURAL_PATH,  map_location="cpu").float().contiguous()   # [T, 1000]
        emotion_tensor_full = torch.load(FULL_EMOTION_PATH, map_location="cpu")
        if emotion_tensor_full.ndim == 1:
            emotion_tensor_full = emotion_tensor_full.unsqueeze(1)
        emotion_tensor_full = emotion_tensor_full.float().contiguous()                                 # [T, 1]

        # Slice to this node (ensure torch.long index)
        feat_idx_t = torch.as_tensor(feat_idx, dtype=torch.long)
        neural_tensor_node = neural_tensor_full.index_select(1, feat_idx_t)      
        model.eval(); model.split_outputs = False                      # [T, 100]
        datasets_full = DatasetxCEBRA(neural=neural_tensor_node, continuous=emotion_tensor_full)

        datasets_full.configure_for(model)
        batch = datasets_full[torch.arange(len(datasets_full))]
        emb = model(batch.to(device)).detach().cpu()
        torch.save(emb, node_dir / "embedding.pt")

        # Metadata
        meta = {
            "patient_id": patient_id,
            "feature_mode": args.feature_mode,
            "node": node_name,
            "electrodes_names": name_list,
            "electrodes_indices": elec_indices,
            "feature_indices": feat_idx.tolist(),
            "train_idx_path": str((MODEL_DIR / "train_idx.npy").resolve()),
            "neural_source": str(NEURAL_PATH),
            "emotion_source": str(EMOTION_PATH),
            "n_latents": int(N_LATENTS),
            "num_steps": int(args.num_steps),
            "batch_size": int(args.batch_size),
            "embedding_shape": list(emb.shape)
        }
        (node_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        print(f"[SAVE] {node_name}: weights → {node_dir/'model_weights.pt'} | embedding → {node_dir/'embedding.pt'}")

    print(f"[DONE] Single-node training complete for patient {patient_id}. Root: {out_root}")

if __name__ == "__main__":
    main()
