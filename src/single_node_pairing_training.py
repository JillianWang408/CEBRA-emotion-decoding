#still use lag data, but taking two node into consideration at one training, use backbone-freezing training to minimize computation

import os
import json
import argparse
from pathlib import Path
from itertools import combinations

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
from cebra.models.model import Model
from src.utils import load_fixed_cebra_model

from src.config import (
    MODEL_DIR, NEURAL_PATH, EMOTION_PATH,
    N_LATENTS, BEHAVIOR_INDICES,
    ELECTRODE_NAMES, NODE_MAP, PATIENT_CONFIG,
    FULL_NEURAL_PATH, FULL_EMOTION_PATH
)

# ---- DC5 constants ----
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


class FeatureMaskWrapper(Model):
    def __init__(self, model, mask):
        super().__init__(num_input=model.num_input, num_output=model.num_output)
        self.model = model
        self.register_buffer("mask", mask.view(1, -1, 1).float())

    def forward(self, x):
        # print(">>> DEBUG: x.shape =", x.shape)
        # print(">>> DEBUG: mask.shape =", self.mask.shape)
        x = x * self.mask  
        return self.model(x)

    def get_offset(self):
        return self.model.get_offset()

    @property
    def split_outputs(self):
        return self.model.split_outputs

    @split_outputs.setter
    def split_outputs(self, value):
        self.model.split_outputs = value


def freeze_backbone_unfreeze_head(model):
    """Freeze backbone layers, unfreeze projection head."""
    for p in model.parameters():
        p.requires_grad = False
    # Find last parameter group (projection head)
    last_prefix = None
    for n, _ in model.named_parameters():
        last_prefix = n.rsplit('.', 1)[0]  # e.g. "net.5"

    # Unfreeze everything under that last group
    for n, p in model.named_parameters():
        if n.startswith(last_prefix):
            p.requires_grad = True
            print(f"[UNFREEZE] {n}")
    #If you change the architecture (e.g. use "offset20-model" or custom configs), 
    # the last group might not strictly be the projection head.

def main():
    ap = argparse.ArgumentParser(description="Pairwise xCEBRA fine-tune with frozen backbone.")
    ap.add_argument("--feature-mode", choices=["lags", "cov"], default="lags",
                help="Feature structure (forwarded from main).")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_steps", type=int, default=1000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pairs_limit", type=int, default=-1)
    args = ap.parse_args()

    device = torch.device(args.device)
    pid = int(float(os.environ["PATIENT_ID"]))
    _, numeric_code = PATIENT_CONFIG[pid]
    patient_id = numeric_code

    out_root = Path(f"./output_xCEBRA_lags/{patient_id}/pair_nodes")
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Patient={patient_id} | Using global model from {MODEL_DIR/'xcebra_weights.pt'}")

    # --- Load data ---
    train_idx = np.load(MODEL_DIR / "train_idx.npy")
    neural_array = mat73.loadmat(NEURAL_PATH)["stim"].T
    emotion_array = scipy.io.loadmat(EMOTION_PATH)["resp"].flatten()

    # Torch tensors
    X_full = torch.tensor(neural_array, dtype=torch.float32)
    y_full = torch.tensor(emotion_array, dtype=torch.float32).unsqueeze(1)

    # Cached full tensors if available
    if Path(FULL_NEURAL_PATH).exists():
        neural_tensor_full = torch.load(FULL_NEURAL_PATH, map_location="cpu").float().contiguous()
        emotion_tensor_full = torch.load(FULL_EMOTION_PATH, map_location="cpu")
        if emotion_tensor_full.ndim == 1:
            emotion_tensor_full = emotion_tensor_full.unsqueeze(1)
        emotion_tensor_full = emotion_tensor_full.float().contiguous()
    else:
        neural_tensor_full, emotion_tensor_full = X_full, y_full

    # --- Load pretrained global model ---
    model_global = load_fixed_cebra_model(MODEL_DIR / "xcebra_weights.pt", 
                                      name="offset10-model",
                                      num_units=256,
                                      num_output=N_LATENTS,
                                      num_neurons=N_FEATS)

    # --- Null sanity model ---
    print("[NULL] training shuffled baseline...")
    y_shuffled = y_full[torch.randperm(len(y_full))]
    ds_null = DatasetxCEBRA(neural=X_full[train_idx], position=y_shuffled[train_idx])
    loader_null = ContrastiveMultiObjectiveLoader(ds_null, batch_size=args.batch_size, num_steps=args.num_steps)

    config_null = MultiObjectiveConfig(loader_null)
    config_null.set_slice(*BEHAVIOR_INDICES)
    config_null.set_loss("FixedCosineInfoNCE", temperature=1.0)
    config_null.set_distribution("time_delta", time_delta=1, label_name="position")
    config_null.push(); config_null.finalize()
    criterion_null = config_null.criterion
    feature_ranges_null = config_null.feature_ranges  # <- iterable of slices

    model_null = init_model(
        name="offset10-model",
        num_neurons=N_FEATS,      # 1000
        num_units=256,            # <- match your global checkpoint width
        num_output=N_LATENTS
    ).to(device)

    ds_null.configure_for(model_null)
    opt_null = torch.optim.Adam(
    list(model_null.parameters()) + list(criterion_null.parameters()),
    lr=3e-4, weight_decay=0.0
    )

    solver_null = cebra.solver.init(
        name="multiobjective-solver",
        model=model_null,
        feature_ranges=feature_ranges_null,
        regularizer=JacobianReg(),
        renormalize=True,
        use_sam=False,
        criterion=criterion_null,
        optimizer=opt_null,
        tqdm_on=True
    ).to(device)

    weight_scheduler_null = LinearRampUp(
        n_splits=1,
        step_to_switch_on_reg=max(1, args.num_steps // 4),
        step_to_switch_off_reg=max(2, args.num_steps // 2),
        start_weight=0.0,
        end_weight=0.1
    )

    solver_null.fit(loader=loader_null, valid_loader=None, scheduler_regularizer=weight_scheduler_null)

    emb_null = model_null(neural_tensor_full.T.unsqueeze(0).to(device)).detach().cpu()

    null_dir = out_root.parent / "null_model"
    null_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model_null.state_dict(), null_dir / "model_weights.pt")
    torch.save(emb_null, null_dir / "embedding.pt")
    (null_dir / "meta.json").write_text(json.dumps({"patient_id": patient_id, "type": "null"}, indent=2))
    print(f"[NULL] saved to {null_dir}")

    # --- Pairwise fine-tune ---
    name_to_idx = {nm: i for i, nm in enumerate(ELECTRODE_NAMES)}
    pairs = list(combinations(NODE_MAP.keys(), 2))
    if args.pairs_limit > 0:
        pairs = pairs[:args.pairs_limit]

    for A, B in pairs:
        elec_idx = [name_to_idx[nm] for nm in (NODE_MAP[A] + NODE_MAP[B])]
        feat_idx = feat_indices_for_electrodes(elec_idx)

        # Boolean mask over 1000 features
        mask = torch.zeros(N_FEATS, dtype=torch.bool)
        mask[feat_idx] = True

        # Start from global weights
        model_pair = init_model(
            name="offset10-model",
            num_neurons=N_FEATS,
            num_units=256,          # must match global
            num_output=N_LATENTS
        ).to(device)
        model_pair.load_state_dict(model_global.state_dict())

        # Freeze backbone, unfreeze head
        freeze_backbone_unfreeze_head(model_pair)

        # Wrap with the feature mask
        masked_model = FeatureMaskWrapper(model_pair, mask.to(device))

        # Dataset & loader (train split only)
        ds_pair = DatasetxCEBRA(neural=X_full[train_idx], position=y_full[train_idx])
        loader_pair = ContrastiveMultiObjectiveLoader(
            dataset=ds_pair, batch_size=args.batch_size, num_steps=150
        )

        # Config (match your single-node format)
        config_pair = MultiObjectiveConfig(loader_pair)
        config_pair.set_slice(*BEHAVIOR_INDICES)
        config_pair.set_loss("FixedCosineInfoNCE", temperature=1.0)
        config_pair.set_distribution("time_delta", time_delta=1, label_name="position")
        config_pair.push()
        config_pair.finalize()

        criterion_pair = config_pair.criterion
        feature_ranges_pair = config_pair.feature_ranges

        # Configure dataset for the masked model
        ds_pair.configure_for(masked_model)

        # Optimizer over masked model + criterion params
        opt_pair = torch.optim.Adam(
            list(masked_model.parameters()) + list(criterion_pair.parameters()),
            lr=3e-4, weight_decay=0.0
        )

        # Solver init (keywords, not positional)
        solver_pair = cebra.solver.init(
            name="multiobjective-solver",
            model=masked_model,
            feature_ranges=feature_ranges_pair,
            regularizer=JacobianReg(),
            renormalize=True,
            use_sam=False,
            criterion=criterion_pair,
            optimizer=opt_pair,
            tqdm_on=True
        ).to(device)

        # Optional: scheduler for consistency
        weight_scheduler_pair = LinearRampUp(
            n_splits=1,
            step_to_switch_on_reg=max(1, args.num_steps // 4),
            step_to_switch_off_reg=max(2, args.num_steps // 2),
            start_weight=0.0,
            end_weight=0.1
        )

        # Train
        print(f"[PAIR] {A}__{B}: feats={len(feat_idx)} | training head-only...")
        solver_pair.fit(loader=loader_pair, valid_loader=None, scheduler_regularizer=weight_scheduler_pair)

        # Save outputs
        pair_dir = out_root / f"{A}__{B}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        emb_pair = masked_model(neural_tensor_full.T.unsqueeze(0).to(device)).detach().cpu()
        torch.save(model_pair.state_dict(), pair_dir / "model_weights.pt")
        torch.save(emb_pair, pair_dir / "embedding.pt")

        meta = {
            "patient_id": patient_id,
            "pair": [A, B],
            "feature_indices": feat_idx.tolist(),
            "num_steps": int(args.num_steps),
            "batch_size": int(args.batch_size),
            "freeze_backbone": True
        }
        (pair_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[PAIR] {A}__{B} saved → {pair_dir}")

        print("[DONE] All pairs processed.")

if __name__ == "__main__":
    main()
