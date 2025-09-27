import json
from pathlib import Path
import numpy as np
import torch
import cebra
from cebra.solver import MultiObjectiveConfig
from cebra.solver.schedulers import LinearRampUp
from cebra.models.jacobian_regularizer import JacobianReg
from cebra.models.model import Model


# ---- DC5 constants ----
def feat_indices_for_electrodes(elec_indices, n_electrodes=40, n_bands=5, n_lags=5):
    """Return DC5 column indices for given electrode indices across all lags & bands."""
    f_per_lag = n_electrodes * n_bands
    idxs = []
    for lag in range(n_lags):
        base_lag = lag * f_per_lag
        for e in elec_indices:
            base_e = base_lag + e * n_bands
            for b in range(n_bands):
                idxs.append(base_e + b)
    return np.array(idxs, dtype=int)


def electrode_is_missing(neural_array, e_idx, n_electrodes=40, n_bands=5, n_lags=5):
    """Check if electrode is completely missing (all zero features across time)."""
    col_idx = feat_indices_for_electrodes([e_idx], n_electrodes, n_bands, n_lags)
    sub = neural_array[:, col_idx]
    return np.allclose(sub, 0.0)


def build_cebra_config(loader, behavior_indices, temperature=1.0):
    """Setup a standard MultiObjectiveConfig for CEBRA training."""
    config = MultiObjectiveConfig(loader)
    config.set_slice(*behavior_indices)
    config.set_loss("FixedCosineInfoNCE", temperature=temperature)
    config.set_distribution("time_delta", time_delta=1, label_name="position")
    config.push()
    config.finalize()
    return config


def freeze_backbone_unfreeze_head(model):
    """Freeze backbone layers, unfreeze projection head."""
    for p in model.parameters():
        p.requires_grad = False
    last_prefix = None
    for n, _ in model.named_parameters():
        last_prefix = n.rsplit('.', 1)[0]
    for n, p in model.named_parameters():
        if n.startswith(last_prefix):
            p.requires_grad = True
            print(f"[UNFREEZE] {n}")


class FeatureMaskWrapper(Model):
    """Apply a boolean mask to input features before passing through model."""
    def __init__(self, model, mask):
        super().__init__(num_input=model.num_input, num_output=model.num_output)
        self.model = model
        self.register_buffer("mask", mask.view(1, -1, 1).float())

    def forward(self, x):
        return self.model(x * self.mask)

    def get_offset(self):
        return self.model.get_offset()

    @property
    def split_outputs(self):
        return self.model.split_outputs

    @split_outputs.setter
    def split_outputs(self, value):
        self.model.split_outputs = value


def train_and_save(model, loader, config, out_dir: Path,
                   full_neural_tensor=None, meta: dict = None,
                   device=None, num_steps=1000):
    """Generic training loop with saving of weights, embeddings, and metadata."""
    opt = torch.optim.Adam(
        list(model.parameters()) + list(config.criterion.parameters()),
        lr=3e-4, weight_decay=0.0
    )

    solver = cebra.solver.init(
        name="multiobjective-solver",
        model=model,
        feature_ranges=config.feature_ranges,
        regularizer=JacobianReg(),
        renormalize=True,
        use_sam=False,
        criterion=config.criterion,
        optimizer=opt,
        tqdm_on=True
    ).to(device)

    scheduler = LinearRampUp(
        n_splits=1,
        step_to_switch_on_reg=max(1, num_steps // 4),
        step_to_switch_off_reg=max(2, num_steps // 2),
        start_weight=0.0,
        end_weight=0.1
    )

    solver.fit(loader=loader, valid_loader=None, scheduler_regularizer=scheduler)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model_weights.pt")

    #Embeddings are run on the full dataset, but training itself uses only train_idx.
    if full_neural_tensor is not None: 
        model.eval(); model.split_outputs = False
        
        # Apply the same feature indices as during training
        if full_neural_tensor is not None and "feat_idx" in meta:
            feat_idx = torch.as_tensor(meta["feat_idx"], dtype=torch.long, device=device)
            batch = full_neural_tensor[:, feat_idx].T.unsqueeze(0).to(device)
        else:
            batch = full_neural_tensor.T.unsqueeze(0).to(device)

        emb = model(batch).detach().cpu()
        torch.save(emb, out_dir / "embedding.pt")


    if meta:
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    return solver
