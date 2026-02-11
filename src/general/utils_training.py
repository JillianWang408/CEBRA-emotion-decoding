import json
from pathlib import Path
import numpy as np
import torch
import cebra
from cebra.solver import MultiObjectiveConfig
from cebra.solver.schedulers import LinearRampUp
from cebra.models.jacobian_regularizer import JacobianReg
from cebra.models.model import Model
import matplotlib.pyplot as plt

def make_and_save_split(T: int, out_dir: Path, train_frac: float = 0.8, seed: int = 0):
    """
    Generate a reproducible train/test split for length T and save into out_dir.
    
    Args:
        T: total number of time steps
        out_dir: directory to save train_idx.npy / test_idx.npy
        train_frac: fraction of data for training
        seed: RNG seed for reproducibility
    Returns:
        train_idx, test_idx (as numpy arrays)
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(T)
    rng.shuffle(idx)

    split = int(train_frac * T)
    train_idx, test_idx = idx[:split], idx[split:]

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "train_idx.npy", train_idx)
    np.save(out_dir / "test_idx.npy", test_idx)

    print(f"[INFO] Saved train/test split to {out_dir} "
          f"(train={len(train_idx)}, test={len(test_idx)}, total={T})")

    return train_idx, test_idx

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


class FeatureMaskWrapper(Model): #keep the same model structure but zero out unwanted features.
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


def build_cebra_config_unsupervised(loader, behavior_indices=None, temperature=1.0):
    config = MultiObjectiveConfig(loader)
    if behavior_indices is not None:
        try:
            config.set_slice(*behavior_indices)
        except Exception as e:
            print(f"[WARN] Ignoring behavior_indices={behavior_indices}: {e}")
    config.set_loss("FixedCosineInfoNCE", temperature=temperature)
    config.set_distribution("time", time_offset=1)
    config.push(); config.finalize()
    return config

def build_cebra_config_supervised(loader, behavior_indices=None, temperature=1.0):
    config = MultiObjectiveConfig(loader)
    if behavior_indices is not None:
        try:
            config.set_slice(*behavior_indices)
        except Exception as e:
            print(f"[WARN] Ignoring behavior_indices={behavior_indices}: {e}")
    config.set_loss("FixedCosineInfoNCE", temperature=temperature)
    config.set_distribution("time_delta", time_delta=1, label_name="position")
    config.push(); config.finalize()
    return config

def train_and_save(model, loader, config, out_dir: Path,
                   full_neural_tensor=None, meta: dict = None,
                   device=None, num_steps=1000):
    """Generic training loop with saving of weights, embeddings, and metadata."""
    opt = torch.optim.Adam(
        list(model.parameters()) + list(config.criterion.parameters()), 
        #all weights of your encoder network
        # parameters of the InfoNCE loss (how “strict” the contrastive similarity measure is)
        lr=3e-4, weight_decay=0.0 #learning rate, L2 penalty on weights-regularization to prevent overfitting (disabled here)
    )

    solver = cebra.solver.init(
        name="multiobjective-solver", #optimize multiple objectives at once(InfoNCE loss and regularization)
        model=model,
        feature_ranges=config.feature_ranges,
        regularizer=JacobianReg(),#smoothness penalty on embedding
        renormalize=True,#L2-normalize embeddings after each step
        use_sam=False, #Sharpness-Aware Minimization (SAM) optimizer (disabled here), standard gradient descent instead
        criterion=config.criterion, #InfoNCE loss function, defines what “similar” vs “dissimilar” means
        optimizer=opt,
        tqdm_on=True
    ).to(device)

    scheduler = LinearRampUp( #the model first learns alignment, then smooths the learned structure.
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
            X = full_neural_tensor[:, feat_idx].T.unsqueeze(0).to(device)

        else:
            X = full_neural_tensor.T.unsqueeze(0).to(device)

        F = model.num_input
        print("F", F)
        if X.shape[0] != 1:
            raise ValueError(f"Expected batch=1 for 3D input, got {tuple(X.shape)}")
        if X.shape[1] == F:         # already (1,F,T)
            batch = X.to(device)
        elif X.shape[2] == F:       # (1,T,F) -> swap last two dims
            batch = X.permute(0, 2, 1).contiguous().to(device)
            print(batch.shape)
        else:
            raise ValueError(f"3D shape incompatible with channels {F}: {tuple(X.shape)}")

        emb = model(batch).detach().cpu()
        torch.save(emb, out_dir / "embedding.pt")


    if meta:
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    return solver

#Embedding plotting (embedding ahve less tinesteps than original data due to conv/offset, need to align)

def plot_embedding_split(Z, y_aligned, split, out_dir: Path, prefix: str, title_prefix: str):
    """
    Generate and save interactive Plotly embeddings for train/test splits.
    """
    # Interactive Plotly visualizations
    fig_train = cebra.integrations.plotly.plot_embedding_interactive(
        Z[:split], embedding_labels=y_aligned[:split],
        title=f"{title_prefix} (train)", markersize=3, cmap="tab10"
    )
    fig_test = cebra.integrations.plotly.plot_embedding_interactive(
        Z[split:], embedding_labels=y_aligned[split:],
        title=f"{title_prefix} (test)", markersize=3, cmap="tab10"
    )

    # Save to HTML
    fig_train.write_html(out_dir / f"{prefix}_train_interactive.html", include_plotlyjs="embed")
    fig_test.write_html(out_dir / f"{prefix}_test_interactive.html", include_plotlyjs="embed")

    print(f"[PLOT] Saved interactive embeddings: {prefix}_train/test_interactive.html")