from __future__ import annotations
import torch
from cebra.models import init as init_model
from src.config import NEURAL_TENSOR_PATH
import numpy as np
from pathlib import Path

def load_fixed_cebra_model(model_path, name="offset10-model",
                           num_units=256, num_output=20, num_neurons=None):
    # Load weights first so we can infer when needed
    raw_state_dict = torch.load(model_path, map_location="cpu")

    # Fix keys (your original logic)
    fixed_state_dict = {}
    for k, v in raw_state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        k = k.replace(".module.", ".")
        if k.startswith("net.2.0"):
            k = k.replace("net.2.0", "net.2.module.0")
        elif k.startswith("net.3.0"):
            k = k.replace("net.3.0", "net.3.module.0")
        elif k.startswith("net.4.0"):
            k = k.replace("net.4.0", "net.4.module.0")
        fixed_state_dict[k] = v

    # If caller didn't specify, infer input channels from first Conv1d weight
    if num_neurons is None:
        inferred = None
        for k, v in fixed_state_dict.items():
            if k.endswith("weight") and v.ndim == 3:  # Conv1d: [out_ch, in_ch, k]
                inferred = int(v.shape[1])
                break
        if inferred is None:
            raise RuntimeError("Could not infer num_neurons from state_dict; please pass num_neurons explicitly.")
        num_neurons = inferred

    # Build the correct-arity model and load weights
    model = init_model(name=name, num_neurons=num_neurons,
                       num_units=num_units, num_output=num_output)
    model.load_state_dict(fixed_state_dict)
    model.eval()
    return model

# ---------- Kernel utilities ----------


def rbf_1d_grid(n: int, lengthscale: float, variance: float = 1.0, jitter: float = 1e-6) -> np.ndarray:
    """
    RBF kernel on a 1D integer grid {0,1,...,n-1}.
    K_ij = variance * exp( - (i-j)^2 / (2*ell^2) ) + jitter * δ_ij
    """
    idx = np.arange(n, dtype=np.float32)[:, None]    # [n,1]
    d2  = (idx - idx.T) ** 2                          # [n,n]
    ell2 = (lengthscale if lengthscale > 0 else 1e-6) ** 2
    K = variance * np.exp(-0.5 * d2 / ell2)
    # numerical stability
    K.flat[:: n + 1] += jitter
    return K.astype(np.float32)

def make_kernel_lag_major(
    n_electrodes: int,
    n_bands: int,
    n_lags: int,
    ell_lag: float = 1.0,
    var_lag: float = 1.0,
    jitter: float = 1e-6,
    save_path: Path | None = None,
) -> np.ndarray:
    """
    Build feature prior kernel for LAG-MAJOR layout:
        flat_idx(l,e,b) = l*(E*B) + e*B + b
    We impose correlation across lags for the SAME channel (e,b),
    and independence across different channels:
        K = K_lag ⊗ I_channels   where channels = E*B
    """
    E, B, L = n_electrodes, n_bands, n_lags
    C = E * B                       # channels per lag
    K_lag = rbf_1d_grid(L, lengthscale=ell_lag, variance=var_lag, jitter=jitter)   # [L,L]
    K_ch  = np.eye(C, dtype=np.float32)                                           # [C,C]
    K = np.kron(K_lag, K_ch).astype(np.float32)                                   # [L*C, L*C] = [F,F]

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, K)
    return K

def load_or_make_kernel(
    n_electrodes: int,
    n_bands: int,
    n_lags: int,
    ell_lag: float,
    var_lag: float,
    cache_dir: Path,
    name: str = "feature_kernel_lagmajor.npy",
) -> np.ndarray:
    """
    Cache helper: reuse kernel if saved; else build & save.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / name
    if p.exists():
        return np.load(p)
    return make_kernel_lag_major(n_electrodes, n_bands, n_lags, ell_lag, var_lag, save_path=p)
