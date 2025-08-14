import numpy as np
import scipy.io
import mat73
import json
import re
from pathlib import Path
import joblib
from src.config import (
    MODEL_DIR, NEURAL_PATH, EMOTION_PATH, N_ELECTRODES, DATA_DIR,
    FULL_NEURAL_PATH, FULL_EMOTION_PATH, NEURAL_TENSOR_PATH, EMOTION_TENSOR_PATH, 
    GDEC_HP
)

# ---------- JAX (CPU) + compatibility shim for gdec importing jax.config ----------
import os, sys, types, time, inspect
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
try:
    from jax.config import config
except Exception:
    m = types.ModuleType("jax.config"); m.config = jax.config; sys.modules["jax.config"] = m

import torch
from torch.optim import lr_scheduler as lrs

try:
    params = inspect.signature(lrs.ReduceLROnPlateau.__init__).parameters
except Exception:
    params = {}

if "verbose" not in params:
    class _ReduceLROnPlateauPatched(lrs.ReduceLROnPlateau):
        def __init__(self, optimizer, *args, **kwargs):
            kwargs.pop("verbose", None)
            super().__init__(optimizer, *args, **kwargs)
    lrs.ReduceLROnPlateau = _ReduceLROnPlateauPatched
    torch.optim.lr_scheduler.ReduceLROnPlateau = _ReduceLROnPlateauPatched

import gdec # safe after shim

# ---------- Layout constants ----------
N_LAGS, N_BANDS = 5, 5
N_FEATURES = N_LAGS * N_ELECTRODES * N_BANDS  # 1000 expected
FEATURE_ORDERING = "lag_first_then_band_within_each_electrode"

# Output dirs
OUT_DIR = MODEL_DIR / "gdec_gpmd"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FOLDS_DIR = DATA_DIR
FOLD_MODELS_DIR = OUT_DIR / "fold_models"  
FOLD_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Fold file swapping ----------
# Replace the FINAL “_<number>.mat” with desired fold_id (e.g., *_1.mat -> *_3.mat)
_fold_suffix_re = re.compile(r"_(\d+)\.mat$")

def _swap_fold(path: Path, fold_id: int) -> Path:
    new_name = _fold_suffix_re.sub(f"_{fold_id}.mat", path.name)
    return path.with_name(new_name)

# ---------- Data I/O ----------
def load_xy_from_fold(fold_id: int):
    """Load X,y from fold-specific MAT files."""
    X_path = _swap_fold(NEURAL_PATH, fold_id)
    y_path = _swap_fold(EMOTION_PATH, fold_id)
    X = mat73.loadmat(X_path)['stim'].T.astype(np.float32)           # [T, 1000]
    y = scipy.io.loadmat(y_path)['resp'].flatten().astype(np.int64)  # [T]
    assert X.shape[1] == N_FEATURES, f"Expected {N_FEATURES}, got {X.shape[1]} (fold {fold_id})"
    return X, y

def get_or_make_split_indices(fold_id: int, n: int, train_frac: float = 0.8):
    """
    Per-fold split; if saved, load; else create & save.
    Indices are RELATIVE to THIS fold’s order (do not reuse across folds).
    """
    fold_dir = FOLDS_DIR / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    p_train = fold_dir / "train_idx.npy"
    p_test  = fold_dir / "test_idx.npy"  # we call the held-out 20% 'test' for reuse downstream

    if p_train.exists() and p_test.exists():
        return np.load(p_train), np.load(p_test), fold_dir

    cut = int(n * train_frac)
    train_idx = np.arange(0, cut, dtype=np.int64)
    test_idx  = np.arange(cut, n, dtype=np.int64)
    np.save(p_train, train_idx)
    np.save(p_test,  test_idx)
    return train_idx, test_idx, fold_dir

# ---------- Labels ----------
def remap_to_dense(y_train):
    """
    Map labels present in y_train to 0..K_seen-1 for loss compatibility.
    Save/return classes_seen to map predictions back during evaluation.
    """
    classes_seen = np.unique(y_train)
    map_to_dense = {c: i for i, c in enumerate(classes_seen)}
    y_dense = np.vectorize(map_to_dense.get)(y_train).astype(np.int64)
    return y_dense, classes_seen

# ---------- Attribution reshape ----------
def reshape_coefs_for_attrib(coefs_flat: np.ndarray) -> np.ndarray:
    """
    [K_seen, 1000] -> [K_seen, 5, 40, 5] under lag-first-then-band within each electrode.
    """
    K, F = coefs_flat.shape
    assert F == N_FEATURES
    idx_map = np.zeros((N_LAGS, N_ELECTRODES, N_BANDS), dtype=int)
    for e in range(N_ELECTRODES):
        base_e = e * (N_BANDS * N_LAGS)
        for b in range(N_BANDS):
            base_eb = base_e + b * N_LAGS
            for l in range(N_LAGS):
                idx_map[l, e, b] = base_eb + l
    coefs_tensor = np.zeros((K, N_LAGS, N_ELECTRODES, N_BANDS), dtype=coefs_flat.dtype)
    for k in range(K):
        coefs_tensor[k] = coefs_flat[k, idx_map]
    return coefs_tensor

def main():
    pid = os.environ.get("PATIENT_ID", "unknown")
    print(f"[final] Patient {pid} — training 5 fold models with frozen HPs: {GDEC_HP}")

    # -------- Train per-fold models (saved for later evaluation/ensembling) --------
    fold_train_parts = []   # to build the union-of-train set for the optional final model
    fold_train_labels = []

    for fold_id in range(1, 6):  # folds 1..5
        X, y = load_xy_from_fold(fold_id)
        train_idx, test_idx, fold_dir = get_or_make_split_indices(fold_id, len(y), train_frac=0.8)

        Xtr, ytr = X[train_idx], y[train_idx]
        ytr_dense, classes_seen = remap_to_dense(ytr)

        model_fold = gdec.GaussianProcessMulticlassDecoder()
        model_fold.fit(Xtr, ytr_dense, **GDEC_HP)

        # save fold model + mapping
        joblib.dump(model_fold, FOLD_MODELS_DIR / f"fold_{fold_id}.joblib")
        np.save(FOLD_MODELS_DIR / f"classes_seen_fold_{fold_id}.npy", classes_seen)

        # accumulate for union model (attribution/deployment)
        fold_train_parts.append(Xtr)
        fold_train_labels.append(ytr)

    # -------- Train a single "union" model for attribution/deployment --------
    # IMPORTANT: Do NOT evaluate this union model on any fold test (would leak).
    X_final = np.concatenate(fold_train_parts, axis=0)
    y_final = np.concatenate(fold_train_labels, axis=0)
    y_final_dense, classes_seen_final = remap_to_dense(y_final)
    np.save(OUT_DIR / "classes_seen.npy", classes_seen_final)

    model_final = gdec.GaussianProcessMulticlassDecoder()
    model_final.fit(X_final, y_final_dense, **GDEC_HP)

    model_path = OUT_DIR / "gpmd_model_final.joblib"
    joblib.dump(model_final, model_path)

    # save weights for attribution
    if hasattr(model_final, "coefs_") and model_final.coefs_ is not None:
        coefs = model_final.coefs_                                # [K_seen_final, 1000]
        np.save(OUT_DIR / "coefs_raw.npy", coefs)
        intercept = getattr(model_final, "intercept_", None)
        if intercept is not None:
            np.save(OUT_DIR / "intercept.npy", intercept)
        np.save(OUT_DIR / "coefs_by_lag_elect_band.npy", reshape_coefs_for_attrib(coefs))
    else:
        (OUT_DIR / "WARNING.txt").write_text("Model has no attribute 'coefs_'; cannot save weights.\n")

    # -------- Metadata (no metrics here) --------
    meta = {
        "feature_dims": {"n_lags": N_LAGS, "n_electrodes": N_ELECTRODES, "n_bands": N_BANDS, "n_features": N_FEATURES},
        "feature_ordering_assumption": FEATURE_ORDERING,
        "frozen_hp": GDEC_HP,
        "classes_seen_final": classes_seen_final.tolist(),
        "paths": {
            "final_model": str(model_path),
            "fold_models_dir": str(FOLD_MODELS_DIR),
            "coefs_raw": str(OUT_DIR / "coefs_raw.npy"),
            "coefs_by_lag_elect_band": str(OUT_DIR / "coefs_by_lag_elect_band.npy"),
            "intercept": str(OUT_DIR / "intercept.npy"),
            "fold_splits_root": str(FOLDS_DIR),
        },
        "note": "Per-fold models trained with frozen HPs. Union model is for attribution/deployment only; no evaluation here.",
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[final] Done. Saved per-fold models in {FOLD_MODELS_DIR} and union model → {model_path}")

if __name__ == "__main__":
    main()