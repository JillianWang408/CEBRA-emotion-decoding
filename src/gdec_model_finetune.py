import numpy as np
import scipy.io
import mat73
import json
import re
from pathlib import Path
import joblib
from src.config import (
    MODEL_DIR, NEURAL_PATH, EMOTION_PATH, N_ELECTRODES, DATA_DIR,
    FULL_NEURAL_PATH, FULL_EMOTION_PATH, NEURAL_TENSOR_PATH, EMOTION_TENSOR_PATH
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
OUT_DIR = MODEL_DIR
FOLDS_DIR = DATA_DIR
FOLD_MODELS_DIR = OUT_DIR / "fold_models"  
OUT_DIR.mkdir(parents=True, exist_ok=True)
FOLD_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Helpers ----------
_fold_suffix_re = re.compile(r"_(\d+)\.mat$")

def _swap_fold(path: Path, fold_id: int) -> Path:
    """# NEW: replace trailing '_<num>.mat' with '_{fold_id}.mat'."""
    new_name = _fold_suffix_re.sub(f"_{fold_id}.mat", path.name)
    return path.with_name(new_name)

def load_xy_from_fold(fold_id: int):
    """# NEW: load X,y for a given fold from its own MAT files."""
    X_path = _swap_fold(NEURAL_PATH, fold_id)
    y_path = _swap_fold(EMOTION_PATH, fold_id)
    X = mat73.loadmat(X_path)['stim'].T.astype(np.float32)           # [T, 1000]
    y = scipy.io.loadmat(y_path)['resp'].flatten().astype(np.int64)  # [T]
    assert X.shape[1] == N_FEATURES, f"Expected {N_FEATURES} features, got {X.shape[1]} (fold {fold_id})"
    return X, y

def split_train_val(X, y, train_frac=0.8):
    """# NEW: within each fold, use first 80% as train (data already shuffled)."""
    n = len(y)
    cut = int(n * train_frac)
    idx_train = np.arange(0, cut, dtype=np.int64)
    idx_val   = np.arange(cut, n, dtype=np.int64)
    return X[idx_train], y[idx_train], X[idx_val], y[idx_val]

def remap_to_dense(y_train):
    """Map labels in y_train to 0..K_seen-1; return dense labels and classes_seen."""
    classes_seen = np.unique(y_train)
    map_to_dense = {c: i for i, c in enumerate(classes_seen)}
    y_dense = np.vectorize(map_to_dense.get)(y_train).astype(np.int64)
    return y_dense, classes_seen

def iv_accuracy(model, Xv, yv, classes_seen):
    """In-vocab accuracy on validation set (mask to classes_seen)."""
    mask = np.isin(yv, classes_seen)
    if not np.any(mask):
        return 0.0
    Xv2, yv2 = Xv[mask], yv[mask]
    y_pred_dense = model.predict(Xv2)
    y_pred = classes_seen[y_pred_dense]
    return float((y_pred == yv2).mean())

def reshape_coefs_for_attrib(coefs_flat: np.ndarray) -> np.ndarray:
    """[K_seen, 1000] -> [K_seen, 5, 40, 5] using lag-first then band within electrode."""
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
    print(f"[cv] Patient {pid} — using 5 folds via file suffixes _1.mat.._5.mat")

    # ---- Hyperparameter grid (small; expand if you like) ----
    grid = [
        dict(lr=0.02, max_steps=5000, n_samples=4),
        dict(lr=0.05, max_steps=4000, n_samples=4),
        dict(lr=0.05, max_steps=5000, n_samples=8),
        dict(lr=0.10, max_steps=3000, n_samples=4),
    ]
    common = dict(log_every=50, cuda=False, cuda_device=0)  # macOS CPU

    # ---- Cross-validate across the 5 folds (each from its own files)  ----
    cv_results = []
    for hp in grid:
        fold_scores = []
        for fold_id in range(1, 6):  # folds 1..5
            X, y = load_xy_from_fold(fold_id) 
            Xtr, ytr, Xval, yval = split_train_val(X, y, train_frac=0.8)  

            ytr_dense, classes_seen = remap_to_dense(ytr)
            model = gdec.GaussianProcessMulticlassDecoder()
            fit_kwargs = {**hp, **common}
            model.fit(Xtr, ytr_dense, **fit_kwargs)

            acc = iv_accuracy(model, Xval, yval, classes_seen)
            fold_scores.append(acc)

        cv_results.append({
            "hp": hp,
            "mean_val_acc": float(np.mean(fold_scores)),
            "per_fold": [float(s) for s in fold_scores],
        })

    cv_results.sort(key=lambda r: r["mean_val_acc"], reverse=True)
    best = cv_results[0]
    best_hp = best["hp"]
    print(f"[cv] Best mean val acc={best['mean_val_acc']:.3f} with {best_hp}")
    (OUT_DIR / "cv_results.json").write_text(json.dumps(cv_results, indent=2))

    # ---- Final training on the union of all folds' TRAIN parts with best HP ----
    X_all_trains, y_all_trains = [], []
    for fold_id in range(1, 6):
        X, y = load_xy_from_fold(fold_id)
        Xtr, ytr, _, _ = split_train_val(X, y, train_frac=0.8)
        X_all_trains.append(Xtr)
        y_all_trains.append(ytr)
    X_final = np.concatenate(X_all_trains, axis=0)
    y_final = np.concatenate(y_all_trains, axis=0)

    y_final_dense, classes_seen_final = remap_to_dense(y_final)
    np.save(OUT_DIR / "classes_seen.npy", classes_seen_final) 

    model_final = gdec.GaussianProcessMulticlassDecoder()
    fit_kwargs_final = {**best_hp, **common}
    print(f"[final] Training with best hp: {fit_kwargs_final} on {X_final.shape[0]} samples")
    model_final.fit(X_final, y_final_dense, **fit_kwargs_final)

    # ---- Save final model + weights (same artifacts as before) ----
    model_path = OUT_DIR / "gpmd_model_best.joblib"
    joblib.dump(model_final, model_path)

    if hasattr(model_final, "coefs_") and model_final.coefs_ is not None:
        coefs = model_final.coefs_                              # [K_seen_final, 1000]
        np.save(OUT_DIR / "coefs_raw.npy", coefs)
        intercept = getattr(model_final, "intercept_", None)
        if intercept is not None:
            np.save(OUT_DIR / "intercept.npy", intercept)
        np.save(OUT_DIR / "coefs_by_lag_elect_band.npy", reshape_coefs_for_attrib(coefs))
    else:
        (OUT_DIR / "WARNING.txt").write_text("Model has no attribute 'coefs_'; cannot save weights.\n")

    # ---- Save each fold model for ensembling later ----
    for fold_id in range(1, 6):
        X, y = load_xy_from_fold(fold_id)
        Xtr, ytr, _, _ = split_train_val(X, y, train_frac=0.8)
        ytr_dense, classes_seen = remap_to_dense(ytr)

        model_fold = gdec.GaussianProcessMulticlassDecoder()
        model_fold.fit(Xtr, ytr_dense, **fit_kwargs_final)
        joblib.dump(model_fold, FOLD_MODELS_DIR / f"fold_{fold_id}.joblib")
        np.save(FOLD_MODELS_DIR / f"classes_seen_fold_{fold_id}.npy", classes_seen)

    # ---- Metadata ----
    meta = {
        "feature_dims": {"n_lags": N_LAGS, "n_electrodes": N_ELECTRODES, "n_bands": N_BANDS, "n_features": N_FEATURES},
        "feature_ordering_assumption": FEATURE_ORDERING,
        "cv_results_path": str(OUT_DIR / "cv_results.json"),
        "best_hp": fit_kwargs_final,
        "classes_seen_final": classes_seen_final.tolist(),
        "paths": {
            "final_model": str(model_path),
            "fold_models_dir": str(FOLD_MODELS_DIR),
            "coefs_raw": str(OUT_DIR / "coefs_raw.npy"),
            "coefs_by_lag_elect_band": str(OUT_DIR / "coefs_by_lag_elect_band.npy"),
            "intercept": str(OUT_DIR / "intercept.npy"),
        },
        "note": "5-fold CV across files *_1.mat..*_5.mat; final model on union of train parts from all folds.",
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"[cv] Done. Best mean val acc={best['mean_val_acc']:.3f}. Final model → {model_path}")

if __name__ == "__main__":
    main()