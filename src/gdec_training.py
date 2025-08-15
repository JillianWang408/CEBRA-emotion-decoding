import numpy as np
import scipy.io
import mat73
import json
import re
from pathlib import Path
import joblib
from src.config import (OUT_DIR, NEURAL_PATH, EMOTION_PATH, N_ELECTRODES, GDEC_HP, FOLD_MODELS_DIR, FOLDS_DIR)

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

# ---------- Shift helpers ----------
def shift_labels(y: np.ndarray, shift: int) -> np.ndarray:
    """Global label shift; invalid positions set to -1."""
    y_shift = np.full_like(y, fill_value=-1)
    if shift > 0:
        y_shift[shift:] = y[:-shift]
    elif shift < 0:
        y_shift[:shift] = y[-shift:]
    else:
        y_shift[:] = y
    return y_shift

def run_label_shift_sweep(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    fit_kwargs: dict,
    shift_range=range(-5, 6),
):
    """
    Try multiple shifts using ONLY (train_sub,val_sub) from TRAIN (no test leakage).
    For each shift:
      - mask invalid (-1)
      - z-score on TRAIN of this shift
      - remap TRAIN labels to dense
      - evaluate in-vocab acc on VAL
    Returns: (best_shift, best_val_acc)
    """
    best_shift, best_acc = 0, -1.0
    for s in shift_range:
        y_s = shift_labels(y, s)
        valid = (y_s != -1)
        tr_mask = valid[train_idx]
        va_mask = valid[val_idx]
        if not np.any(tr_mask) or not np.any(va_mask):
            continue
        Xtr_raw, ytr_raw = X[train_idx][tr_mask], y_s[train_idx][tr_mask]
        Xva_raw, yva_raw = X[val_idx][va_mask],   y_s[val_idx][va_mask]
        if np.unique(ytr_raw).size < 2 or np.unique(yva_raw).size < 2:
            continue

        mu = Xtr_raw.mean(0).astype(np.float32)
        std = Xtr_raw.std(0).astype(np.float32); std[std < 1e-8] = 1.0
        Xtr = (Xtr_raw - mu) / std
        Xva = (Xva_raw - mu) / std
        ytr_dense, classes_seen = remap_to_dense(ytr_raw)

        va_seen = np.isin(yva_raw, classes_seen)
        if not np.any(va_seen):
            continue
        Xva2, yva2 = Xva[va_seen], yva_raw[va_seen]

        m = gdec.GaussianProcessMulticlassDecoder()
        m.fit(Xtr, ytr_dense, **fit_kwargs)
        y_pred_dense = m.predict(Xva2)
        y_pred = classes_seen[y_pred_dense]
        acc = float((y_pred == yva2).mean())

        print(f"[shift_sweep] shift={s:+d} | val_acc={acc:.3f}")
        if acc > best_acc:
            best_acc, best_shift = acc, s
    return best_shift, best_acc

# ---------- Caching utilities ----------
def get_or_compute_fold_shift(X, y, train_idx, shift_range, fold_dir, fit_kwargs):
    """
    Return (best_shift, source) where source in {'cache','computed'}.
    Uses a small validation split from TRAIN to compute if missing.
    """
    p_best = fold_dir / "best_shift.npy"
    if p_best.exists():
        return int(np.load(p_best)), "cache"

    # make a small val from TRAIN (no leakage)
    cut_val = int(len(train_idx) * 0.9)
    tr_sub = train_idx[:cut_val]
    va_sub = train_idx[cut_val:]
    best_shift, best_val_acc = run_label_shift_sweep(
        X, y, tr_sub, va_sub, fit_kwargs=fit_kwargs, shift_range=shift_range
    )
    np.save(p_best, np.array(best_shift, dtype=np.int32))
    print(f"[cache] saved best_shift={best_shift:+d} at {p_best}")
    return best_shift, "computed"

def get_or_compute_fold_scaler(Xtr_raw, fold_dir):
    """
    Return (mu,std, source) where source in {'cache','computed'}.
    """
    p_mu = fold_dir / "scaler_mean.npy"
    p_sd = fold_dir / "scaler_std.npy"
    if p_mu.exists() and p_sd.exists():
        mu = np.load(p_mu); std = np.load(p_sd)
        return mu.astype(np.float32), std.astype(np.float32), "cache"
    mu = Xtr_raw.mean(0).astype(np.float32)
    std = Xtr_raw.std(0).astype(np.float32); std[std < 1e-8] = 1.0
    np.save(p_mu, mu); np.save(p_sd, std)
    print(f"[cache] saved scaler at {fold_dir}")
    return mu, std, "computed"

def get_or_compute_patient_shift(fold_ids, folds_root, out_dir):
    """
    Majority vote over existing per-fold shifts; compute vote and save patient_shift.npy.
    If already saved, re-use it.
    """
    p_patient = out_dir / "patient_shift.npy"
    if p_patient.exists():
        return int(np.load(p_patient)), "cache"

    shifts = []
    for k in fold_ids:
        p = folds_root / f"fold_{k}" / "best_shift.npy"
        if p.exists():
            shifts.append(int(np.load(p)))
    if not shifts:
        raise RuntimeError("No per-fold best_shift.npy found to derive patient shift.")
    vals, counts = np.unique(shifts, return_counts=True)
    patient_shift = int(vals[np.argmax(counts)])
    ties = vals[counts == counts.max()]
    if ties.size > 1:
        patient_shift = int(ties[np.argmin(np.abs(ties))])
    np.save(p_patient, np.array(patient_shift, dtype=np.int32))
    print(f"[cache] saved patient_shift={patient_shift:+d} at {p_patient}")
    return patient_shift, "computed"

def get_or_compute_patient_scaler(X_raw, out_dir):
    """
    Return (mu,std, source) for the patient-level scaler; cache at OUT_DIR.
    """
    p_mu = out_dir / "patient_scaler_mean.npy"
    p_sd = out_dir / "patient_scaler_std.npy"
    if p_mu.exists() and p_sd.exists():
        mu = np.load(p_mu); std = np.load(p_sd)
        return mu.astype(np.float32), std.astype(np.float32), "cache"
    mu = X_raw.mean(0).astype(np.float32)
    std = X_raw.std(0).astype(np.float32); std[std < 1e-8] = 1.0
    np.save(p_mu, mu); np.save(p_sd, std)
    print(f"[cache] saved patient scaler at {out_dir}")
    return mu, std, "computed"

# ---------- Main ----------
def main():
    pid = os.environ.get("PATIENT_ID", "unknown")
    print(f"[train] Patient {pid} — per-fold models + single patient model (attribution only).")
    print(f"[train] Frozen HPs: {GDEC_HP}")

    # ---- Per-fold training (for evaluation) ----
    fold_shifts = []
    for fold_id in range(1, 6):
        X, y = load_xy_from_fold(fold_id)
        train_idx, test_idx, fold_dir = get_or_make_split_indices(fold_id, len(y), train_frac=0.8)

        # 1) SHIFT: reuse if exists, else compute w/ small val split inside TRAIN
        best_shift, src_shift = get_or_compute_fold_shift(
            X, y, train_idx, shift_range=range(-5, 6), fold_dir=fold_dir, fit_kwargs=GDEC_HP
        )
        print(f"[fold {fold_id}] best_shift={best_shift:+d} ({src_shift})")
        fold_shifts.append(int(best_shift))

        # Build shifted TRAIN subset
        y_best = shift_labels(y, best_shift)
        valid = (y_best != -1)
        tr_mask = valid[train_idx]
        Xtr_raw = X[train_idx][tr_mask]
        ytr_raw = y_best[train_idx][tr_mask]

        # 2) SCALER: reuse if exists, else compute on this shifted TRAIN
        mu, std, src_scaler = get_or_compute_fold_scaler(Xtr_raw, fold_dir)
        Xtr = (Xtr_raw - mu) / std

        # Dense labels + classes
        ytr_dense, classes_seen = remap_to_dense(ytr_raw)
        FOLD_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(FOLD_MODELS_DIR / f"classes_seen_fold_{fold_id}.npy", classes_seen)

        # 3) Train (always trains model; you can add a "skip if exists" gate if you want)
        model_fold = gdec.GaussianProcessMulticlassDecoder()
        model_fold.fit(Xtr, ytr_dense, **GDEC_HP)
        joblib.dump(model_fold, FOLD_MODELS_DIR / f"fold_{fold_id}.joblib")
        print(f"[fold {fold_id}] model saved. shift:{src_shift}, scaler:{src_scaler}")

    # ---- Patient-level (single consistent model for attribution/deployment) ----
    patient_shift, src_p_shift = get_or_compute_patient_shift(range(1, 6), FOLDS_DIR, OUT_DIR)
    print(f"[patient] chosen patient_shift={patient_shift:+d} ({src_p_shift}) from folds")

    # Use canonical fold file (fold 1) to avoid duplicated rows
    X_all, y_all = load_xy_from_fold(1)
    y_shifted = shift_labels(y_all.astype(np.int64), patient_shift)
    valid = (y_shifted != -1)
    Xp_raw, yp_raw = X_all[valid], y_shifted[valid]

    mu_p, std_p, src_p_scaler = get_or_compute_patient_scaler(Xp_raw, OUT_DIR)
    Xp = (Xp_raw - mu_p) / std_p

    y_dense_p, classes_seen_patient = remap_to_dense(yp_raw)
    np.save(OUT_DIR / "classes_seen_patient.npy", classes_seen_patient)

    model_patient = gdec.GaussianProcessMulticlassDecoder()
    model_patient.fit(Xp, y_dense_p, **GDEC_HP)

    patient_model_path = OUT_DIR / "gpmd_model_patient.joblib"
    joblib.dump(model_patient, patient_model_path)

    # Save weights for attribution
    if hasattr(model_patient, "coefs_") and model_patient.coefs_ is not None:
        coefs = model_patient.coefs_
        np.save(OUT_DIR / "coefs_raw.npy", coefs)
        intercept = getattr(model_patient, "intercept_", None)
        if intercept is not None:
            np.save(OUT_DIR / "intercept.npy", intercept)
        np.save(OUT_DIR / "coefs_by_lag_elect_band.npy", reshape_coefs_for_attrib(coefs))
    else:
        (OUT_DIR / "WARNING.txt").write_text("Model has no attribute 'coefs_'; cannot save weights.\n")

    # ---- Metadata ----
    meta = {
        "feature_dims": {"n_lags": N_LAGS, "n_electrodes": N_ELECTRODES, "n_bands": N_BANDS, "n_features": N_FEATURES},
        "feature_ordering_assumption": FEATURE_ORDERING,
        "frozen_hp": GDEC_HP,
        "paths": {
            "fold_models_dir": str(FOLD_MODELS_DIR),
            "folds_root": str(FOLDS_DIR),
            "patient_model": str(patient_model_path),
            "patient_shift": str(OUT_DIR / "patient_shift.npy"),
            "patient_scaler_mean": str(OUT_DIR / "patient_scaler_mean.npy"),
            "patient_scaler_std": str(OUT_DIR / "patient_scaler_std.npy"),
            "classes_seen_patient": str(OUT_DIR / "classes_seen_patient.npy"),
            "coefs_raw": str(OUT_DIR / "coefs_raw.npy"),
            "coefs_by_lag_elect_band": str(OUT_DIR / "coefs_by_lag_elect_band.npy"),
            "intercept": str(OUT_DIR / "intercept.npy"),
        },
        "notes": [
            "Per-fold: caches best_shift.npy and scaler_* in fold directories; reuses if present.",
            "Patient: caches patient_shift.npy and patient_scaler_* in OUT_DIR; reuses if present.",
            "Fold models are for evaluation; patient model is for attribution/deployment only.",
        ],
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[patient] Final patient model saved → {patient_model_path}")

if __name__ == "__main__":
    main()