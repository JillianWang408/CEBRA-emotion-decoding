# RUN: python -u -m src.gdec_model_finetune --feature-mode {lags|cov}
import os, sys, types, re, json, inspect, argparse
from pathlib import Path
import numpy as np
import joblib
import mat73, scipy.io

from src.config import (
    MODEL_DIR, DATA_DIR, NEURAL_PATH, EMOTION_PATH, N_ELECTRODES,
    OUT_DIR, FOLDS_DIR, FOLD_MODELS_DIR
)

# ---------- JAX (CPU) shim ----------
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
try:
    from jax.config import config  # noqa: F401
except Exception:
    m = types.ModuleType("jax.config"); m.config = jax.config; sys.modules["jax.config"] = m

# ---------- Torch ReduceLROnPlateau 'verbose' guard ----------
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

import gdec  # after shim

# ---------- Defaults for lag/band layout ----------
N_LAGS_DEFAULT  = 5
N_BANDS_DEFAULT = 5

# Output dirs
OUT_DIR.mkdir(parents=True, exist_ok=True)
FOLD_MODELS_DIR.mkdir(parents=True, exist_ok=True)
FOLDS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Fold file swapping ----------
_fold_suffix_re = re.compile(r"_(\d+)\.mat$")
def _swap_fold(path: Path, fold_id: int) -> Path:
    """Replace trailing '_<num>.mat' with '_{fold_id}.mat'."""
    return path.with_name(_fold_suffix_re.sub(f"_{fold_id}.mat", path.name))

# ---------- Loaders (feature-mode specific) ----------
def load_xy_from_fold_lags(fold_id: int, n_lags: int, n_bands: int):
    """Expect stim [T, n_lags * N_ELECTRODES * n_bands] in MAT."""
    X_path = _swap_fold(Path(NEURAL_PATH), fold_id)
    y_path = _swap_fold(Path(EMOTION_PATH), fold_id)
    X = mat73.loadmat(X_path)['stim'].T.astype(np.float32)           # [T, F]
    y = scipy.io.loadmat(y_path)['resp'].flatten().astype(np.int64)  # [T]
    expected = n_lags * N_ELECTRODES * n_bands
    assert X.shape[1] == expected, f"Expected {expected} features, got {X.shape[1]} (fold {fold_id})"
    return X, y

def load_xy_from_fold_cov_ut(fold_id: int):
    """
    Covariance mode (upper-triangle). Accepts:
      - [T,1600] flattened full 40x40 -> we vech to 820
      - [T,820]  already vech -> used as-is
    """
    X_path = _swap_fold(Path(NEURAL_PATH), fold_id)
    y_path = _swap_fold(Path(EMOTION_PATH), fold_id)
    X_raw = mat73.loadmat(X_path)['stim'].T.astype(np.float32)       # [T, F]
    y = scipy.io.loadmat(y_path)['resp'].flatten().astype(np.int64)  # [T]

    if X_raw.shape[1] == N_ELECTRODES * N_ELECTRODES:  # 1600
        T = X_raw.shape[0]
        M = X_raw.reshape(T, N_ELECTRODES, N_ELECTRODES)  # assume row-major flatten from .mat
        # Optional: enforce symmetry (data should already be symmetric)
        M = 0.5 * (M + np.swapaxes(M, 1, 2))
        iu = np.triu_indices(N_ELECTRODES, k=0)
        X = M[:, iu[0], iu[1]]  # [T, 820]
    elif X_raw.shape[1] == (N_ELECTRODES * (N_ELECTRODES + 1)) // 2:  # 820
        X = X_raw
    else:
        raise AssertionError(f"Covariance mode expects 1600 or 820 features, got {X_raw.shape[1]}")
    return X.astype(np.float32), y

# ---------- Split / zscore / remap ----------
def split_train_val_idx(n: int, train_frac=0.8):
    cut = int(n * train_frac)
    idx_train = np.arange(0, cut, dtype=np.int64)
    idx_val   = np.arange(cut, n, dtype=np.int64)
    return idx_train, idx_val

def zscore_from_train(Xtr_raw: np.ndarray):
    mu = Xtr_raw.mean(axis=0).astype(np.float32)
    sd = Xtr_raw.std(axis=0).astype(np.float32)
    sd[sd < 1e-8] = 1.0
    return mu, sd

def apply_zscore(X: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return (X - mu) / sd

def remap_to_dense(y_train):
    classes_seen = np.unique(y_train)
    map_to_dense = {c: i for i, c in enumerate(classes_seen)}
    y_dense = np.vectorize(map_to_dense.get)(y_train).astype(np.int64)
    return y_dense, classes_seen

def iv_accuracy(model, Xv, yv, classes_seen):
    mask = np.isin(yv, classes_seen)
    if not np.any(mask):
        return 0.0
    Xv2, yv2 = Xv[mask], yv[mask]
    y_pred_dense = model.predict(Xv2)
    y_pred = classes_seen[y_pred_dense]
    return float((y_pred == yv2).mean())

# ---------- Reshapes for attribution ----------
def reshape_coefs_lags(coefs_flat: np.ndarray, n_lags: int, n_bands: int) -> np.ndarray:
    """
    [K, F] -> [K, n_lags, N_ELECTRODES, n_bands]
    Order: lag-major: idx = l*(E*B) + e*B + b
    """
    K, F = coefs_flat.shape
    expected = n_lags * N_ELECTRODES * n_bands
    assert F == expected, f"Expected {expected}, got {F}"
    return coefs_flat.reshape(K, n_lags, N_ELECTRODES, n_bands)

def reshape_coefs_cov_ut(coefs_ut: np.ndarray) -> np.ndarray:
    """
    [K, 820] -> [K, 40, 40] symmetric matrices.
    Upper-triangle order from np.triu_indices.
    """
    K, F = coefs_ut.shape
    n = N_ELECTRODES
    expected = (n * (n + 1)) // 2
    assert F == expected, f"Expected {expected}, got {F}"
    iu = np.triu_indices(n, k=0)
    out = np.zeros((K, n, n), dtype=coefs_ut.dtype)
    for k in range(K):
        M = np.zeros((n, n), dtype=coefs_ut.dtype)
        M[iu] = coefs_ut[k]
        M = M + np.triu(M, 1).T  # reflect
        out[k] = M
    return out

# ---------- Label shift ----------
def shift_labels(y: np.ndarray, shift: int) -> np.ndarray:
    y_shift = np.full_like(y, fill_value=-1)
    if shift > 0:
        y_shift[shift:] = y[:-shift]
    elif shift < 0:
        y_shift[:shift] = y[-shift:]
    else:
        y_shift[:] = y
    return y_shift

# ---------- Small GDEC for shift sweep ----------
SWEEP_HP = dict(lr=0.05, max_steps=1500, n_samples=4, log_every=50, cuda=False, cuda_device=0)

def run_label_shift_sweep_gdec(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray,
                               shift_range=range(-5, 6), make_decoder=None):
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
        sd = Xtr_raw.std(0).astype(np.float32); sd[sd < 1e-8] = 1.0
        Xtr = (Xtr_raw - mu) / sd
        Xva = (Xva_raw - mu) / sd
        ytr_dense, classes_seen = remap_to_dense(ytr_raw)

        va_seen = np.isin(yva_raw, classes_seen)
        if not np.any(va_seen):
            continue
        Xva2, yva2 = Xva[va_seen], yva_raw[va_seen]

        m = make_decoder() if make_decoder is not None else gdec.GaussianProcessMulticlassDecoder()
        m.fit(Xtr, ytr_dense, **SWEEP_HP)
        y_pred_dense = m.predict(Xva2)
        y_pred = classes_seen[y_pred_dense]
        acc = float((y_pred == yva2).mean())
        print(f"[shift_sweep] shift={s:+d} | val_acc={acc:.3f}")
        if acc > best_acc:
            best_acc, best_shift = acc, s
    return best_shift, best_acc

# ---------- Caches ----------
def get_or_compute_fold_split(n, fold_dir: Path, train_frac=0.8, force=False):
    p_tr, p_va = fold_dir / "train_idx.npy", fold_dir / "test_idx.npy"
    if p_tr.exists() and p_va.exists() and not force:
        return np.load(p_tr), np.load(p_va), "cache"
    tr, va = split_train_val_idx(n, train_frac=train_frac)
    np.save(p_tr, tr); np.save(p_va, va)
    return tr, va, "computed"

def get_or_compute_fold_shift(X, y, train_idx, fold_dir: Path, shift_range, make_decoder, force=False):
    p_best = fold_dir / "best_shift.npy"
    if p_best.exists() and not force:
        return int(np.load(p_best)), "cache"
    cut_val = int(len(train_idx) * 0.9)
    tr_sub = train_idx[:cut_val]; va_sub = train_idx[cut_val:]
    best_shift, _ = run_label_shift_sweep_gdec(X, y, tr_sub, va_sub,
                                               shift_range=shift_range, make_decoder=make_decoder)
    np.save(p_best, np.array(best_shift, dtype=np.int32))
    print(f"[cache] saved best_shift={best_shift:+d} at {p_best}")
    return best_shift, "computed"

def get_or_compute_fold_scaler(Xtr_raw, fold_dir: Path, force=False):
    p_mu = fold_dir / "scaler_mean.npy"
    p_sd = fold_dir / "scaler_std.npy"
    if p_mu.exists() and p_sd.exists() and not force:
        mu = np.load(p_mu); sd = np.load(p_sd)
        return mu.astype(np.float32), sd.astype(np.float32), "cache"
    mu, sd = zscore_from_train(Xtr_raw)
    np.save(p_mu, mu); np.save(p_sd, sd)
    print(f"[cache] saved scaler at {fold_dir}")
    return mu, sd, "computed"

def get_or_compute_best_hp_across_folds(fold_ids, classes_seen_by_fold, train_blobs, val_blobs,
                                        grid, common, make_decoder, force=False):
    p_cv  = OUT_DIR / "cv_results_gdec.json"
    p_best = OUT_DIR / "best_hp_gdec.json"
    if p_best.exists() and p_cv.exists() and not force:
        best_hp = json.loads(p_best.read_text())
        print(f"[cv-gdec] using cached best hp: {best_hp}")
        return best_hp

    cv_results = []
    for hp in grid:
        fold_scores = []
        for k in fold_ids:
            (Xtr, ytr_dense) = train_blobs[k]
            (Xva, yva_raw)   = val_blobs[k]
            classes_seen     = classes_seen_by_fold[k]
            model = make_decoder()
            model.fit(Xtr, ytr_dense, **{**hp, **common})
            acc = iv_accuracy(model, Xva, yva_raw, classes_seen)
            fold_scores.append(acc)
        cv_results.append({
            "hp": hp,
            "mean_val_acc": float(np.mean(fold_scores)) if fold_scores else 0.0,
            "per_fold": [float(s) for s in fold_scores],
        })

    cv_results.sort(key=lambda r: r["mean_val_acc"], reverse=True)
    best = cv_results[0]
    best_hp = best["hp"]
    print(f"[cv-gdec] Best mean val acc={best['mean_val_acc']:.3f} with {best_hp}")
    p_cv.write_text(json.dumps(cv_results, indent=2))
    p_best.write_text(json.dumps(best_hp))
    return best_hp

def get_or_compute_patient_shift(fold_ids, folds_root: Path, out_dir: Path, force=False):
    p_patient = out_dir / "patient_shift.npy"
    if p_patient.exists() and not force:
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

def get_or_compute_patient_scaler(X_raw, out_dir: Path, force=False):
    p_mu = out_dir / "patient_scaler_mean.npy"
    p_sd = out_dir / "patient_scaler_std.npy"
    if p_mu.exists() and p_sd.exists() and not force:
        mu = np.load(p_mu); sd = np.load(p_sd)
        return mu.astype(np.float32), sd.astype(np.float32), "cache"
    mu, sd = zscore_from_train(X_raw)
    np.save(p_mu, mu); np.save(p_sd, sd)
    print(f"[cache] saved patient scaler at {out_dir}")
    return mu, sd, "computed"

# ---------- Kernels ----------
def rbf_on_indices(n: int, ell: float = 1.0, var: float = 1.0, eps: float = 1e-6):
    idx = np.arange(n).astype(np.float32)
    D = np.abs(idx[:, None] - idx[None, :])
    K = var * np.exp(-0.5 * (D / max(ell, 1e-6))**2)
    K.flat[::n+1] += eps  # jitter on diagonal
    return K.astype(np.float32)

def block_region_kernel(n_elec: int, group_size: int = 4, rho: float = 0.5, eps: float = 1e-6):
    """
    40 electrodes, groups of 4 share correlation rho; diagonal=1.
    """
    K = np.eye(n_elec, dtype=np.float32) * (1.0 + eps)
    n_groups = n_elec // group_size
    for g in range(n_groups):
        start = g * group_size
        sl = slice(start, start + group_size)
        K[np.ix_(range(start, start+group_size), range(start, start+group_size))] += \
            rho * (np.ones((group_size, group_size), dtype=np.float32) - np.eye(group_size, dtype=np.float32))
    return K

def kron3(KB, KE, KL):
    # C-order flatten of [L,E,B] → K = KB ⊗ KE ⊗ KL (band last)
    return np.kron(np.kron(KB, KE), KL).astype(np.float32)

def cov_ut_kernel_from_region(KE: np.ndarray, eps: float = 1e-6):
    """
    Build kernel on upper-triangle pairs (i<=j) from KE (40x40).
    K_ut[(i,j),(k,l)] = 0.5 * (KE[i,k]*KE[j,l] + KE[i,l]*KE[j,k]) for off-diags, and KE[i,k]*KE[j,l] on diags.
    """
    n = KE.shape[0]
    iu = np.triu_indices(n, k=0)
    pairs = list(zip(iu[0], iu[1]))
    m = len(pairs)
    K_ut = np.zeros((m, m), dtype=np.float32)
    for a, (i, j) in enumerate(pairs):
        for b, (k, l) in enumerate(pairs):
            base = KE[i, k] * KE[j, l]
            if i != j or k != l:
                cross = KE[i, l] * KE[j, k]
                val = 0.5 * (base + cross)
            else:
                val = base
            K_ut[a, b] = val
    K_ut.flat[::m+1] += eps
    return K_ut

def build_feature_kernel(feature_mode: str, n_lags: int, n_bands: int,
                         ell_lag: float, var_lag: float, rho_region: float, region_size: int,
                         cache_dir: Path):
    """
    Returns (K_feat, path) for the chosen feature mode.

    - lags:  temporal-only kernel  => K = I_band ⊗ I_elec ⊗ K_lag
    - cov:   NO feature kernel     => return (None, None)
    """
    if feature_mode == "lags":
        # temporal-only kernel
        K_lag  = rbf_on_indices(n_lags, ell=ell_lag, var=var_lag)
        K_elec = np.eye(N_ELECTRODES, dtype=np.float32)   # <-- no spatial coupling
        K_band = np.eye(n_bands, dtype=np.float32)        # <-- bands independent

        K_feat = kron3(K_band, K_elec, K_lag)             # (n_lags*E*n_bands)^2
        path = cache_dir / "kernel_feat_lags_temporal.npy"
        np.save(path, K_feat.astype(np.float32))
        return K_feat.astype(np.float32), path

    elif feature_mode == "cov":
        # No kernel for covariance features (use model's default / identity).
        return None, None

    else:
        raise ValueError("feature_mode must be 'lags' or 'cov'")

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-mode", choices=["lags", "cov"], default="lags",
                        help="Choose feature structure: 'lags' or 'cov' (upper-triangle)")
    parser.add_argument("--force", action="store_true", help="Force re-run of all stages")
    parser.add_argument("--force-shift", action="store_true", help="Force recompute label shift")
    parser.add_argument("--force-scaler", action="store_true", help="Force recompute scalers")
    parser.add_argument("--force-grid", action="store_true", help="Force recompute hyperparameter CV")
    parser.add_argument("--force-fold-models", action="store_true", help="Force retrain per-fold models")
    parser.add_argument("--force-final", action="store_true", help="Force retrain patient union model")
    parser.add_argument("--shift-min", type=int, default=-5)
    parser.add_argument("--shift-max", type=int, default=5)
    # kernel hyperparams
    parser.add_argument("--ell-lag", type=float, default=1.0, help="RBF length-scale (lag units) for lags mode")
    parser.add_argument("--var-lag", type=float, default=1.0, help="RBF variance for lags mode")
    parser.add_argument("--rho-region", type=float, default=0.5, help="Within-region electrode correlation (groups of 4)")
    parser.add_argument("--region-size", type=int, default=4, help="Electrodes per region")
    args = parser.parse_args()
    mode_tag = args.feature_mode  # "lags" or "cov"

    # Use mode-specific subdirectories to avoid cache collisions between feature spaces
    OUT_DIR_MODE        = OUT_DIR / mode_tag
    FOLDS_DIR_MODE      = FOLDS_DIR / mode_tag
    FOLD_MODELS_DIR_MODE= FOLD_MODELS_DIR / mode_tag

    OUT_DIR_MODE.mkdir(parents=True, exist_ok=True)
    FOLDS_DIR_MODE.mkdir(parents=True, exist_ok=True)
    FOLD_MODELS_DIR_MODE.mkdir(parents=True, exist_ok=True)

    # also update any paths that were previously using OUT_DIR / FOLDS_DIR / FOLD_MODELS_DIR
    K_path = OUT_DIR_MODE / "feature_kernel.npy"   # was OUT_DIR / "feature_kernel_lagmajor.npy"

    if args.force:
        args.force_shift = args.force_scaler = args.force_grid = True
        args.force_fold_models = args.force_final = True

    pid = os.environ.get("PATIENT_ID", "unknown")
    print(f"[cv-gdec] Patient {pid} — feature-mode={args.feature_mode} | 5 folds via file suffixes _1.mat.._5.mat")

    # Build the feature kernel
    K_feat, K_path = build_feature_kernel(
        feature_mode=args.feature_mode,
        n_lags=N_LAGS_DEFAULT, n_bands=N_BANDS_DEFAULT,
        ell_lag=args.ell_lag, var_lag=args.var_lag,
        rho_region=args.rho_region, region_size=args.region_size,
        cache_dir=OUT_DIR_MODE
    )

    def make_gdec_with_kernel():
        # If no kernel (cov mode), just construct the decoder without feature_kernel.
        if K_feat is None:
            return gdec.GaussianProcessMulticlassDecoder()

        # Else, pass the feature kernel (lags mode).
        try:
            return gdec.GaussianProcessMulticlassDecoder(feature_kernel=K_feat)
        except TypeError:
            try:
                return gdec.GaussianProcessMulticlassDecoder(kernel=K_feat)
            except TypeError:
                m = gdec.GaussianProcessMulticlassDecoder()
                if hasattr(m, "set_feature_kernel"):
                    m.set_feature_kernel(K_feat)
                else:
                    m.feature_kernel_ = K_feat
                return m

    # Hyperparameter grid (unchanged)
    grid = [
        dict(lr=0.02, max_steps=5000, n_samples=4),
        dict(lr=0.05, max_steps=4000, n_samples=4),
        dict(lr=0.05, max_steps=5000, n_samples=8),
        dict(lr=0.10, max_steps=3000, n_samples=4),
    ]
    common = dict(log_every=50, cuda=False, cuda_device=0)

    # Feature mode registry
    FEAT = {
        "lags": {
            "loader": lambda k: load_xy_from_fold_lags(k, N_LAGS_DEFAULT, N_BANDS_DEFAULT),
            "expected_dim": N_LAGS_DEFAULT * N_ELECTRODES * N_BANDS_DEFAULT,
        },
        "cov": {
            "loader": load_xy_from_fold_cov_ut,
            "expected_dim": (N_ELECTRODES * (N_ELECTRODES + 1)) // 2,  # 820
        },
    }[args.feature_mode]

    # ---- Stage A: split, shift, scaler ----
    fold_ids = range(1, 6)
    train_blobs = {}
    val_blobs   = {}
    classes_by_fold = {}
    for k in fold_ids:
        X, y = FEAT["loader"](k)
        assert X.shape[1] == FEAT["expected_dim"], f"Fold {k}: expected {FEAT['expected_dim']}, got {X.shape[1]}"
        fold_dir = FOLDS_DIR_MODE / f"fold_{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_idx, val_idx, _ = get_or_compute_fold_split(len(y), fold_dir, train_frac=0.8, force=False)

        best_shift, src_shift = get_or_compute_fold_shift(
            X, y, train_idx, fold_dir, shift_range=range(args.shift_min, args.shift_max + 1),
            make_decoder=make_gdec_with_kernel, force=args.force_shift
        )
        print(f"[fold {k}] best_shift={best_shift:+d} ({src_shift})")

        y_best = shift_labels(y, best_shift)
        valid = (y_best != -1)
        tr_mask = valid[train_idx]; va_mask = valid[val_idx]
        Xtr_raw, ytr_raw = X[train_idx][tr_mask], y_best[train_idx][tr_mask]
        Xva_raw, yva_raw = X[val_idx][va_mask],   y_best[val_idx][va_mask]
        if Xtr_raw.size == 0 or Xva_raw.size == 0 or np.unique(ytr_raw).size < 2:
            print(f"[fold {k}] WARNING: insufficient data after shift; skipping fold.")
            continue

        mu, sd, _ = get_or_compute_fold_scaler(Xtr_raw, fold_dir, force=args.force_scaler)
        Xtr = apply_zscore(Xtr_raw, mu, sd)
        Xva = apply_zscore(Xva_raw, mu, sd)
        ytr_dense, classes_seen = remap_to_dense(ytr_raw)

        train_blobs[k] = (Xtr, ytr_dense)
        val_blobs[k]   = (Xva, yva_raw)
        classes_by_fold[k] = classes_seen
        np.save(FOLD_MODELS_DIR_MODE / f"classes_seen_fold_{k}.npy", classes_seen)

    if not train_blobs:
        raise RuntimeError("No usable folds after shift/scaler stage.")

    # ---- Stage B: grid search ----
    best_hp = get_or_compute_best_hp_across_folds(
        list(train_blobs.keys()), classes_by_fold, train_blobs, val_blobs,
        grid, common, make_decoder=make_gdec_with_kernel, force=args.force_grid
    )

    # ---- Stage C: per-fold models ----
    for k in train_blobs.keys():
        model_path = FOLD_MODELS_DIR_MODE / f"fold_{k}.joblib"
        if model_path.exists() and not args.force_fold_models:
            print(f"[fold {k}] [skip] model exists: {model_path}")
        else:
            Xtr, ytr_dense = train_blobs[k]
            model_fold = make_gdec_with_kernel()
            model_fold.fit(Xtr, ytr_dense, **{**best_hp, **common})
            joblib.dump(model_fold, model_path)
            W = getattr(model_fold, "coefs_", None)
            if W is not None:
                if args.feature_mode == "lags":
                    np.save(FOLD_MODELS_DIR_MODE / f"fold_{k}_coefs_lags.npy",
                            reshape_coefs_lags(W, N_LAGS_DEFAULT, N_BANDS_DEFAULT))
                else:
                    np.save(FOLD_MODELS_DIR_MODE / f"fold_{k}_coefs_cov_mat.npy",
                            reshape_coefs_cov_ut(W))

    # ---- Stage D: patient-level union model ----
    patient_shift, src_p_shift = get_or_compute_patient_shift(fold_ids, FOLDS_DIR_MODE, OUT_DIR_MODE,
                                                              force=args.force_shift)
    print(f"[patient] chosen patient_shift={patient_shift:+d} ({src_p_shift}) from folds")

    union_X_raw, union_y_raw = [], []
    for k in fold_ids:
        X, y = FEAT["loader"](k)
        tr_idx, _ = get_or_compute_fold_split(len(y), FOLDS_DIR_MODE / f"fold_{k}", train_frac=0.8, force=False)[:2]
        y_s = shift_labels(y, patient_shift)
        valid = (y_s != -1)
        tr_mask = valid[tr_idx]
        if not np.any(tr_mask):
            continue
        union_X_raw.append(X[tr_idx][tr_mask])
        union_y_raw.append(y_s[tr_idx][tr_mask])

    X_final_raw = np.concatenate(union_X_raw, axis=0)
    y_final_raw = np.concatenate(union_y_raw, axis=0)
    mu_p, sd_p, _ = get_or_compute_patient_scaler(X_final_raw, OUT_DIR_MODE, force=args.force_scaler)
    X_final = apply_zscore(X_final_raw, mu_p, sd_p)
    y_final_dense, classes_seen_final = remap_to_dense(y_final_raw)
    np.save(OUT_DIR_MODE / "classes_seen_patient.npy", classes_seen_final)

    final_model_path = OUT_DIR_MODE / "gpmd_model_best.joblib"
    if final_model_path.exists() and not args.force_final:
        print(f"[final-gdec] [skip] model exists: {final_model_path}")
    else:
        model_final = make_gdec_with_kernel()
        print(f"[final-gdec] Training with best hp: {{**{best_hp}, **{common}}} on {X_final.shape[0]} samples")
        model_final.fit(X_final, y_final_dense, **{**best_hp, **common})
        joblib.dump(model_final, final_model_path)

        Wp = getattr(model_final, "coefs_", None)
        if Wp is not None:
            np.save(OUT_DIR_MODE / "coefs_raw.npy", Wp)
            if args.feature_mode == "lags":
                np.save(OUT_DIR_MODE / "coefs_by_lag_elec_band.npy", reshape_coefs_lags(Wp, N_LAGS_DEFAULT, N_BANDS_DEFAULT))
            else:
                np.save(OUT_DIR_MODE / "coefs_as_covmat.npy", reshape_coefs_cov_ut(Wp))
            intercept = getattr(model_final, "intercept_", None)
            if intercept is not None:
                np.save(OUT_DIR_MODE / "intercept.npy", intercept)
        else:
            (OUT_DIR_MODE / "WARNING.txt").write_text("GDEC model has no coefs_; cannot save weights.\n")

    # ---- Metadata ----
    meta = {
        "feature_mode": args.feature_mode,
        "feature_dims": {
            "lags": dict(n_lags=N_LAGS_DEFAULT, n_electrodes=N_ELECTRODES, n_bands=N_BANDS_DEFAULT,
                         n_features=N_LAGS_DEFAULT*N_ELECTRODES*N_BANDS_DEFAULT),
            "cov": dict(n_electrodes=N_ELECTRODES, n_features=(N_ELECTRODES*(N_ELECTRODES+1))//2),
        }[args.feature_mode],
        "kernel": {
            "mode": args.feature_mode,
            "type": "temporal_rbf_only" if args.feature_mode == "lags" else "none",
            "ell_lag": args.ell_lag if args.feature_mode == "lags" else None,
            "var_lag": args.var_lag if args.feature_mode == "lags" else None,
            "path": str(K_path) if K_path is not None else "",
        },
        "best_hp": {**best_hp, **common},
        "paths": {
            "final_model": str(final_model_path),
            "fold_models_dir": str(FOLD_MODELS_DIR_MODE),
            "folds_meta_root": str(FOLDS_DIR_MODE),
            "patient_shift": str(OUT_DIR_MODE / "patient_shift.npy"),
            "patient_scaler_mean": str(OUT_DIR_MODE / "patient_scaler_mean.npy"),
            "patient_scaler_std":  str(OUT_DIR_MODE / "patient_scaler_std.npy"),
            "classes_seen_patient": str(OUT_DIR_MODE / "classes_seen_patient.npy"),
            "cv_results": str(OUT_DIR_MODE / "cv_results_gdec.json"),
            "best_hp_file": str(OUT_DIR_MODE / "best_hp_gdec.json"),
        },
        "notes": [
            "Per-fold: caches split, best_shift, scaler; retrains fold model only if missing or --force-fold-models.",
            "Grid search: cached in best_hp_gdec.json/cv_results_gdec.json; reuse unless --force-grid.",
            "Patient model: majority-vote shift + patient-level scaler; cached unless --force-final.",
            "Kernels: lags → I_band ⊗ I_elec ⊗ K_lag; cov → none.",
        ],
    }
    (OUT_DIR_MODE / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[cv-gdec] Done. Final model → {final_model_path}")

if __name__ == "__main__":
    main()
