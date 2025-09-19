# RUN: python -u -m src.glmnet_training --feature-mode {lags|cov}
import os, sys, types, re, json, inspect, argparse
from pathlib import Path
import numpy as np
import joblib
import mat73, scipy.io
from sklearn.linear_model import LogisticRegression as SkLogReg

from src.config import (
    MODEL_DIR, DATA_DIR, NEURAL_PATH, EMOTION_PATH, N_ELECTRODES,
    OUT_DIR, FOLDS_DIR, FOLD_MODELS_DIR
)

# ---------- JAX (CPU) shim (some libs import jax.config) ----------
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

# ---------- Layout defaults ----------
N_LAGS_DEFAULT  = 5
N_BANDS_DEFAULT = 5

# Ensure base dirs exist
OUT_DIR.mkdir(parents=True, exist_ok=True)
FOLD_MODELS_DIR.mkdir(parents=True, exist_ok=True)
FOLDS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Helpers: folds / loading ----------
_fold_suffix_re = re.compile(r"_(\d+)\.mat$")
def _swap_fold(path: Path, fold_id: int) -> Path:
    return path.with_name(_fold_suffix_re.sub(f"_{fold_id}.mat", path.name))

def load_xy_from_fold_lags(fold_id: int, n_lags: int, n_bands: int):
    """Expect stim [T, n_lags * N_ELECTRODES * n_bands] in MAT (lag-major)."""
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

    if X_raw.shape[1] == N_ELECTRODES * N_ELECTRODES:  # 1600 (40x40)
        T = X_raw.shape[0]
        M = X_raw.reshape(T, N_ELECTRODES, N_ELECTRODES)
        # enforce symmetry just in case
        M = 0.5 * (M + np.swapaxes(M, 1, 2))
        iu = np.triu_indices(N_ELECTRODES, k=0)
        X = M[:, iu[0], iu[1]]  # [T, 820]
    elif X_raw.shape[1] == (N_ELECTRODES * (N_ELECTRODES + 1)) // 2:  # 820
        X = X_raw
    else:
        raise AssertionError(f"Covariance mode expects 1600 or 820 features, got {X_raw.shape[1]}")
    return X.astype(np.float32), y

# ---------- Split / scale / labels ----------
def split_train_val_idx(n: int, train_frac=0.8):
    cut = int(n * train_frac)
    return np.arange(0, cut, dtype=np.int64), np.arange(cut, n, dtype=np.int64)

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

# ---------- Attribution reshapes ----------
def reshape_coefs_lags(coefs_flat: np.ndarray, n_lags: int, n_bands: int) -> np.ndarray:
    """
    [K, F] -> [K, n_lags, N_ELECTRODES, n_bands]
    Assumes lag-major flatten: idx = l*(E*B) + e*B + b
    """
    Kc, F = coefs_flat.shape
    expected = n_lags * N_ELECTRODES * n_bands
    assert F == expected, f"Expected {expected}, got {F}"
    return coefs_flat.reshape(Kc, n_lags, N_ELECTRODES, n_bands)

def reshape_coefs_cov_ut(coefs_ut: np.ndarray) -> np.ndarray:
    """
    [K, 820] -> [K, 40, 40] symmetric matrices.
    Upper-triangle order from np.triu_indices.
    """
    Kc, F = coefs_ut.shape
    n = N_ELECTRODES
    expected = (n * (n + 1)) // 2
    assert F == expected, f"Expected {expected}, got {F}"
    iu = np.triu_indices(n, k=0)
    out = np.zeros((Kc, n, n), dtype=coefs_ut.dtype)
    for k in range(Kc):
        M = np.zeros((n, n), dtype=coefs_ut.dtype)
        M[iu] = coefs_ut[k]
        M = M + np.triu(M, 1).T
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

# ---------- GLMNET factory ----------
def _make_glmnet(C, l1_ratio, *, max_iter=20000, tol=1e-3,
                 class_weight="balanced", n_jobs=-1, random_state=0):
    # elastic-net requires saga
    return SkLogReg(
        penalty="elasticnet",
        solver="saga",
        C=C,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        tol=tol,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=random_state
    )

# ---------- Shift sweep using a small fixed glmnet ----------
def run_label_shift_sweep_glmnet(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    shift_range=range(-5, 6),
    sweep_hp=dict(C=1.0, l1_ratio=0.5, max_iter=5000, class_weight="balanced"),
):
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

        m = _make_glmnet(**sweep_hp)
        m.fit(Xtr, ytr_dense)
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

def get_or_compute_fold_shift(X, y, train_idx, fold_dir: Path, shift_range, force=False):
    p_best = fold_dir / "best_shift.npy"
    if p_best.exists() and not force:
        return int(np.load(p_best)), "cache"
    cut_val = int(len(train_idx) * 0.9)
    tr_sub = train_idx[:cut_val]; va_sub = train_idx[cut_val:]
    best_shift, _ = run_label_shift_sweep_glmnet(X, y, tr_sub, va_sub, shift_range=shift_range)
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
                                        grid, common, out_dir: Path, force=False):
    p_cv  = out_dir / "cv_results_glmnet.json"
    p_best = out_dir / "best_hp_glmnet.json"
    if p_best.exists() and p_cv.exists() and not force:
        best_hp = json.loads(p_best.read_text())
        print(f"[cv-glmnet] using cached best hp: {best_hp}")
        return best_hp

    cv_results = []
    for hp in grid:
        fold_scores = []
        for k in fold_ids:
            (Xtr, ytr_dense) = train_blobs[k]
            (Xva, yva_raw)   = val_blobs[k]
            classes_seen     = classes_seen_by_fold[k]

            model = _make_glmnet(
                C=hp["C"], l1_ratio=hp["l1_ratio"],
                max_iter=common["max_iter"],
                tol=common["tol"],
                class_weight=common["class_weight"],
                n_jobs=common["n_jobs"],
                random_state=common["random_state"],
            )
            model.fit(Xtr, ytr_dense)
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
    print(f"[cv-glmnet] Best mean val acc={best['mean_val_acc']:.3f} with {best_hp}")
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

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-mode", choices=["lags", "cov"], default="lags",
                        help="Use lag features (5x40x5=1000) or covariance upper-triangle (820).")
    parser.add_argument("--force", action="store_true", help="Force re-run of all stages")
    parser.add_argument("--force-shift", action="store_true", help="Force recompute label shift")
    parser.add_argument("--force-scaler", action="store_true", help="Force recompute scalers")
    parser.add_argument("--force-grid", action="store_true", help="Force recompute hyperparameter CV")
    parser.add_argument("--force-fold-models", action="store_true", help="Force retrain per-fold models")
    parser.add_argument("--force-final", action="store_true", help="Force retrain patient union model")
    parser.add_argument("--shift-min", type=int, default=-5)
    parser.add_argument("--shift-max", type=int, default=5)
    args = parser.parse_args()

    if args.force:
        args.force_shift = args.force_scaler = args.force_grid = True
        args.force_fold_models = args.force_final = True

    # Mode-scoped dirs to avoid collisions
    mode_tag = args.feature_mode
    OUT_DIR_M         = OUT_DIR / mode_tag
    FOLDS_DIR_M       = FOLDS_DIR / mode_tag
    FOLD_MODELS_DIR_M = FOLD_MODELS_DIR / mode_tag
    OUT_DIR_M.mkdir(parents=True, exist_ok=True)
    FOLDS_DIR_M.mkdir(parents=True, exist_ok=True)
    FOLD_MODELS_DIR_M.mkdir(parents=True, exist_ok=True)

    pid = os.environ.get("PATIENT_ID", "unknown")
    print(f"[cv-glmnet] Patient {pid} — feature-mode={args.feature_mode} | folds via _1.mat.._5.mat")

    # Hyperparameter grid for GLMNET
    grid = [dict(C=C, l1_ratio=a)
            for C in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
            for a in [0.5, 0.9]]
    common = dict(max_iter=20000, tol=1e-3, class_weight="balanced", n_jobs=-1, random_state=0)

    # Feature registry
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

    # ---- Stage A: per-fold split, shift, scaler ----
    fold_ids = range(1, 6)
    train_blobs = {}           # k -> (Xtr_z, ytr_dense)
    val_blobs   = {}           # k -> (Xva_z, yva_raw)
    classes_by_fold = {}       # k -> classes_seen
    for k in fold_ids:
        X, y = FEAT["loader"](k)
        assert X.shape[1] == FEAT["expected_dim"], f"Fold {k}: expected {FEAT['expected_dim']}, got {X.shape[1]}"
        fold_dir = FOLDS_DIR_M / f"fold_{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # split (cache)
        train_idx, val_idx, _ = get_or_compute_fold_split(len(y), fold_dir, train_frac=0.8, force=False)

        # shift (cache) — uses a small fixed glmnet to rank shifts
        best_shift, src_shift = get_or_compute_fold_shift(
            X, y, train_idx, fold_dir,
            shift_range=range(args.shift_min, args.shift_max + 1),
            force=args.force_shift
        )
        print(f"[fold {k}] best_shift={best_shift:+d} ({src_shift})")

        # shifted TRAIN/VAL, scaler on TRAIN (cache)
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

        # persist classes_seen for evaluation
        np.save(FOLD_MODELS_DIR_M / f"classes_seen_fold_{k}.npy", classes_seen)

    if not train_blobs:
        raise RuntimeError("No usable folds after shift/scaler stage.")

    # ---- Stage B: grid search across folds (cache) ----
    best_hp = get_or_compute_best_hp_across_folds(
        list(train_blobs.keys()), classes_by_fold, train_blobs, val_blobs,
        grid, common, out_dir=OUT_DIR_M, force=args.force_grid
    )

    # ---- Stage C: per-fold models (cache) ----
    for k in train_blobs.keys():
        model_path = FOLD_MODELS_DIR_M / f"fold_{k}.joblib"
        if model_path.exists() and not args.force_fold_models:
            print(f"[fold {k}] [skip] model exists: {model_path}")
        else:
            Xtr, ytr_dense = train_blobs[k]
            model_fold = _make_glmnet(
                C=best_hp["C"], l1_ratio=best_hp["l1_ratio"],
                max_iter=common["max_iter"], tol=common["tol"],
                class_weight=common["class_weight"],
                n_jobs=common["n_jobs"], random_state=common["random_state"]
            )
            model_fold.fit(Xtr, ytr_dense)
            joblib.dump(model_fold, model_path)

            # optional per-fold weights for attribution
            coefs_f = getattr(model_fold, "coefs_", getattr(model_fold, "coef_", None))
            if coefs_f is not None:
                if args.feature_mode == "lags":
                    np.save(FOLD_MODELS_DIR_M / f"fold_{k}_coefs_by_lag_elec_band.npy",
                            reshape_coefs_lags(coefs_f, N_LAGS_DEFAULT, N_BANDS_DEFAULT))
                else:
                    np.save(FOLD_MODELS_DIR_M / f"fold_{k}_coefs_as_covmat.npy",
                            reshape_coefs_cov_ut(coefs_f))

    # ---- Stage D: patient-level union model (consistent shift + scaler; cache) ----
    patient_shift, src_p_shift = get_or_compute_patient_shift(fold_ids, FOLDS_DIR_M, OUT_DIR_M,
                                                              force=args.force_shift)
    print(f"[patient] chosen patient_shift={patient_shift:+d} ({src_p_shift}) from folds")

    # build union-of-train across folds using patient_shift; compute patient scaler (cache)
    union_X_raw, union_y_raw = [], []
    for k in fold_ids:
        X, y = FEAT["loader"](k)
        tr_idx, _ = get_or_compute_fold_split(len(y), FOLDS_DIR_M / f"fold_{k}",
                                              train_frac=0.8, force=False)[:2]
        y_s = shift_labels(y, patient_shift)
        valid = (y_s != -1)
        tr_mask = valid[tr_idx]
        if not np.any(tr_mask):
            continue
        union_X_raw.append(X[tr_idx][tr_mask])
        union_y_raw.append(y_s[tr_idx][tr_mask])

    X_final_raw = np.concatenate(union_X_raw, axis=0)
    y_final_raw = np.concatenate(union_y_raw, axis=0)
    mu_p, sd_p, _ = get_or_compute_patient_scaler(X_final_raw, OUT_DIR_M, force=args.force_scaler)
    X_final = apply_zscore(X_final_raw, mu_p, sd_p)
    y_final_dense, classes_seen_final = remap_to_dense(y_final_raw)
    np.save(OUT_DIR_M / "classes_seen_patient.npy", classes_seen_final)

    final_model_path = OUT_DIR_M / "glmnet_model_best.joblib"
    if final_model_path.exists() and not args.force_final:
        print(f"[final-glmnet] [skip] model exists: {final_model_path}")
        model_final = joblib.load(final_model_path)
    else:
        model_final = _make_glmnet(
            C=best_hp["C"], l1_ratio=best_hp["l1_ratio"],
            max_iter=common["max_iter"], tol=common["tol"],
            class_weight=common["class_weight"],
            n_jobs=common["n_jobs"], random_state=common["random_state"]
        )
        print(f"[final-glmnet] Training with best hp: {{**{best_hp}, **{common}}} on {X_final.shape[0]} samples")
        model_final.fit(X_final, y_final_dense)
        joblib.dump(model_final, final_model_path)

        coefs_attr = getattr(model_final, "coefs_", getattr(model_final, "coef_", None))
        if coefs_attr is not None:
            if args.feature_mode == "lags":
                np.save(OUT_DIR_M / "coefs_by_lag_elect_band.npy",
                        reshape_coefs_lags(coefs_attr, N_LAGS_DEFAULT, N_BANDS_DEFAULT))
            else:
                np.save(OUT_DIR_M / "coefs_as_covmat.npy",
                        reshape_coefs_cov_ut(coefs_attr))
            intercept = getattr(model_final, "intercept_", None)
            if intercept is not None:
                np.save(OUT_DIR_M / "intercept.npy", intercept)
        else:
            (OUT_DIR_M / "WARNING.txt").write_text("GLMNET model has no coef(s); cannot save weights.\n")

    # ---- Metadata ----
    meta = {
        "feature_mode": args.feature_mode,
        "feature_dims": {
            "lags": dict(n_lags=N_LAGS_DEFAULT, n_electrodes=N_ELECTRODES, n_bands=N_BANDS_DEFAULT,
                         n_features=N_LAGS_DEFAULT*N_ELECTRODES*N_BANDS_DEFAULT),
            "cov": dict(n_electrodes=N_ELECTRODES, n_features=(N_ELECTRODES*(N_ELECTRODES+1))//2),
        }[args.feature_mode],
        "best_hp": {**best_hp, **common},
        "paths": {
            "final_model": str(final_model_path),
            "fold_models_dir": str(FOLD_MODELS_DIR_M),
            "folds_meta_root": str(FOLDS_DIR_M),
            "patient_shift": str(OUT_DIR_M / "patient_shift.npy"),
            "patient_scaler_mean": str(OUT_DIR_M / "patient_scaler_mean.npy"),
            "patient_scaler_std":  str(OUT_DIR_M / "patient_scaler_std.npy"),
            "classes_seen_patient": str(OUT_DIR_M / "classes_seen_patient.npy"),
            "cv_results": str(OUT_DIR_M / "cv_results_glmnet.json"),
            "best_hp_file": str(OUT_DIR_M / "best_hp_glmnet.json"),
        },
        "notes": [
            "Per-fold: caches split, best_shift, scaler; retrain fold model only if missing or --force-fold-models.",
            "Grid search: cached in best_hp_glmnet.json/cv_results_glmnet.json; reuse unless --force-grid.",
            "Patient model: uses majority-vote shift and patient-level scaler; cached unless --force-final.",
            "Feature modes: lags (1000 features) or cov (820 upper-triangle, auto-vech from 1600 if given).",
        ],
    }
    (OUT_DIR_M / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[cv-glmnet] Done. Final model → {final_model_path}")

if __name__ == "__main__":
    main()
