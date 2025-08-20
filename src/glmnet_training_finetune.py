# RUN: python -u -m src.glmnet_training
import os, sys, types, re, json, inspect
from pathlib import Path
import numpy as np
import joblib
import mat73, scipy.io
from sklearn.linear_model import LogisticRegression as SkLogReg
import argparse

from src.config import (
    MODEL_DIR, DATA_DIR, NEURAL_PATH, EMOTION_PATH, N_ELECTRODES, OUT_DIR, FOLDS_DIR, FOLD_MODELS_DIR
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

# ---------- Layout constants ----------
N_LAGS, N_BANDS = 5, 5
N_FEATURES = N_LAGS * N_ELECTRODES * N_BANDS  # 1000 expected
FEATURE_ORDERING = "lag_first_then_band_within_each_electrode"

# Output dirs (GLMNET under its own subdir)
OUT_DIR.mkdir(parents=True, exist_ok=True)
FOLD_MODELS_DIR.mkdir(parents=True, exist_ok=True)
FOLDS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Helpers ----------
_fold_suffix_re = re.compile(r"_(\d+)\.mat$")
def _swap_fold(path: Path, fold_id: int) -> Path:
    """Replace trailing '_<num>.mat' with '_{fold_id}.mat'."""
    return path.with_name(_fold_suffix_re.sub(f"_{fold_id}.mat", path.name))

def load_xy_from_fold(fold_id: int):
    """Load X,y for a given fold from its own MAT files."""
    X_path = _swap_fold(Path(NEURAL_PATH), fold_id)
    y_path = _swap_fold(Path(EMOTION_PATH), fold_id)
    X = mat73.loadmat(X_path)['stim'].T.astype(np.float32)           # [T, 1000]
    y = scipy.io.loadmat(y_path)['resp'].flatten().astype(np.int64)  # [T]
    assert X.shape[1] == N_FEATURES, f"Expected {N_FEATURES} features, got {X.shape[1]} (fold {fold_id})"
    return X, y

def split_train_val_idx(n: int, train_frac=0.8):
    """Return indices for first 80% as train (data already shuffled), rest as val."""
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

# replace your reshape_coefs_for_attrib() with this lag-major version
def reshape_coefs_for_attrib(coefs_flat: np.ndarray) -> np.ndarray:
    """
    coefs_flat: [K_seen, N_FEATURES]
    return: [K_seen, N_LAGS, N_ELECTRODES, N_BANDS]
    Order: LAG-MAJOR (first all channels for lag0, then lag1, ...)
    flat_idx = l*(E*B) + e*B + b
    """
    K, F = coefs_flat.shape
    E = N_ELECTRODES
    B = N_BANDS
    L = N_LAGS
    assert F == L * E * B, f"Expected {L*E*B}, got {F}"

    # build index map [L, E, B] -> flat
    idx_map = np.zeros((L, E, B), dtype=int)
    for l in range(L):
        base_l = l * (E * B)
        for e in range(E):
            base_le = base_l + e * B
            for b in range(B):
                idx_map[l, e, b] = base_le + b

    coefs_tensor = np.zeros((K, L, E, B), dtype=coefs_flat.dtype)
    for k in range(K):
        coefs_tensor[k] = coefs_flat[k, idx_map]
    return coefs_tensor

# ---------- Label shift helpers ----------
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

def run_label_shift_sweep_glmnet(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    shift_range=range(-5, 6),
    # fixed small model to rank shifts (kept constant so we can cache the shift)
    sweep_hp=dict(C=1.0, l1_ratio=0.5, max_iter=5000, class_weight="balanced"),
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

# ---------- Caching utilities ----------
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
    # small validation-from-TRAIN for shift search (last 10% of TRAIN)
    cut_val = int(len(train_idx) * 0.9)
    tr_sub = train_idx[:cut_val]
    va_sub = train_idx[cut_val:]
    best_shift, best_val_acc = run_label_shift_sweep_glmnet(
        X, y, tr_sub, va_sub, shift_range=shift_range
    )
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
                                        grid, common, force=False):
    """
    Grid search over (C,l1_ratio) using shifted+standardized blobs provided per fold.
    train_blobs[k] = (Xtr, ytr_dense); val_blobs[k] = (Xva, yva_raw)
    classes_seen_by_fold[k] maps dense->original for fold k.
    """
    p_cv = OUT_DIR / "cv_results.json"
    p_best = OUT_DIR / "best_hp.json"
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force re-run of all stages")
    parser.add_argument("--force-shift", action="store_true", help="Force recompute label shift")
    parser.add_argument("--force-scaler", action="store_true", help="Force recompute scalers")
    parser.add_argument("--force-grid", action="store_true", help="Force recompute hyperparameter CV")
    parser.add_argument("--force-fold-models", action="store_true", help="Force retrain per-fold models")
    parser.add_argument("--force-final", action="store_true", help="Force retrain patient union model")
    parser.add_argument("--shift-min", type=int, default=-5)
    parser.add_argument("--shift-max", type=int, default=5)
    args = parser.parse_args()

    # coarse global force flips everything on
    if args.force:
        args.force_shift = args.force_scaler = args.force_grid = True
        args.force_fold_models = args.force_final = True

    pid = os.environ.get("PATIENT_ID", "unknown")
    print(f"[cv-glmnet] Patient {pid} — using 5 folds via file suffixes _1.mat.._5.mat")

    # ---- Hyperparameter grid for GLMNET ----
    # (slightly richer Cs to help convergence/regularization; EN with two l1 ratios)
    grid = [dict(C=C, l1_ratio=a)
            for C in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
            for a in [0.5, 0.9]]
    common = dict(max_iter=20000, tol=1e-3, class_weight="balanced", n_jobs=-1, random_state=0)

    # ---- Stage A: per-fold split, shift, scaler, cache ----
    fold_ids = range(1, 6)
    train_blobs = {}           # k -> (Xtr_z, ytr_dense)
    val_blobs   = {}           # k -> (Xva_z, yva_raw)
    classes_by_fold = {}       # k -> classes_seen
    for k in fold_ids:
        X, y = load_xy_from_fold(k)
        fold_dir = FOLDS_DIR / f"fold_{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # split (cache)
        train_idx, val_idx, src_split = get_or_compute_fold_split(len(y), fold_dir, train_frac=0.8,
                                                                  force=False)
        # shift (cache)
        best_shift, src_shift = get_or_compute_fold_shift(
            X, y, train_idx, fold_dir, shift_range=range(args.shift_min, args.shift_max + 1),
            force=args.force_shift
        )
        print(f"[fold {k}] best_shift={best_shift:+d} ({src_shift})")

        # build shifted TRAIN/VAL and compute scaler on TRAIN (cache)
        y_best = shift_labels(y, best_shift)
        valid = (y_best != -1)
        tr_mask = valid[train_idx]
        va_mask = valid[val_idx]
        Xtr_raw, ytr_raw = X[train_idx][tr_mask], y_best[train_idx][tr_mask]
        Xva_raw, yva_raw = X[val_idx][va_mask],   y_best[val_idx][va_mask]
        if Xtr_raw.size == 0 or Xva_raw.size == 0 or np.unique(ytr_raw).size < 2:
            print(f"[fold {k}] WARNING: insufficient data after shift; skipping fold.")
            continue

        mu, sd, src_scaler = get_or_compute_fold_scaler(Xtr_raw, fold_dir, force=args.force_scaler)
        Xtr = apply_zscore(Xtr_raw, mu, sd)
        Xva = apply_zscore(Xva_raw, mu, sd)
        ytr_dense, classes_seen = remap_to_dense(ytr_raw)

        # blobs for grid CV
        train_blobs[k] = (Xtr, ytr_dense)
        val_blobs[k]   = (Xva, yva_raw)
        classes_by_fold[k] = classes_seen

        # persist classes_seen for evaluation
        np.save(FOLD_MODELS_DIR / f"classes_seen_fold_{k}.npy", classes_seen)

    # If nothing usable, bail out
    if not train_blobs:
        raise RuntimeError("No usable folds after shift/scaler stage.")

    # ---- Stage B: grid search across folds (cache) ----
    best_hp = get_or_compute_best_hp_across_folds(
        list(train_blobs.keys()), classes_by_fold, train_blobs, val_blobs,
        grid, common, force=args.force_grid
    )

    # ---- Stage C: per-fold models (cache) ----
    for k in train_blobs.keys():
        model_path = FOLD_MODELS_DIR / f"fold_{k}.joblib"
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
                np.save(FOLD_MODELS_DIR / f"fold_{k}_coefs_raw.npy", coefs_f)
                np.save(FOLD_MODELS_DIR / f"fold_{k}_coefs_by_lag_elec_band.npy",
                        reshape_coefs_for_attrib(coefs_f))

    # ---- Stage D: patient-level union model (consistent shift + scaler; cache) ----
    # majority vote patient shift from cached per-fold best_shift.npy
    patient_shift, src_p_shift = get_or_compute_patient_shift(fold_ids, FOLDS_DIR, OUT_DIR,
                                                              force=args.force_shift)
    print(f"[patient] chosen patient_shift={patient_shift:+d} ({src_p_shift}) from folds")

    # build union-of-train across folds using patient_shift; compute patient scaler (cache)
    union_X_raw = []
    union_y_raw = []
    for k in fold_ids:
        X, y = load_xy_from_fold(k)
        tr_idx, _ = get_or_compute_fold_split(len(y), FOLDS_DIR / f"fold_{k}",
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
    mu_p, sd_p, src_p_scaler = get_or_compute_patient_scaler(X_final_raw, OUT_DIR,
                                                             force=args.force_scaler)
    X_final = apply_zscore(X_final_raw, mu_p, sd_p)
    y_final_dense, classes_seen_final = remap_to_dense(y_final_raw)
    np.save(OUT_DIR / "classes_seen_patient.npy", classes_seen_final)

    final_model_path = OUT_DIR / "glmnet_model_best.joblib"
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
            np.save(OUT_DIR / "coefs_raw.npy", coefs_attr)
            np.save(OUT_DIR / "coefs_by_lag_elect_band.npy", reshape_coefs_for_attrib(coefs_attr))
            intercept = getattr(model_final, "intercept_", None)
            if intercept is not None:
                np.save(OUT_DIR / "intercept.npy", intercept)
        else:
            (OUT_DIR / "WARNING.txt").write_text("GLMNET model has no coef(s); cannot save weights.\n")

    # ---- Metadata ----
    meta = {
        "feature_dims": {
            "n_lags": N_LAGS, "n_electrodes": N_ELECTRODES, "n_bands": N_BANDS, "n_features": N_FEATURES
        },
        "feature_ordering_assumption": FEATURE_ORDERING,
        "best_hp": {**best_hp, **common},
        "paths": {
            "final_model": str(final_model_path),
            "fold_models_dir": str(FOLD_MODELS_DIR),
            "folds_meta_root": str(FOLDS_DIR),
            "patient_shift": str(OUT_DIR / "patient_shift.npy"),
            "patient_scaler_mean": str(OUT_DIR / "patient_scaler_mean.npy"),
            "patient_scaler_std":  str(OUT_DIR / "patient_scaler_std.npy"),
            "classes_seen_patient": str(OUT_DIR / "classes_seen_patient.npy"),
            "cv_results": str(OUT_DIR / "cv_results.json"),
        },
        "notes": [
            "Per-fold: caches split, best_shift, scaler; retrain fold model only if missing or --force-fold-models.",
            "Grid search: cached in best_hp.json/cv_results.json; reuse unless --force-grid.",
            "Patient model: uses majority-vote shift and patient-level scaler; cached unless --force-final.",
        ],
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"[cv-glmnet] Done. Final model → {final_model_path}")

if __name__ == "__main__":
    main()