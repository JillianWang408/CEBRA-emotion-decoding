# RUN: python -m src.gdec_evaluation --feature-mode {lags|cov}
import os, sys, types, re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import joblib
import mat73, scipy.io

from sklearn.metrics import confusion_matrix, accuracy_score

from src.config import (
    MODEL_DIR, NEURAL_PATH, EMOTION_PATH, EMOTION_MAP,
    FOLD_MODELS_DIR, FOLDS_DIR, EVALUATION_OUTPUT_DIR, N_ELECTRODES
)

# ---------- JAX shim ----------
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
try:
    from jax.config import config
except Exception:
    m = types.ModuleType("jax.config"); m.config = jax.config; sys.modules["jax.config"] = m

_fold_suffix_re = re.compile(r"_(\d+)\.mat$")
def _swap_fold(path: Path, fold_id: int) -> Path:
    return path.with_name(_fold_suffix_re.sub(f"_{fold_id}.mat", path.name))

# ---------- loaders (match training) ----------
def load_xy_lags(fold_id: int, n_lags=5, n_bands=5):
    X_path = _swap_fold(Path(NEURAL_PATH), fold_id)
    y_path = _swap_fold(Path(EMOTION_PATH), fold_id)
    X = mat73.loadmat(X_path)['stim'].T.astype(np.float32)
    y = scipy.io.loadmat(y_path)['resp'].flatten().astype(np.int64)
    expected = n_lags * N_ELECTRODES * n_bands
    assert X.shape[1] == expected, f"Expected {expected} features, got {X.shape[1]} (fold {fold_id})"
    return X, y

def load_xy_cov_ut(fold_id: int):
    X_path = _swap_fold(Path(NEURAL_PATH), fold_id)
    y_path = _swap_fold(Path(EMOTION_PATH), fold_id)
    X_raw = mat73.loadmat(X_path)['stim'].T.astype(np.float32)
    y = scipy.io.loadmat(y_path)['resp'].flatten().astype(np.int64)
    if X_raw.shape[1] == N_ELECTRODES * N_ELECTRODES:
        T = X_raw.shape[0]
        M = X_raw.reshape(T, N_ELECTRODES, N_ELECTRODES)
        M = 0.5 * (M + np.swapaxes(M, 1, 2))
        iu = np.triu_indices(N_ELECTRODES, k=0)
        X = M[:, iu[0], iu[1]]
    elif X_raw.shape[1] == (N_ELECTRODES * (N_ELECTRODES + 1)) // 2:
        X = X_raw
    else:
        raise AssertionError(f"Covariance mode expects 1600 or 820 features, got {X_raw.shape[1]}")
    return X, y

def _shift_labels(y: np.ndarray, shift: int) -> np.ndarray:
    y_shift = np.full_like(y, fill_value=-1)
    if shift > 0:
        y_shift[shift:] = y[:-shift]
    elif shift < 0:
        y_shift[:shift] = y[-shift:]
    else:
        y_shift[:] = y
    return y_shift

def _ids_to_names(id_list):
    if isinstance(EMOTION_MAP, dict):
        return [EMOTION_MAP.get(int(i), str(i)) for i in id_list]
    else:
        return [EMOTION_MAP[int(i)] if int(i) < len(EMOTION_MAP) else str(i) for i in id_list]

def _plot_confusion(cm, class_ids, title, subtitle, save_path, normalize=False):
    class_names = _ids_to_names(list(class_ids))
    cm = cm.astype(float)
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=140)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion" if normalize else "Count", rotation=90)
    ax.set_title(title, pad=10)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_aspect("equal")
    vmax = np.nanmax(cm) if cm.size else 1.0
    thresh = vmax * 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            txt = f"{v:.2f}" if normalize and v > 0 else (f"{int(v)}" if (not normalize and v > 0) else "")
            if txt:
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=10, color=("white" if v > thresh else "black"))
    fig.text(0.5, 0.01, subtitle, ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight"); plt.close(fig)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-mode", choices=["lags", "cov"], default="lags",
                        help="Match the feature structure used in training")
    args = parser.parse_args()

    # ---- mode-specific subdirs to avoid collisions ----
    mode_tag = args.feature_mode  # "lags" or "cov"
    FOLD_MODELS_DIR_MODE = Path(FOLD_MODELS_DIR) / mode_tag
    FOLDS_DIR_MODE       = Path(FOLDS_DIR) / mode_tag
    RESULTS_DIR          = Path(EVALUATION_OUTPUT_DIR) / mode_tag / "cv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pid = os.environ.get("PATIENT_ID", "unknown")
    print(f"[eval_cv] Patient {pid}: feature-mode={args.feature_mode}; evaluating 5 folds with their own models / best shift / scaler")
    print(f"[eval_cv] using models from: {FOLD_MODELS_DIR_MODE}")
    print(f"[eval_cv] using fold meta from: {FOLDS_DIR_MODE}")
    print(f"[eval_cv] writing results to: {RESULTS_DIR}")

    loader = load_xy_lags if args.feature_mode == "lags" else load_xy_cov_ut
    expected_dim = (5 * N_ELECTRODES * 5) if args.feature_mode == "lags" else (N_ELECTRODES * (N_ELECTRODES + 1)) // 2

    per_fold_acc = []
    for fold_id in range(1, 6):
        model_path    = FOLD_MODELS_DIR_MODE / f"fold_{fold_id}.joblib"
        classes_path  = FOLD_MODELS_DIR_MODE / f"classes_seen_fold_{fold_id}.npy"
        fold_dir      = FOLDS_DIR_MODE / f"fold_{fold_id}"
        test_idx_path = fold_dir / "test_idx.npy"
        best_shift_path = fold_dir / "best_shift.npy"
        mu_path         = fold_dir / "scaler_mean.npy"
        std_path        = fold_dir / "scaler_std.npy"

        missing = []
        for pth, tag in [(model_path,"model"), (classes_path,"classes_seen"),
                         (test_idx_path,"test_idx"), (best_shift_path,"best_shift"),
                         (mu_path,"scaler_mean"), (std_path,"scaler_std")]:
            if not pth.exists():
                missing.append(tag)
        if missing:
            print(f"[eval_cv] WARNING: fold {fold_id} missing {missing}; skipping this fold.")
            continue

        model        = joblib.load(model_path)
        classes_seen = np.load(classes_path)
        X, y         = loader(fold_id)
        assert X.shape[1] == expected_dim, f"Fold {fold_id}: expected {expected_dim}, got {X.shape[1]}"
        test_idx     = np.load(test_idx_path)
        best_shift   = int(np.load(best_shift_path))
        mu           = np.load(mu_path).astype(np.float32)
        std          = np.load(std_path).astype(np.float32)

        y_shifted = _shift_labels(y, best_shift)
        valid = (y_shifted != -1)
        test_idx_valid = test_idx[valid[test_idx]]
        if test_idx_valid.size == 0:
            print(f"[eval_cv] fold {fold_id}: no valid test rows after applying shift {best_shift:+d}; skipping")
            continue

        X_test = X[test_idx_valid]
        y_test = y_shifted[test_idx_valid]

        mask_seen = np.isin(y_test, classes_seen)
        if not np.any(mask_seen):
            print(f"[eval_cv] fold {fold_id}: no test samples in seen classes; skipping")
            continue
        Xs, ys = X_test[mask_seen], y_test[mask_seen]

        std_safe = std.copy(); std_safe[std_safe < 1e-8] = 1.0
        Xs = (Xs - mu) / std_safe

        y_pred_dense = model.predict(Xs)
        y_pred = classes_seen[y_pred_dense]

        acc = accuracy_score(ys, y_pred)
        per_fold_acc.append(acc)
        labels = classes_seen
        cm_counts = confusion_matrix(ys, y_pred, labels=labels)
        row_sums = cm_counts.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm_counts, row_sums, out=np.zeros_like(cm_counts, dtype=float), where=row_sums != 0)

        np.save(RESULTS_DIR / f"fold_{fold_id}_cm_counts.npy", cm_counts)
        np.save(RESULTS_DIR / f"fold_{fold_id}_cm_norm.npy", cm_norm)
        np.save(RESULTS_DIR / f"fold_{fold_id}_y_test_seen.npy", ys)
        np.save(RESULTS_DIR / f"fold_{fold_id}_y_pred_seen.npy", y_pred)

        supports = cm_counts.sum(axis=1)
        subtitle = (f"Patient: {pid} | Fold: {fold_id} | Shift: {best_shift:+d} | "
                    f"Valid classes: {labels.tolist()} | Support per row: {supports.tolist()} | "
                    f"In-vocab acc: {acc:.3f}")
        _plot_confusion(cm_counts, labels,
                        f"Confusion (counts) — fold {fold_id}",
                        subtitle,
                        RESULTS_DIR / f"fold_{fold_id}_cm_counts.png",
                        normalize=False)
        _plot_confusion(cm_norm, labels,
                        f"Confusion (row-normalized) — fold {fold_id}",
                        subtitle,
                        RESULTS_DIR / f"fold_{fold_id}_cm_norm.png",
                        normalize=True)

        print(f"[eval_cv] fold {fold_id}: in-vocab acc={acc:.3f}")

    if per_fold_acc:
        mean_acc, std_acc = float(np.mean(per_fold_acc)), float(np.std(per_fold_acc))
        np.save(RESULTS_DIR / "acc_per_fold.npy", np.array(per_fold_acc, dtype=float))
        with open(RESULTS_DIR / "summary.txt", "w") as f:
            f.write(f"Patient {pid}\n")
            f.write(f"Per-fold in-vocab acc: {per_fold_acc}\n")
            f.write(f"Mean ± SD: {mean_acc:.4f} ± {std_acc:.4f}\n")
        print(f"[eval_cv] mean in-vocab acc over folds: {mean_acc:.3f} ± {std_acc:.3f}")
        print(f"[eval_cv] saved results to {RESULTS_DIR}")
    else:
        print("[eval_cv] no folds evaluated (missing artifacts or no seen-class samples).")

if __name__ == "__main__":
    main()
