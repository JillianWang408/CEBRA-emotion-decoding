import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import joblib
import mat73, scipy.io
import matplotlib.pyplot as plt
import re

from sklearn.metrics import confusion_matrix, accuracy_score

from src.config import (
    DATA_DIR, MODEL_DIR, NEURAL_PATH, EMOTION_PATH, N_ELECTRODES,
    FULL_NEURAL_PATH, FULL_EMOTION_PATH, EVALUATION_OUTPUT_DIR, EMOTION_MAP
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
    
import gdec # safe after shim


# ---------- Paths ----------
OUT_DIR = MODEL_DIR / "gdec_gpmd"                 
FOLD_MODELS_DIR = OUT_DIR / "fold_models"
FOLDS_DIR = DATA_DIR                 
RESULTS_DIR = Path(EVALUATION_OUTPUT_DIR) / "cv"   
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Robust fold-id swapping (handles ..._1.mat, ...movHeldOut_1.mat, etc.) ----------
_fold_suffix_re = re.compile(r"_(\d+)\.mat$")

def _swap_fold(path: Path, fold_id: int) -> Path:
    new_name = _fold_suffix_re.sub(f"_{fold_id}.mat", path.name)
    return path.with_name(new_name)

def _load_fold_xy(fold_id: int):
    X_path = _swap_fold(Path(NEURAL_PATH), fold_id)
    y_path = _swap_fold(Path(EMOTION_PATH), fold_id)
    X = mat73.loadmat(X_path)['stim'].T.astype(np.float32)           # [T, F]
    y = scipy.io.loadmat(y_path)['resp'].flatten().astype(np.int64)  # [T]
    return X, y

# ---------- Label names ----------
def _ids_to_names(id_list):
    # EMOTION_MAP can be dict or list-like
    if isinstance(EMOTION_MAP, dict):
        return [EMOTION_MAP.get(int(i), str(i)) for i in id_list]
    else:
        return [EMOTION_MAP[int(i)] if int(i) < len(EMOTION_MAP) else str(i) for i in id_list]

# ---------- Plotting (Blues, emotion labels, row/col ticks, clean annotation) ----------
def _plot_confusion(cm, class_ids, title, subtitle, save_path, normalize=False):
    class_names = _ids_to_names(list(class_ids))
    cm = cm.astype(float)

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=140)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion" if normalize else "Count", rotation=90)

    ax.set_title(title, pad=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
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
    pid = os.environ.get("PATIENT_ID", "unknown")
    print(f"[eval_cv] Patient {pid}: evaluating 5 folds with their own models / splits")

    per_fold_acc = []
    for fold_id in range(1, 6):
        model_path = FOLD_MODELS_DIR / f"fold_{fold_id}.joblib"
        classes_path = FOLD_MODELS_DIR / f"classes_seen_fold_{fold_id}.npy"
        split_dir = FOLDS_DIR / f"fold_{fold_id}"
        test_idx_path = split_dir / "test_idx.npy"

        # Existence checks
        if not model_path.exists():
            print(f"[eval_cv] WARNING: missing model for fold {fold_id} → {model_path}")
            continue
        if not classes_path.exists():
            print(f"[eval_cv] WARNING: missing classes_seen for fold {fold_id} → {classes_path}")
            continue
        if not test_idx_path.exists():
            print(f"[eval_cv] WARNING: missing test_idx for fold {fold_id} → {test_idx_path}")
            continue

        # Load artifacts and data for THIS fold
        model = joblib.load(model_path)
        classes_seen = np.load(classes_path)
        X, y = _load_fold_xy(fold_id)
        test_idx = np.load(test_idx_path)

        X_test, y_test = X[test_idx], y[test_idx]
        mask_seen = np.isin(y_test, classes_seen)
        if not np.any(mask_seen):
            print(f"[eval_cv] fold {fold_id}: no test samples in seen classes; skipping")
            continue
        Xs, ys = X_test[mask_seen], y_test[mask_seen]

        # Predict (model outputs dense indices 0..K_seen-1); map back to original label IDs
        y_pred_dense = model.predict(Xs)
        y_pred = classes_seen[y_pred_dense]

        # Metrics and matrices
        acc = accuracy_score(ys, y_pred)
        per_fold_acc.append(acc)
        labels = classes_seen
        cm_counts = confusion_matrix(ys, y_pred, labels=labels)
        row_sums = cm_counts.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm_counts, row_sums, out=np.zeros_like(cm_counts, dtype=float), where=row_sums != 0)

        # Save arrays
        np.save(RESULTS_DIR / f"fold_{fold_id}_cm_counts.npy", cm_counts)
        np.save(RESULTS_DIR / f"fold_{fold_id}_cm_norm.npy", cm_norm)
        np.save(RESULTS_DIR / f"fold_{fold_id}_y_test_seen.npy", ys)
        np.save(RESULTS_DIR / f"fold_{fold_id}_y_pred_seen.npy", y_pred)

        # Plots
        supports = cm_counts.sum(axis=1)
        subtitle = f"Patient: {pid} | Fold: {fold_id} | Valid: {labels.tolist()} | Support per row: {supports.tolist()} | In-vocab acc: {acc:.3f}"
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

    # Summary
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
        print("[eval_cv] no folds evaluated (missing models/splits or no seen-class samples).")

if __name__ == "__main__":
    main()