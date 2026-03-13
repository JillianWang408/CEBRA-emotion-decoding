"""
DPAD pipeline for 9-emotion decoding (xCEBRA).

Uses calc (train) and pred (test) files - same split as single_emotion.

Run from project root with DPAD env activated:
  pip install -r requirements-dpad.txt
  python -m src.DPAD.DPAD_9emotion --patient-id 1 [--skip-flexible]

  # Two-stage gating (gate: no vs emotion, then emotion head):
  python -m src.DPAD.DPAD_9emotion --patient-id 1 --two-stage

  # Override: single file with temporal split (80/20):
  python -m src.DPAD.DPAD_9emotion --neural-path <path> --emotion-path <path>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

# TF/keras setup before DPAD (standalone keras in TF 2.15+)
import tensorflow as tf
import keras
if not hasattr(tf, "keras"):
    tf.keras = keras
# Legacy Adam avoids v2.11+ slowdown on M1/M2 Macs (must run before DPAD imports)
if hasattr(tf.keras.optimizers, "legacy") and hasattr(tf.keras.optimizers.legacy, "Adam"):
    tf.keras.optimizers.Adam = tf.keras.optimizers.legacy.Adam

from DPAD import DPADModel
from DPAD.tools.tools import get_one_hot
from DPAD.tools.flexible import (
    fitDPADWithFlexibleNonlinearity,
    prepareHyperParameterSearchSpaceFromMethodCode,
)

# Monkey-patch: DPAD validation log passes Z as (T,n_z) int indices to CategoricalCrossentropy
# which expects one-hot. Convert to one-hot so verbose=True works.
import importlib
_dpad_model_module = importlib.import_module("DPAD.DPADModel")
_original_getLossLogStr = _dpad_model_module.getLossLogStr


def _patched_getLossLogStr(trueVals, predVals, steps, sigType, lossFuncs):
    if sigType == "cat":
        true_list = [trueVals] if not isinstance(trueVals, (list, tuple)) else list(trueVals)
        pred_list = [predVals] if not isinstance(predVals, (list, tuple)) else list(predVals)
        converted = []
        for tv, pv in zip(true_list, pred_list):
            tv, pv = np.asarray(tv), np.asarray(pv)
            # (T, n_z) int indices -> (T, n_z, n_classes) one-hot; get n_classes from pred
            if tv.ndim == 2 and pv.ndim >= 2:
                n_classes = int(pv.shape[-1])
                converted.append(get_one_hot(tv.astype(np.int64), n_classes))
            else:
                converted.append(tv)
        trueVals = converted[0] if len(converted) == 1 else converted
    return _original_getLossLogStr(trueVals, predVals, steps, sigType, lossFuncs)


_dpad_model_module.getLossLogStr = _patched_getLossLogStr

# -----------------------------------------------------------------------------
# 1) PROJECT SETUP – ensure project root and optional PATIENT_ID for config
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.general.utils_visualization import (
    collect_decoding_timecourse,
    save_decoding_timecourse,
    plot_decoding_timecourses,
)
from src.DPAD.two_stage import train_two_stage_heads, predict_two_stage
from src.DPAD.DPAD_valence import oversample_train


def _load_config(patient_id: int):
    """Load config after setting PATIENT_ID so NEURAL_PATH, EMOTION_PATH, output_dir exist."""
    os.environ["PATIENT_ID"] = str(patient_id)
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", PROJECT_ROOT / "src" / "config.py")
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

# -----------------------------------------------------------------------------
# 2) DATA LOADING – neural (T×D) and emotion (T,) aligned in time
# -----------------------------------------------------------------------------
def load_neural_emotion(neural_path: Path, emotion_path: Path, label_to_idx: dict | None = None):
    """
    Load ECoG (stim) and emotion labels (resp); return arrays aligned in time.
    Labels are 0-9; some patients may have only a subset (e.g. 0,1,2,3,4,5,6,7,9).
    Same remapping as train_eegnet: label_to_idx (original -> model index), idx_to_label
    (model index -> original). Remap to contiguous 0..n-1 so model expects [0, nb_classes-1].
    If label_to_idx is provided, use it for remapping (consistent mapping across train/test).
    Returns (neural, emotion_remapped, label_to_idx, idx_to_label, missing_classes).
    """
    try:
        import mat73
        neural = mat73.loadmat(str(neural_path))["stim"].T  # (T, F)
    except Exception:
        neural = scipy.io.loadmat(str(neural_path))["stim"].T
    emotion = scipy.io.loadmat(str(emotion_path))["resp"].flatten()  # (T,)
    emotion = np.asarray(emotion, dtype=np.int64)
    T = min(neural.shape[0], emotion.shape[0])
    neural = np.asarray(neural[:T], dtype=np.float64)
    emotion = emotion[:T]

    unique_emotions = np.unique(emotion)
    expected_classes = set(range(10))  # 0-9
    missing_classes = expected_classes - set(unique_emotions)
    if label_to_idx is None:
        label_to_idx = {int(orig): idx for idx, orig in enumerate(sorted(unique_emotions))}
    idx_to_label = {idx: int(orig) for orig, idx in label_to_idx.items()}
    # Same remapping: original -> model index (labels not in mapping get -1)
    emotion_remapped = np.array([label_to_idx.get(int(v), -1) for v in emotion], dtype=np.int64)
    return neural, emotion_remapped, label_to_idx, idx_to_label, missing_classes


# -----------------------------------------------------------------------------
# 3) TRAIN / TEST SPLIT – by time (contiguous)
# -----------------------------------------------------------------------------
def split_train_val_test(neural: np.ndarray, emotion: np.ndarray, train_ratio: float = 0.8, val_ratio: float = 0.0):
    """
    Split by time (contiguous). Returns (y_train,z_train), (y_val,z_val), (y_test,z_test).
    Default 80% train / 20% test; val_ratio=0 skips val. DPAD creates its own validation
    from the last 20% of train for early stopping; flexible search with ErSV ignores
    external validation anyway.
    """
    T = neural.shape[0]
    t1 = int(T * train_ratio)
    t2 = int(T * (train_ratio + val_ratio)) if val_ratio > 0 else t1
    y_train, z_train = neural[:t1], emotion[:t1]
    y_val = neural[t1:t2] if val_ratio > 0 else None
    z_val = emotion[t1:t2] if val_ratio > 0 else None
    y_test, z_test = neural[t2:], emotion[t2:]
    return (y_train, z_train), (y_val, z_val), (y_test, z_test)


# -----------------------------------------------------------------------------
# 4) FLEXIBLE NONLINEARITY – search over nonlinearities
# -----------------------------------------------------------------------------

def run_flexible_dpad(y_train: np.ndarray, z_train: np.ndarray, nx: int, n1: int, save_dir: Path, settings: dict | None = None):
    """
    Search over nonlinearity combinations via fitDPADWithFlexibleNonlinearity; returns
    best method code for final fit. Slower but often better than fixed CzNonLin.
    Can be skipped with --skip-flexible.
    """
    settings = settings or {}
    settings.setdefault("min_cores_to_enable_parallelization", 100)  # disable parallel if few cores
    # Base method code: GSUT=iCV, RTR2=decoder, ErSV=internal validation (ignores external val)
    method_code = "DPAD_GSUT_iCVF4_RTR2_uAKCzCy1HL64U_ErSV16"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Show which combinations will be searched
    sub_methods, _ = prepareHyperParameterSearchSpaceFromMethodCode(method_code)
    print(f"Searching {len(sub_methods)} combinations:")
    for i, m in enumerate(sub_methods, 1):
        print(f"  {i:2d}/{len(sub_methods)}: {m}")

    # Enable DPAD flexible progress logs (fold X, method Y)
    logging.getLogger("DPAD.tools.flexible").setLevel(logging.INFO)
    if not logging.getLogger("DPAD.tools.flexible").handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger("DPAD.tools.flexible").addHandler(h)

    # DPAD flexible expects Z as 2D (time x n_z), float dtype (supports NaN placeholders)
    z_train_2d = np.asarray(z_train, dtype=np.float64).reshape(-1, 1) if z_train.ndim == 1 else np.asarray(z_train, dtype=np.float64)
    selected_code, icv_res = fitDPADWithFlexibleNonlinearity(
        y_train, Z=z_train_2d, nx=nx, n1=n1, settings=settings, methodCode=method_code, saveDir=str(save_dir)
    )
    return selected_code


# -----------------------------------------------------------------------------
# 5) TRAIN DPAD – fit model with chosen or user-defined nonlinearity
# -----------------------------------------------------------------------------
def _import_dpad_model():
    """Import DPADModel; PyPI package is 'dpad' (lowercase)."""
    try:
        from dpad import DPADModel
        return DPADModel
    except ModuleNotFoundError:
        try:
            from DPAD import DPADModel
            return DPADModel
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "No module named 'dpad' or 'DPAD'. In env_dpad run: "
                "pip install -r requirements-dpad.txt && pip install dpad --no-deps"
            ) from None


def train_dpad(y_train: np.ndarray, z_train: np.ndarray, nx: int, n1: int, method_code: str, epochs: int = 2500, **fit_kw):
    """
    Build and fit DPADModel. Categorical z uses cross-entropy with integer class indices.
    DPAD fit expects y (D, T) and Z (n_z, T); we transpose from (T, D) and (T,) to match.
    """
    DPADModel = _import_dpad_model()
    id_sys = DPADModel()
    args = DPADModel.prepare_args(method_code)
    # y: (D, T); Z: (n_z, T); reshape z (T,) -> (1, T)
    z_2d = z_train.reshape(1, -1) if z_train.ndim == 1 else z_train
    if z_2d.shape[0] > z_2d.shape[1]:
        z_2d = z_2d.T  # (T, 1) -> (1, T)
    # Prepend one sample per class so DPAD's first-80% val split sees all classes
    # Each prepended sample is the same neural vector as the first training time point
    # Repeated for each class (0, 1, 2, 3, …). It’s arbitrary but avoids empty data and keeps the indices in a valid range for DPAD.
    uniq = np.unique(z_train)
    if len(uniq) > 0:
        n_prepend = len(uniq)
        z_prepend = np.arange(n_prepend, dtype=z_2d.dtype).reshape(1, -1)
        y_prepend = np.repeat(y_train[0:1], n_prepend, axis=0)  # (n_prepend, D)
        z_2d = np.concatenate([z_prepend, z_2d], axis=1)
        y_train = np.concatenate([y_prepend, y_train], axis=0)
    fit_args = {**args, **fit_kw, "epochs": epochs, "nx": nx, "n1": n1}
    id_sys.fit(y_train.T, Z=z_2d, **fit_args)
    return id_sys


# -----------------------------------------------------------------------------
# 6) INFERENCE – decode emotion and get latent embedding
# -----------------------------------------------------------------------------
def predict_dpad(model, y_test: np.ndarray):
    """
    Predict behavior (emotion), neural self-prediction, and latent state.
    DPAD predict expects Y (sample x ny) = (T, D). Returns zPred, yPred, xPred.
    xPred is latent embedding (use-case 3 for downstream CEBRA).
    """
    out = model.predict(y_test)
    z_pred = out[0] if isinstance(out[0], np.ndarray) else np.array(out[0])
    y_pred = out[1] if isinstance(out[1], np.ndarray) else np.array(out[1])
    x_pred = out[2] if isinstance(out[2], np.ndarray) else np.array(out[2])
    return z_pred, y_pred, x_pred


# -----------------------------------------------------------------------------
# 7) EVALUATION – decoding accuracy; optional embedding-based accuracy
# -----------------------------------------------------------------------------
def _z_pred_to_class(z_pred, smooth_window: int | None = None) -> np.ndarray:
    """Convert z_pred (probs or logits) to class indices (T,). Optionally smooth probs temporally first."""
    if hasattr(z_pred, "numpy"):
        z_pred = np.asarray(z_pred)
    z_pred = np.asarray(z_pred)
    if smooth_window is not None and smooth_window > 1:
        from scipy.ndimage import uniform_filter1d
        if z_pred.ndim == 3:
            z_pred = uniform_filter1d(z_pred.astype(np.float64), size=smooth_window, axis=0, mode="nearest")
            z_pred = z_pred / (z_pred.sum(axis=-1, keepdims=True) + 1e-9)
        elif z_pred.ndim == 2:
            z_pred = uniform_filter1d(z_pred.astype(np.float64), size=smooth_window, axis=0, mode="nearest")
            z_pred = z_pred / (z_pred.sum(axis=1, keepdims=True) + 1e-9)
    if z_pred.ndim == 3:
        return np.argmax(z_pred, axis=-1).squeeze()
    if z_pred.ndim == 2:
        return np.argmax(z_pred, axis=1)
    return np.round(z_pred).astype(np.int64)


def evaluate_decoding(z_true: np.ndarray, z_pred, n_classes: int):
    """
    z_true: (T,) integer class indices. z_pred: (T, n_classes) probs or (T,) class indices.
    Returns macro F1 in [0, 1] (average over classes; robust to no-emotion dominance).
    """
    z_pred_class = _z_pred_to_class(z_pred)
    z_true = np.asarray(z_true).flatten()
    T = min(len(z_true), len(z_pred_class))
    return float(f1_score(z_true[:T], z_pred_class[:T], average="macro", zero_division=0))


# -----------------------------------------------------------------------------
# 8) TWO-STAGE GATING – imported from two_stage.py (StandardScaler + decoder grid)
# -----------------------------------------------------------------------------


def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    parser = argparse.ArgumentParser(description="DPAD pipeline: emotion from neural (ECoG)")
    parser.add_argument("--patient-id", type=int, default=None, help="Patient ID (sets PATIENT_ID env and uses config paths)")
    parser.add_argument("--neural-path", type=str, default=None, help="Override: path to neural .mat (stim)")
    parser.add_argument("--emotion-path", type=str, default=None, help="Override: path to emotion .mat (resp)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: output_DPAD/<patient>)")
    parser.add_argument("--nx", type=int, default=16, help="Total latent state dimension")
    parser.add_argument("--n1", type=int, default=16, help="Behavior-relevant latent dimension (use nx=n1 for decoding-only)")
    parser.add_argument("--skip-flexible", action="store_true", help="Skip flexible nonlinearity search; use fixed CzNonLin")
    parser.add_argument("--epochs", type=int, default=5000, help="Max training epochs")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="[Override only] Train fraction when using single file (temporal split)")
    parser.add_argument("--val-ratio", type=float, default=0.0, help="[Override only] Val fraction when using single file")
    parser.add_argument("--two-stage", action="store_true", help="Use two-stage gating: gate (no vs emotion) + emotion head on latent")
    parser.add_argument("--smooth-window", type=int, default=None, metavar="W",
        help="Temporal smoothing window (uniform filter on probs). E.g. 5 or 10. Default: none.")
    parser.add_argument("--oversample", action="store_true", help="Oversample minority classes (adapts to each patient distribution)")
    parser.add_argument("--oversample-strategy", type=str, default="median",
        help="Oversample target: max (balance to majority), median, or float e.g. 0.5 (50%% of max). Default: median.")
    args = parser.parse_args()

    if args.patient_id is None and (args.neural_path is None or args.emotion_path is None):
        parser.error("Provide either --patient-id or both --neural-path and --emotion-path")

    # ----- Paths -----
    config = None
    use_calc_pred = True
    if args.patient_id is not None:
        config = _load_config(args.patient_id)
        neural_train_path = config.NEURAL_PATH
        emotion_train_path = config.EMOTION_PATH
        neural_test_path = config.NEURAL_PRED_PATH
        emotion_test_path = config.EMOTION_PRED_PATH
        out_dir = Path(args.output_dir or str(PROJECT_ROOT / "output_DPAD" / config.output_dir))
    else:
        neural_train_path = Path(args.neural_path)
        emotion_train_path = Path(args.emotion_path)
        neural_test_path = neural_train_path
        emotion_test_path = emotion_train_path
        use_calc_pred = False
        out_dir = Path(args.output_dir or str(PROJECT_ROOT / "output_DPAD" / "custom"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- 1) Load train and test (calc for train, pred for test; same as single_emotion) -----
    if use_calc_pred:
        for p, name in [(neural_train_path, "train neural"), (emotion_train_path, "train emotion"),
                        (neural_test_path, "test neural"), (emotion_test_path, "test emotion")]:
            if not p.exists():
                raise FileNotFoundError(f"{name} not found: {p}")
        print("[1] Loading neural and emotion data (calc=train, pred=test)...")
        print("    Train: calc")
        y_train, z_train, label_to_idx, idx_to_label, missing_classes = load_neural_emotion(
            neural_train_path, emotion_train_path
        )
        nb_classes = len(idx_to_label)
        print(f"    Train neural shape: {y_train.shape}, emotion: {z_train.shape} ({nb_classes} classes: {sorted(idx_to_label.values())})")
        if missing_classes:
            print(f"    Missing emotion classes in train: {sorted(missing_classes)}")
        print("    Test: pred (same label_to_idx as train)")
        y_test_raw, z_test, _, _, _ = load_neural_emotion(
            neural_test_path, emotion_test_path, label_to_idx=label_to_idx
        )
        mask = z_test >= 0
        if mask.sum() < len(z_test):
            print(f"    [WARN] {(z_test < 0).sum()} test samples have labels not in train; excluding from eval")
        y_test = y_test_raw[mask]
        z_test = z_test[mask]
        if len(y_test) == 0:
            raise ValueError("No test samples after filtering (all test labels unseen in train?)")
        print(f"    Test neural shape: {y_test.shape}, emotion: {z_test.shape}")
    else:
        # Override: single file, temporal split
        print("[1] Loading neural and emotion data (single file, temporal split)...")
        neural, emotion, label_to_idx, idx_to_label, missing_classes = load_neural_emotion(
            neural_train_path, emotion_train_path
        )
        nb_classes = len(idx_to_label)
        print(f"    Loaded {neural.shape[0]} timepoints, {nb_classes} classes. Splitting by time (train_ratio={args.train_ratio})...")
        (y_train, z_train), (_, _), (y_test, z_test) = split_train_val_test(
            neural, emotion, train_ratio=args.train_ratio, val_ratio=args.val_ratio
        )
        print(f"    Train {y_train.shape[0]}, Test {y_test.shape[0]}")
    train_dist = {idx_to_label[int(k)]: int(v) for k, v in zip(*np.unique(z_train, return_counts=True))}
    test_dist = {idx_to_label[int(k)]: int(v) for k, v in zip(*np.unique(z_test, return_counts=True))}
    print(f"    Train emotion classes: {sorted(train_dist.items())} (orig_label, count)")
    print(f"    Test emotion classes:  {sorted(test_dist.items())} (orig_label, count)")

    # ----- 2b) Optional oversampling (per patient distribution) -----
    if args.oversample:
        try:
            strat = args.oversample_strategy
            if strat not in ("max", "median"):
                strat = float(strat)
        except (ValueError, TypeError):
            strat = "median"
        n_before = len(z_train)
        y_train, z_train = oversample_train(y_train, z_train, strategy=strat)
        n_after = len(z_train)
        new_dist = {idx_to_label[int(k)]: int(v) for k, v in zip(*np.unique(z_train, return_counts=True))}
        print(f"    Oversampled train: {n_before} -> {n_after} (strategy={args.oversample_strategy})")
        print(f"    Train after oversample: {sorted(new_dist.items())} (orig_label, count)")

    nx, n1 = args.nx, args.n1

    # ----- 3) Optional flexible nonlinearity -----
    flex_dir = out_dir / "flexible_search"
    if args.skip_flexible:
        method_code = "DPAD_RTR2_uAKCzCy1HL64U_ErSV16" # uAKCzCy1HL64U: unified A/K, nonlinear Cz+Cy1 (1×64 MLP), RTR2, ErSV16
        #method_code = "DPAD_CzNonLin" #--> only behavior readout nonlinear (often sufficient per DPAD paper)
        print("[3] Using fixed nonlinearity:", method_code)
    else:
        print("[3] Running flexible nonlinearity search (can be slow)...")
        method_code = run_flexible_dpad(y_train, z_train, nx, n1, out_dir / "flexible_search")
        print("    Selected method code:", method_code)

    # ----- 4) Train DPAD -----
    print("[4] Training DPAD...")
    model = train_dpad(y_train, z_train, nx=nx, n1=n1, method_code=method_code, epochs=args.epochs)

    # ----- 5) Inference -----
    print("[5] Inference on test set...")
    z_pred, y_pred, x_pred = predict_dpad(model, y_test)

    # ----- 5b) Two-stage gating (optional) -----
    gate_clf, emo_clf = None, None
    no_emotion_model_idx = label_to_idx.get(0, None)  # original 0 = "No emotion"
    if args.two_stage and no_emotion_model_idx is not None:
        print("[5b] Two-stage gating: training gate + emotion heads on latent...")
        _, _, x_train_pred = predict_dpad(model, y_train)
        result = train_two_stage_heads(
            x_train_pred, z_train, no_emotion_model_idx,
            idx_to_label, label_to_idx,
        )
        if result[0] is not None:
            gate_clf, emo_clf, scaler, active_indices, best_tau, best_scale, best_dec_type = result
            z_pred_class = predict_two_stage(
                x_pred, gate_clf, emo_clf, scaler,
                no_emotion_model_idx, active_indices,
                best_tau, best_scale, n_expected=len(z_test),
                smooth_window=args.smooth_window,
            )
            smooth_str = f", smooth_window={args.smooth_window}" if args.smooth_window else ""
            print(f"    decoder={best_dec_type}, tau={best_tau:.2f}, emotion_scale={best_scale:.2f}{smooth_str}")
            joblib.dump(gate_clf, out_dir / "gate_clf.joblib")
            if emo_clf is not None:
                joblib.dump(emo_clf, out_dir / "emo_clf.joblib")
            joblib.dump(scaler, out_dir / "scaler.joblib")
            with open(out_dir / "two_stage_meta.json", "w") as f:
                meta = {
                    "decoder_type": best_dec_type,
                    "best_tau": best_tau,
                    "best_emotion_scale": best_scale,
                    "no_emotion_model_idx": no_emotion_model_idx,
                    "active_model_indices": active_indices,
                }
                if args.smooth_window is not None:
                    meta["smooth_window"] = args.smooth_window
                json.dump(meta, f, indent=2)
        else:
            print("    [WARN] Two-stage training failed; using direct DPAD predictions")
            z_pred_class = _z_pred_to_class(z_pred, smooth_window=args.smooth_window)
    else:
        if args.two_stage and no_emotion_model_idx is None:
            print("    [WARN] No 'no emotion' class (0) in data; --two-stage disabled")
        z_pred_class = _z_pred_to_class(z_pred, smooth_window=args.smooth_window)

    # ----- 6) Evaluate -----
    f1_macro = evaluate_decoding(z_test, z_pred_class, n_classes=nb_classes)
    print(f"[6] Test decoding macro F1: {f1_macro:.4f}")

    # Per-class F1 (worst first) for class imbalance analysis
    label_map = config.EMOTION_MAP if config else {}
    T_eval = min(len(z_test), len(z_pred_class))
    f1_per_class = f1_score(
        z_test[:T_eval], z_pred_class[:T_eval],
        average=None, labels=np.arange(nb_classes), zero_division=0
    )
    class_info = []
    for i in range(nb_classes):
        orig = idx_to_label.get(i, i)
        name = label_map.get(int(orig), str(orig)) if label_map else str(orig)
        class_info.append((i, orig, name, float(f1_per_class[i])))
    class_info.sort(key=lambda x: x[3])  # sort by F1 ascending (worst first)
    print("    Per-class F1 (worst → best):")
    for i, orig, name, f1 in class_info:
        print(f"      {orig} ({name}): {f1:.4f}")

    # Convert to original labels for interpretation (idx_to_label: model index -> original 0-9)
    z_test_original = np.array([idx_to_label.get(int(i), i) for i in z_test])
    z_pred_original = np.array([idx_to_label.get(int(i), i) for i in z_pred_class])

    # Save outputs: z_pred (raw), x_pred (latent), z_test/z_pred in original label space
    np.save(out_dir / "z_pred.npy", np.asarray(z_pred))
    np.save(out_dir / "x_pred.npy", np.asarray(x_pred))
    np.save(out_dir / "z_test.npy", z_test_original)
    np.save(out_dir / "z_pred_class.npy", z_pred_original)
    # Same format as train_eegnet: label_mapping.json
    with open(out_dir / "label_mapping.json", "w") as f:
        json.dump({
            "original_to_model": {str(k): v for k, v in label_to_idx.items()},
            "model_to_original": {str(k): v for k, v in idx_to_label.items()},
        }, f, indent=2)
    (out_dir / "method_code.txt").write_text(method_code)
    print(f"    Saved z_pred, x_pred, z_test, z_pred_class, label_mapping.json, method_code.txt to {out_dir}")

    # ----- 7) Timecourse plot (same style as CEBRA) -----
    test_idx = np.arange(len(z_test))
    df_timecourse = collect_decoding_timecourse(
        pair_name="DPAD",
        y_true=z_test_original,
        y_pred=z_pred_original,
        test_idx=test_idx,
    )
    save_decoding_timecourse([df_timecourse], out_dir / "decoding_timecourse.csv")
    emotion_map = None
    if config is not None:
        n_classes_display = 10  # 0-9
        emotion_map = {i: config.EMOTION_MAP.get(i, f"Class {i}") for i in range(n_classes_display)}
    plot_decoding_timecourses(
        csv_path=out_dir / "decoding_timecourse.csv",
        out_path=out_dir / "decoding_timecourse_grid.png",
        emotion_map=emotion_map,
        n_cols=1,
    )
    print(f"    Saved decoding timecourse → {out_dir / 'decoding_timecourse_grid.png'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
