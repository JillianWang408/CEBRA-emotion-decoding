"""
DPAD pipeline for Valence/Arousal decoding (xCEBRA).

Decodes DC6 (Arousal), DC7 (Valence), or DC9 (Valence/Arousal Categories) from neural data.
EC238 and EC239 do NOT have Valence/Arousal data.

Run from project root with DPAD env activated:
  pip install -r requirements-dpad.txt

  # Decode Arousal (DC6, 0-6 scale)
  python -m src.DPAD.DPAD_valence --patient-id 9 --target arousal [--skip-flexible]

  # Decode Valence (DC7, 0-6 scale)
  python -m src.DPAD.DPAD_valence --patient-id 9 --target valence [--skip-flexible]

  # Decode Valence/Arousal Categories (DC9, 0-4 quadrants)
  python -m src.DPAD.DPAD_valence --patient-id 9 --target categories [--skip-flexible]

  # Run multiple targets in one command
  python -m src.DPAD.DPAD_valence --patient-id 9 --target arousal valence categories [--skip-flexible]

  # Two-stage gating (categories only): gate (no emotion) + emotion head
  python -m src.DPAD.DPAD_valence --patient-id 9 --target arousal valence categories --two-stage

  # Ablation: run baseline, two_stage, oversample, both; save metrics to output_DPAD_valence/aggregate_metrics.csv
  python -m src.DPAD.DPAD_valence --patient-id 9 --target categories --skip-flexible --ablation

  # Preprocessing (recommended): z-score by default; --notch for 60/120 Hz; --log-transform for power; --class-weight for imbalance
  python -m src.DPAD.DPAD_valence --patient-id 9 --target categories --skip-flexible --notch --class-weight

  # Ordinal window is always applied for arousal/valence (window=5)
"""

from __future__ import annotations

import argparse
import csv
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
    plot_confusion_matrix_heatmap,
)
from src.general.neural_preprocessing import NeuralPreprocessor
from src.DPAD.two_stage import train_two_stage_heads, predict_two_stage


def _load_config(patient_id: int):
    """Load config after setting PATIENT_ID so NEURAL_PATH, EMOTION_PATH, output_dir exist."""
    os.environ["PATIENT_ID"] = str(patient_id)
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", PROJECT_ROOT / "src" / "config.py")
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

# -----------------------------------------------------------------------------
# 2) PREPROCESSING – notch filter, z-score, optional log
# -----------------------------------------------------------------------------
def apply_notch_filter(
    neural: np.ndarray,
    fs: float,
    freqs: tuple[float, ...] = (60.0, 120.0),
    Q: float = 30.0,
) -> np.ndarray:
    """
    Apply notch filters to remove mains power line noise (60 Hz and harmonics).
    neural: (T, F) time × features. Each column filtered along time.
    """
    from scipy.signal import iirnotch, filtfilt
    out = np.asarray(neural, dtype=np.float64).copy()
    for f0 in freqs:
        if f0 >= fs / 2:
            continue
        b, a = iirnotch(f0, Q, fs)
        out = filtfilt(b, a, out, axis=0)
    return out


def preprocess_neural_dpad(
    y_train: np.ndarray,
    y_test: np.ndarray,
    notch: bool = False,
    fs: float = 250.0,
    log_transform: bool = False,
    zscore: bool = True,
) -> tuple[np.ndarray, np.ndarray, NeuralPreprocessor | None, object | None]:
    """
    Preprocess neural data for DPAD: optional notch, optional log, z-score.
    Fit scaler on train only; transform train and test.
    Returns (y_train, y_test, scaler, scaler_params).
    """
    scaler = None
    scaler_params = None
    y_tr = np.asarray(y_train, dtype=np.float64).copy()
    y_te = np.asarray(y_test, dtype=np.float64).copy()

    if notch:
        y_tr = apply_notch_filter(y_tr, fs=fs)
        y_te = apply_notch_filter(y_te, fs=fs)

    if log_transform:
        y_tr = np.log1p(y_tr)
        y_te = np.log1p(y_te)

    if zscore:
        scaler = NeuralPreprocessor(method="zscore", axis=0)
        y_tr = scaler.fit_transform(y_tr)
        y_te = scaler.transform(y_te)
        scaler_params = scaler.get_params()

    return y_tr, y_te, scaler, scaler_params


def compute_class_weights(z_train: np.ndarray, n_classes: int) -> np.ndarray:
    """Inverse of class frequency for loss weighting. Rare classes get higher weight."""
    unique, counts = np.unique(z_train, return_counts=True)
    total = len(z_train)
    weights = np.ones(n_classes, dtype=np.float64)
    for c, n in zip(unique, counts):
        if n > 0:
            weights[int(c)] = total / (n_classes * n)
    return weights


def compute_sample_weights(z_train: np.ndarray, class_weights: np.ndarray) -> np.ndarray:
    """Map class labels to per-sample weights for DPAD fit."""
    return np.array([class_weights[int(c)] for c in z_train], dtype=np.float64)


# -----------------------------------------------------------------------------
# 3) DATA LOADING – neural (T×D) and labels (T,) aligned in time
# -----------------------------------------------------------------------------
def load_neural_labels(
    neural_path: Path,
    label_path: Path,
    expected_classes: set | None = None,
    label_to_idx: dict | None = None,
):
    """
    Load ECoG (stim) and labels (resp); return arrays aligned in time.
    Remap to contiguous 0..n-1 so model expects [0, nb_classes-1].
    If label_to_idx is provided, use it for remapping (consistent mapping across train/test).
    Returns (neural, labels_remapped, label_to_idx, idx_to_label, missing_classes).
    """
    try:
        import mat73
        neural = mat73.loadmat(str(neural_path))["stim"].T  # (T, F)
    except Exception:
        neural = scipy.io.loadmat(str(neural_path))["stim"].T
    labels = scipy.io.loadmat(str(label_path))["resp"].flatten()
    labels = np.asarray(labels, dtype=np.int64)
    T = min(neural.shape[0], labels.shape[0])
    neural = np.asarray(neural[:T], dtype=np.float64)
    labels = labels[:T]

    unique = np.unique(labels)
    if expected_classes is None:
        expected_classes = set(range(int(unique.max()) + 1))
    missing_classes = expected_classes - set(unique)
    if label_to_idx is None:
        label_to_idx = {int(orig): idx for idx, orig in enumerate(sorted(unique))}
    idx_to_label = {idx: int(orig) for orig, idx in label_to_idx.items()}
    labels_remapped = np.array([label_to_idx.get(int(v), -1) for v in labels], dtype=np.int64)
    return neural, labels_remapped, label_to_idx, idx_to_label, missing_classes


# -----------------------------------------------------------------------------
# 4) TRAIN / TEST SPLIT – by time (contiguous)
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
# 4b) OVERSAMPLING – balance training by patient/target class distribution
# -----------------------------------------------------------------------------
def oversample_train(
    y_train: np.ndarray,
    z_train: np.ndarray,
    strategy: str | float = "median",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Oversample minority classes based on current class distribution.
    Strategy is per-run: adapts to each patient/target's train distribution.
    strategy: "max" (balance to majority), "median", or float (e.g. 0.5 = 50% of max).
    Returns (y_train_new, z_train_new).
    """
    unique, counts = np.unique(z_train, return_counts=True)
    count_per_class = dict(zip(unique, counts))
    max_count = int(max(counts))

    if strategy == "max":
        target = max_count
    elif strategy == "median":
        target = int(np.median(counts))
    elif isinstance(strategy, (int, float)):
        target = max(1, int(max_count * float(strategy)))
    else:
        raise ValueError(f"Unknown oversample strategy: {strategy}")

    y_parts = [y_train]
    z_parts = [z_train]

    for c in unique:
        n = count_per_class[c]
        if n < target:
            need = target - n
            indices = np.where(z_train == c)[0]
            if len(indices) == 0:
                continue
            oversampled_idx = np.random.choice(indices, size=need, replace=True)
            y_parts.append(y_train[oversampled_idx])
            z_parts.append(z_train[oversampled_idx])

    return np.concatenate(y_parts, axis=0), np.concatenate(z_parts, axis=0)


# -----------------------------------------------------------------------------
# 5) FLEXIBLE NONLINEARITY – search over nonlinearities
# -----------------------------------------------------------------------------

def run_flexible_dpad(y_train: np.ndarray, z_train: np.ndarray, nx: int, n1: int, save_dir: Path, settings: dict | None = None):
    """
    Search over nonlinearity combinations via fitDPADWithFlexibleNonlinearity; returns
    best method code for final fit. Slower but often better than fixed CzNonLin.
    Can be skipped with --skip-flexible.
    """
    settings = settings or {}
    settings.setdefault("min_cores_to_enable_parallelization", 100)  # disable parallel if few cores
    # Base method code: GSUT=iCV, RTR2=decoder.
    # Use ErS16 (not ErSV16) for small datasets: ErSV16 creates val from training (80/20 split),
    # which with block_samples=128 can yield batch_size=0 when val samples < 128. ErS16 uses
    # provided CV validation and avoids that.
    method_code = "DPAD_GSUT_iCVF4_RTR2_uAKCzCy1HL64U_ErS16"
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
# 6) TRAIN DPAD – fit model with chosen or user-defined nonlinearity
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


def train_dpad(
    y_train: np.ndarray,
    z_train: np.ndarray,
    nx: int,
    n1: int,
    method_code: str,
    epochs: int = 2500,
    sample_weight: np.ndarray | None = None,
    **fit_kw,
):
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
        y_prepend = np.repeat(y_train[0:1], n_prepend, axis=0)
        z_2d = np.concatenate([z_prepend, z_2d], axis=1)
        y_train = np.concatenate([y_prepend, y_train], axis=0)
        if sample_weight is not None:
            w_prepend = np.full(n_prepend, float(np.mean(sample_weight)), dtype=np.float64)
            sample_weight = np.concatenate([w_prepend, sample_weight])
    fit_args = {**args, **fit_kw, "epochs": epochs, "nx": nx, "n1": n1}
    # DPAD fit() does not support sample_weight; skip passing it
    id_sys.fit(y_train.T, Z=z_2d, **fit_args)
    return id_sys


# -----------------------------------------------------------------------------
# 7) INFERENCE – decode emotion and get latent embedding
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
# 8) EVALUATION – decoding accuracy; optional embedding-based accuracy
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

# -----------------------------------------------------------------------------
# 9) ORDINAL WINDOW – for arousal/valence: map to -3..3, average in window, map back
# -----------------------------------------------------------------------------
def _ordinal_window_average(
    z_original: np.ndarray,
    window: int,
    label_to_idx: dict,
    neutral: int = 3,
) -> np.ndarray:
    """
    For arousal/valence (0-6 scale, neutral=3): map to scores -3..3, average in time window, map back.
    score = original - neutral. Averaged score -> round(avg + neutral), clip to [0,6].
    Returns original labels (0-6). If rounded value not in label_to_idx, use nearest valid original.
    """
    from scipy.ndimage import uniform_filter1d
    scores = np.asarray(z_original, dtype=np.float64) - neutral
    smoothed = uniform_filter1d(scores, size=window, mode="nearest")
    rounded = np.clip(np.round(smoothed + neutral), 0, 6).astype(np.int32)
    valid_originals = np.array(sorted(label_to_idx.keys()))
    out = np.empty_like(rounded, dtype=np.int32)
    for i, r in enumerate(rounded):
        if r in label_to_idx:
            out[i] = r
        else:
            out[i] = int(valid_originals[np.argmin(np.abs(valid_originals - r))])
    return out


def _original_to_model_indices(z_original: np.ndarray, label_to_idx: dict) -> np.ndarray:
    """Map original labels to model indices. Use nearest valid if not in label_to_idx."""
    valid_originals = np.array(sorted(label_to_idx.keys()))
    out = np.empty(len(z_original), dtype=np.int64)
    for i, o in enumerate(z_original):
        o = int(o)
        if o in label_to_idx:
            out[i] = label_to_idx[o]
        else:
            nearest = valid_originals[np.argmin(np.abs(valid_originals - o))]
            out[i] = label_to_idx[nearest]
    return out


TARGET_CONFIG = {
    "arousal": ("DC6", set(range(7)), "AROUSAL_MAP", 3),      # 0-6 scale, 3=neutral
    "valence": ("DC7", set(range(7)), "VALENCE_MAP", 3),     # 0-6 scale, 3=neutral
    "categories": ("DC9", set(range(5)), "CATEGORY_MAP", 0),  # 0-4 quadrants, 0=neutral
}


def run_single_target(args, target: str, config, config_suffix: str | None = None):
    """
    Run full DPAD pipeline for one target (arousal, valence, or categories).
    If config_suffix is set (e.g. baseline, two_stage), output files get that suffix and metrics are returned.
    """
    # ----- Paths -----
    dc_name, expected_classes, label_map_attr, neutral_orig_label = TARGET_CONFIG[target]
    if args.neural_path and args.label_path:
        # Override: single file path for both train and test (temporal split)
        neural_train_path = Path(args.neural_path)
        label_train_path = Path(args.label_path)
        neural_test_path = neural_train_path
        label_test_path = label_train_path
        out_dir = Path(args.output_dir or str(PROJECT_ROOT / "output_DPAD_valence" / config.output_dir / target))
    else:
        neural_train_path = getattr(config, f"VALENCE_CALC_NEURAL_{dc_name}")
        label_train_path = getattr(config, f"VALENCE_CALC_RESP_{dc_name}")
        neural_test_path = getattr(config, f"VALENCE_PRED_NEURAL_{dc_name}")
        label_test_path = getattr(config, f"VALENCE_PRED_RESP_{dc_name}")
        out_dir = Path(args.output_dir or str(PROJECT_ROOT / "output_DPAD_valence" / config.output_dir / target))
    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = f"_{config_suffix}" if config_suffix else ""

    use_calc_pred = neural_train_path != neural_test_path
    if use_calc_pred:
        for p, name in [(neural_train_path, "train neural"), (label_train_path, "train labels"),
                        (neural_test_path, "test neural"), (label_test_path, "test labels")]:
            if not p.exists():
                raise FileNotFoundError(f"{name} not found: {p}")
    else:
        for p, name in [(neural_train_path, "neural"), (label_train_path, "labels")]:
            if not p.exists():
                raise FileNotFoundError(f"{name} not found: {p}")

    label_map = getattr(config, label_map_attr, {})

    # ----- 1) Load train and test -----
    print(f"[1] Loading {target} data ({dc_name})...")
    if use_calc_pred:
        print(f"    Train: calc (neural + labels)")
        y_train, z_train, label_to_idx, idx_to_label, missing_train = load_neural_labels(
            neural_train_path, label_train_path, expected_classes=expected_classes
        )
        nb_classes = len(idx_to_label)
        print(f"    Train neural shape: {y_train.shape}, labels: {z_train.shape} ({nb_classes} classes: {sorted(idx_to_label.values())})")
        if missing_train:
            print(f"    Missing classes in train: {sorted(missing_train)}")

        print(f"    Test: pred (same label_to_idx as train)")
        y_test_raw, z_test, _, _, _ = load_neural_labels(
            neural_test_path, label_test_path, expected_classes=expected_classes, label_to_idx=label_to_idx
        )
        mask = z_test >= 0
        if mask.sum() < len(z_test):
            print(f"    [WARN] {(z_test < 0).sum()} test samples have labels not in train; excluding from eval")
        y_test = y_test_raw[mask]
        z_test = z_test[mask]
        if len(y_test) == 0:
            raise ValueError("No test samples after filtering (all test labels unseen in train?)")
        print(f"    Test neural shape: {y_test.shape}, labels: {z_test.shape}")
    else:
        # Override: single file, temporal split
        neural, labels, label_to_idx, idx_to_label, missing_train = load_neural_labels(
            neural_train_path, label_train_path, expected_classes=expected_classes
        )
        nb_classes = len(idx_to_label)
        print(f"    Loaded {neural.shape[0]} timepoints, {nb_classes} classes. Splitting by time (train_ratio={args.train_ratio})...")
        (y_train, z_train), (_, _), (y_test, z_test) = split_train_val_test(
            neural, labels, train_ratio=args.train_ratio, val_ratio=args.val_ratio
        )
        print(f"    Train {y_train.shape[0]}, Test {y_test.shape[0]}")
    train_dist = {idx_to_label[int(k)]: int(v) for k, v in zip(*np.unique(z_train, return_counts=True))}
    test_dist = {idx_to_label[int(k)]: int(v) for k, v in zip(*np.unique(z_test, return_counts=True))}
    print(f"    Train {target} classes: {sorted(train_dist.items())} (orig_label, count)")
    print(f"    Test {target} classes:  {sorted(test_dist.items())} (orig_label, count)")

    # ----- 2) Preprocessing: notch filter, z-score, optional log
    neural_scaler = None
    if args.notch or args.zscore or args.log_transform:
        y_train, y_test, neural_scaler, _ = preprocess_neural_dpad(
            y_train, y_test,
            notch=args.notch,
            fs=args.fs,
            log_transform=args.log_transform,
            zscore=args.zscore,
        )
        steps = []
        if args.notch:
            steps.append("60/120 Hz notch")
        if args.log_transform:
            steps.append("log(1+x)")
        if args.zscore:
            steps.append("z-score")
        print(f"    Preprocessed: {', '.join(steps)}")
        if neural_scaler is not None:
            joblib.dump(neural_scaler, out_dir / f"neural_scaler{sfx}.joblib")

    # ----- 2b) Optional oversampling (per patient/target distribution) -----
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
        method_code = "DPAD_RTR2_Cz1HL64U_ErS16" #272/304 flexibility search result #Nonlinear (only Cz1)
        print("[3] Using fixed nonlinearity:", method_code)
    else:
        print("[3] Running flexible nonlinearity search (can be slow)...")
        method_code = run_flexible_dpad(y_train, z_train, nx, n1, out_dir / "flexible_search")
        print("    Selected method code:", method_code)

    # ----- 4) Train DPAD -----
    sample_weight = None
    if args.class_weight:
        cw = compute_class_weights(z_train, nb_classes)
        sample_weight = compute_sample_weights(z_train, cw)
        print(f"    Class weights (inverse freq): {dict(zip(range(nb_classes), np.round(cw, 3)))}")
        print(f"    [NOTE] DPAD fit() does not support sample_weight; class weights have no effect on DPAD training")
    print("[4] Training DPAD...")
    model = train_dpad(
        y_train, z_train, nx=nx, n1=n1, method_code=method_code, epochs=args.epochs,
        sample_weight=sample_weight,
    )

    # ----- 5) Inference -----
    print("[5] Inference on test set...")
    z_pred, y_pred, x_pred = predict_dpad(model, y_test)

    # ----- 5b) Two-stage gating (categories only) -----
    gate_clf, emo_clf = None, None
    no_emotion_model_idx = label_to_idx.get(neutral_orig_label, None)  # neutral class for two-stage
    if args.two_stage and target == "categories" and no_emotion_model_idx is not None:
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
            joblib.dump(gate_clf, out_dir / f"gate_clf{sfx}.joblib")
            if emo_clf is not None:
                joblib.dump(emo_clf, out_dir / f"emo_clf{sfx}.joblib")
            joblib.dump(scaler, out_dir / f"scaler{sfx}.joblib")
            with open(out_dir / f"two_stage_meta{sfx}.json", "w") as f:
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
        if args.two_stage and target == "categories" and no_emotion_model_idx is None:
            print("    [WARN] No 'no emotion' class (0) in data; --two-stage disabled for categories")
        z_pred_class = _z_pred_to_class(z_pred, smooth_window=args.smooth_window)

    # Convert to original labels for interpretation (idx_to_label: model index -> original)
    z_test_original = np.array([idx_to_label.get(int(i), i) for i in z_test])
    z_pred_original = np.array([idx_to_label.get(int(i), i) for i in z_pred_class])

    # Ordinal window averaging (always for arousal/valence): map 0-6 to -3..3, average in window, map back
    ORDINAL_WINDOW = 5
    if target in ("arousal", "valence"):
        z_pred_original = _ordinal_window_average(
            z_pred_original, ORDINAL_WINDOW, label_to_idx, neutral=neutral_orig_label
        )
        z_pred_class = _original_to_model_indices(z_pred_original, label_to_idx)
        print(f"    ordinal_window={ORDINAL_WINDOW} (scores -3..3 averaged)")

    # ----- 6) Evaluate -----
    f1_macro = evaluate_decoding(z_test, z_pred_class, n_classes=nb_classes)
    print(f"[6] Test decoding macro F1: {f1_macro:.4f}")

    # Per-class F1 (worst first) for class imbalance analysis
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

    # Save outputs: z_pred (raw), x_pred (latent), z_test/z_pred in original label space
    np.save(out_dir / f"z_pred{sfx}.npy", np.asarray(z_pred))
    np.save(out_dir / f"x_pred{sfx}.npy", np.asarray(x_pred))
    np.save(out_dir / f"z_test{sfx}.npy", z_test_original)
    np.save(out_dir / f"z_pred_class{sfx}.npy", z_pred_original)
    # Same format as train_eegnet: label_mapping.json (shared across configs)
    with open(out_dir / "label_mapping.json", "w") as f:
        json.dump({
            "original_to_model": {str(k): v for k, v in label_to_idx.items()},
            "model_to_original": {str(k): v for k, v in idx_to_label.items()},
        }, f, indent=2)
    (out_dir / f"method_code{sfx}.txt").write_text(method_code)
    print(f"    Saved z_pred, x_pred, z_test, z_pred_class, label_mapping.json, method_code.txt to {out_dir}")

    # ----- 7) Timecourse plot (same style as CEBRA) -----
    test_idx = np.arange(len(z_test))
    df_timecourse = collect_decoding_timecourse(
        pair_name=f"DPAD_{target}",
        y_true=z_test_original,
        y_pred=z_pred_original,
        test_idx=test_idx,
    )
    save_decoding_timecourse([df_timecourse], out_dir / f"decoding_timecourse{sfx}.csv")
    n_classes_display = max(label_map.keys()) + 1 if label_map else 10
    display_map = {i: label_map.get(i, f"Class {i}") for i in range(n_classes_display)}
    plot_decoding_timecourses(
        csv_path=out_dir / f"decoding_timecourse{sfx}.csv",
        out_path=out_dir / f"decoding_timecourse_grid{sfx}.png",
        emotion_map=display_map,
        n_cols=1,
    )
    print(f"    Saved decoding timecourse → {out_dir / f'decoding_timecourse_grid{sfx}.png'}")

    # Confusion matrix heatmap (arousal, valence, categories)
    plot_confusion_matrix_heatmap(
        y_true=z_test_original,
        y_pred=z_pred_original,
        label_map=label_map,
        out_path=out_dir / f"confusion_matrix{sfx}.png",
        title=f"DPAD {target.capitalize()} Decoding – Confusion Matrix" + (f" ({config_suffix})" if config_suffix else ""),
    )

    if config_suffix:
        return {
            "macro_f1": f1_macro,
            "f1_per_class": [float(f1_per_class[i]) for i in range(nb_classes)],
            "n_classes": nb_classes,
        }
    return 0


def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    parser = argparse.ArgumentParser(description="DPAD pipeline: Valence/Arousal decoding")
    parser.add_argument("--patient-id", type=int, required=True, help="Patient ID (EC238/239 excluded - no valence data)")
    parser.add_argument("--target", type=str, choices=["arousal", "valence", "categories"], nargs="+", required=True,
        help="Decode: arousal (DC6), valence (DC7), categories (DC9). Pass multiple to run all, e.g. --target arousal valence categories")
    parser.add_argument("--neural-path", type=str, default=None, help="Override: single neural file (temporal split; use default for calc/pred). Ignored when multiple targets.")
    parser.add_argument("--label-path", type=str, default=None, help="Override: single label file (temporal split). Ignored when multiple targets.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory base (default: output_DPAD_valence/<patient>/<target>)")
    parser.add_argument("--nx", type=int, default=16, help="Total latent state dimension")
    parser.add_argument("--n1", type=int, default=16, help="Behavior-relevant latent dimension")
    parser.add_argument("--skip-flexible", action="store_true", help="Skip flexible nonlinearity search")
    parser.add_argument("--epochs", type=int, default=5000, help="Max training epochs")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train fraction (default 80%% train / 20%% test)")
    parser.add_argument("--val-ratio", type=float, default=0.0, help="Val fraction (0 = no separate val)")
    parser.add_argument("--two-stage", action="store_true", help="Use two-stage gating for categories only (gate + emotion head)")
    parser.add_argument("--smooth-window", type=int, default=None, metavar="W",
        help="Temporal smoothing window (uniform filter on probs). E.g. 5 or 10. Default: none.")
    parser.add_argument("--oversample", action="store_true", help="Oversample minority classes (adapts to each patient/target distribution)")
    parser.add_argument("--oversample-strategy", type=str, default="median",
        help="Oversample target: max (balance to majority), median, or float e.g. 0.5 (50%% of max). Default: median.")
    parser.add_argument("--ablation", action="store_true",
        help="Run 4 configs: baseline, two_stage, oversample, both. Save metrics to aggregate CSV and rename outputs per config.")
    parser.add_argument("--notch", action="store_true", help="Apply 60/120 Hz notch filter to remove mains noise")
    parser.add_argument("--fs", type=float, default=250.0, help="Sampling rate (Hz) for notch filter. Default: 250")
    parser.add_argument("--no-zscore", action="store_true", help="Disable z-score (default: z-score is applied)")
    parser.add_argument("--log-transform", action="store_true", help="Apply log(1+x) before z-score (for skewed power features)")
    parser.add_argument("--class-weight", action="store_true",
        help="Use class weights in loss (inverse frequency). Prefer over oversampling for imbalance.")
    args = parser.parse_args()
    args.zscore = not args.no_zscore

    targets = args.target  # nargs="+" always gives a list

    # Override paths only supported for single target
    if len(targets) > 1 and (args.neural_path or args.label_path):
        parser.error("--neural-path and --label-path are not supported with multiple targets. Use default calc/pred paths.")

    # Config & patient check
    config = _load_config(args.patient_id)
    if args.patient_id not in config.PATIENTS_WITH_VALENCE_AROUSAL:
        parser.error(
            f"Patient {args.patient_id} (EC238/EC239) does NOT have Valence/Arousal data. "
            f"Use patients: {config.PATIENTS_WITH_VALENCE_AROUSAL}"
        )

    if args.ablation:
        # Run 4 configs: baseline, two_stage, oversample, both
        ablation_configs = [
            ("baseline", False, False),
            ("two_stage", True, False),
            ("oversample", False, True),
            ("both", True, True),
        ]
        agg_dir = PROJECT_ROOT / "output_DPAD_valence"
        agg_dir.mkdir(parents=True, exist_ok=True)
        csv_path = agg_dir / "aggregate_metrics.csv"
        csv_exists = csv_path.exists()

        for i, target in enumerate(targets):
            for config_name, use_two_stage, use_oversample in ablation_configs:
                print(f"\n{'='*60}\n[Target {i+1}/{len(targets)}] {target.upper()} | config={config_name}\n{'='*60}")
                ab_args = argparse.Namespace(**{k: getattr(args, k) for k in vars(args)})
                ab_args.two_stage = use_two_stage
                ab_args.oversample = use_oversample
                result = run_single_target(ab_args, target, config, config_suffix=config_name)
                if isinstance(result, dict):
                    n = result["n_classes"]
                    f1_list = result["f1_per_class"]
                    row = {
                        "patient_id": args.patient_id,
                        "target": target,
                        "config": config_name,
                        "macro_f1": f"{result['macro_f1']:.6f}",
                        **{f"f1_class_{j}": (f"{f1_list[j]:.6f}" if j < n else "") for j in range(7)},
                    }
                    fieldnames = ["patient_id", "target", "config", "macro_f1"] + [f"f1_class_{j}" for j in range(7)]
                    with open(csv_path, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                        if not csv_exists:
                            w.writeheader()
                            csv_exists = True
                        w.writerow(row)
        print(f"\n[Ablation] Saved aggregate metrics → {csv_path}")
    else:
        for i, target in enumerate(targets):
            print(f"\n{'='*60}\n[Target {i+1}/{len(targets)}] {target.upper()}\n{'='*60}")
            run_single_target(args, target, config)

    return 0


if __name__ == "__main__":
    sys.exit(main())
