"""
DPAD pipeline for emotion-from-neural decoding (xCEBRA).

Run from project root with DPAD env activated:
  pip install -r requirements-dpad.txt
  python -m src.DPAD.DPAD_main --patient-id 1 [--skip-flexible] [--nx 16] [--n1 16]

  # Two-stage gating (gate: no vs emotion, then emotion head) to mitigate no-emotion dominance:
  python -m src.DPAD.DPAD_main --patient-id 1 --two-stage
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
def load_neural_emotion(neural_path: Path, emotion_path: Path):
    """
    Load ECoG (stim) and emotion labels (resp); return arrays aligned in time.
    Labels are 0-9; some patients may have only a subset (e.g. 0,1,2,3,4,5,6,7,9).
    Same remapping as train_eegnet: label_to_idx (original -> model index), idx_to_label
    (model index -> original). Remap to contiguous 0..n-1 so model expects [0, nb_classes-1].
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

    # Same as train_eegnet: remap to consecutive indices (0..n-1) for missing-class handling
    unique_emotions = np.unique(emotion)
    expected_classes = set(range(10))  # 0-9
    missing_classes = expected_classes - set(unique_emotions)
    label_to_idx = {int(orig): idx for idx, orig in enumerate(sorted(unique_emotions))}
    idx_to_label = {idx: int(orig) for orig, idx in label_to_idx.items()}
    emotion_remapped = np.array([label_to_idx[int(v)] for v in emotion], dtype=np.int64)
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
def _z_pred_to_class(z_pred) -> np.ndarray:
    """Convert z_pred (probs or logits) to class indices (T,)."""
    if hasattr(z_pred, "numpy"):
        z_pred = np.asarray(z_pred)
    z_pred = np.asarray(z_pred)
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
# 8) TWO-STAGE GATING – Gate (no vs emotion) + Emotion (active classes only)
# Mirrors CEBRA full_decoding_finetune / full_encoding_finetune two-head approach.
# -----------------------------------------------------------------------------
def _ensure_samples_last(x: np.ndarray, n_expected: int) -> np.ndarray:
    """Return (T, D) with samples as rows for sklearn. DPAD returns latent (nx, T) or (T, nx)."""
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    # If second dim matches n_expected (time points), first dim is latent -> transpose to (T, nx)
    if x.shape[1] == n_expected and x.shape[0] != n_expected:
        return x.T
    return x


def train_two_stage_heads(
    x_train: np.ndarray,
    z_train: np.ndarray,
    no_emotion_model_idx: int,
    idx_to_label: dict,
    label_to_idx: dict,
) -> tuple:
    """
    Train gate (binary: no vs emotion) and emotion (active classes) classifiers on DPAD latent.
    Returns (gate_clf, emo_clf, active_model_indices, best_tau, best_emotion_scale).
    active_model_indices: model indices for emotion classes (excludes no-emotion).
    """
    T = len(z_train)
    x = _ensure_samples_last(x_train, T)  # (T, D)
    n_active = len(idx_to_label) - 1 if no_emotion_model_idx is not None else len(idx_to_label)
    active_model_indices = [i for i in idx_to_label if i != no_emotion_model_idx]
    if n_active == 0:
        return None, None, [], 0.5, 1.0

    # Gate target: 0 = no emotion, 1 = emotion
    y_gate = (z_train != no_emotion_model_idx).astype(np.int32)

    # Emotion target: map model indices to 0..n_active-1 (only for emotion samples)
    model_to_active = {mi: a for a, mi in enumerate(active_model_indices)}
    y_emo = np.array([model_to_active.get(int(z), -1) for z in z_train], dtype=np.int32)
    mask_emo = y_emo >= 0

    gate_clf = LogisticRegression(solver="lbfgs", class_weight="balanced", C=1.0, max_iter=2000)
    gate_clf.fit(x, y_gate)

    if mask_emo.sum() < 2:
        return gate_clf, None, active_model_indices, 0.5, 1.0

    emo_clf = LogisticRegression(solver="lbfgs", class_weight="balanced", C=1.0, max_iter=2000)
    emo_clf.fit(x[mask_emo], y_emo[mask_emo])

    # Calibration split for tau and emotion_scale grid search
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    try:
        tr_idx, cal_idx = next(sss.split(x, y_gate))
    except ValueError:
        tr_idx, cal_idx = np.arange(len(x)), np.array([], dtype=int)
    if len(cal_idx) < 10:
        return gate_clf, emo_clf, active_model_indices, 0.5, 1.0

    x_cal, z_cal = x[cal_idx], z_train[cal_idx]
    gate_proba_cal = gate_clf.predict_proba(x_cal)[:, 1]
    emo_proba_cal = emo_clf.predict_proba(x_cal) if emo_clf is not None else np.ones((len(x_cal), n_active)) / n_active

    # Build P_full: P(no), P(active_0), ..., P(active_{n_active-1})
    p_no_cal = 1.0 - gate_proba_cal
    p_act_cal = gate_proba_cal[:, np.newaxis] * emo_proba_cal
    P_cal = np.concatenate([p_no_cal[:, np.newaxis], p_act_cal], axis=1)

    tau_grid = [0.1, 0.3, 0.4, 0.45, 0.5, 0.55]
    scale_grid = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    best_tau, best_scale, best_f1 = 0.5, 1.0, -1.0

    for scale in scale_grid:
        pa_s = gate_proba_cal * scale
        p_no_s = np.clip(1.0 - pa_s, 1e-6, 1 - 1e-6)
        p_act_s = np.clip(pa_s[:, np.newaxis] * emo_proba_cal, 1e-6, 1 - 1e-6)
        total = p_no_s[:, np.newaxis] + p_act_s.sum(axis=1, keepdims=True)
        p_no_s = p_no_s / total.squeeze()
        p_act_s = p_act_s / total
        P_s = np.concatenate([p_no_s[:, np.newaxis], p_act_s], axis=1)

        for tau in tau_grid:
            y_pred = np.zeros(len(z_cal), dtype=np.int32)
            pa = 1.0 - P_s[:, 0]
            above = pa >= tau
            y_pred[~above] = no_emotion_model_idx
            if above.any():
                y_pred[above] = np.array([active_model_indices[int(np.argmax(P_s[i, 1:]))] for i in np.where(above)[0]])
            f1 = f1_score(z_cal, y_pred, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_tau, best_scale = f1, tau, scale

    return gate_clf, emo_clf, active_model_indices, best_tau, best_scale


def predict_two_stage(
    x_pred: np.ndarray,
    gate_clf,
    emo_clf,
    no_emotion_model_idx: int,
    active_model_indices: list,
    tau: float,
    emotion_scale: float,
    n_expected: int,
) -> np.ndarray:
    """
    Two-stage prediction: gate prob * emotion softmax. Tau rule: if gate_prob < tau -> no_emotion.
    Returns class indices in model space.
    """
    x = _ensure_samples_last(x_pred, n_expected)
    gate_proba = gate_clf.predict_proba(x)[:, 1]
    if emo_clf is None:
        emo_proba = np.ones((len(x), len(active_model_indices))) / len(active_model_indices)
    else:
        emo_proba = emo_clf.predict_proba(x)

    pa = gate_proba * emotion_scale
    p_no = np.clip(1.0 - pa, 1e-6, 1 - 1e-6)
    p_act = np.clip(pa[:, np.newaxis] * emo_proba, 1e-6, 1 - 1e-6)
    total = p_no[:, np.newaxis] + p_act.sum(axis=1, keepdims=True)
    p_no = p_no / total.squeeze()
    p_act = p_act / total

    y_pred = np.full(len(x), no_emotion_model_idx, dtype=np.int32)
    pa = 1.0 - p_no
    above = pa >= tau
    if above.any():
        y_pred[above] = np.array([active_model_indices[int(np.argmax(p_act[i]))] for i in np.where(above)[0]])
    return y_pred


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
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train fraction (default 80%% train / 20%% test)")
    parser.add_argument("--val-ratio", type=float, default=0.0, help="Val fraction (0 = no separate val; DPAD creates val from train)")
    parser.add_argument("--two-stage", action="store_true", help="Use two-stage gating: gate (no vs emotion) + emotion head on latent")
    args = parser.parse_args()

    if args.patient_id is None and (args.neural_path is None or args.emotion_path is None):
        parser.error("Provide either --patient-id or both --neural-path and --emotion-path")

    # ----- Paths -----
    config = None
    if args.patient_id is not None:
        config = _load_config(args.patient_id)
        neural_path = config.NEURAL_PATH
        emotion_path = config.EMOTION_PATH
        out_dir = Path(args.output_dir or str(PROJECT_ROOT / "output_DPAD" / config.output_dir))
    else:
        neural_path = Path(args.neural_path)
        emotion_path = Path(args.emotion_path)
        out_dir = Path(args.output_dir or str(PROJECT_ROOT / "output_DPAD" / "custom"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- 1) Load data -----
    print("[1] Loading neural and emotion data...")
    neural, emotion, label_to_idx, idx_to_label, missing_classes = load_neural_emotion(neural_path, emotion_path)
    nb_classes = len(idx_to_label)
    print(f"    Neural shape: {neural.shape}, Emotion shape: {emotion.shape} ({nb_classes} classes: {sorted(idx_to_label.values())})")
    if missing_classes:
        print(f"    Missing emotion classes (not in this patient's data): {sorted(missing_classes)}")

    # ----- 2) Split -----
    print("[2] Train/test split...")
    (y_train, z_train), (y_val, z_val), (y_test, z_test) = split_train_val_test(
        neural, emotion, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )
    if y_val is not None:
        print(f"    Train {y_train.shape[0]}, Val {y_val.shape[0]}, Test {y_test.shape[0]}")
    else:
        print(f"    Train {y_train.shape[0]}, Test {y_test.shape[0]} (DPAD creates val from train)")
    # Class distribution: train vs test (orig_label -> count)
    train_unique, train_counts = np.unique(z_train, return_counts=True)
    test_unique, test_counts = np.unique(z_test, return_counts=True)
    train_dist = {idx_to_label[int(k)]: int(v) for k, v in zip(train_unique, train_counts)}
    test_dist = {idx_to_label[int(k)]: int(v) for k, v in zip(test_unique, test_counts)}
    print(f"    Train emotion classes: {sorted(train_dist.items())} (orig_label, count)")
    print(f"    Test emotion classes:  {sorted(test_dist.items())} (orig_label, count)")

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
            gate_clf, emo_clf, active_indices, best_tau, best_scale = result
            z_pred_class = predict_two_stage(
                x_pred, gate_clf, emo_clf,
                no_emotion_model_idx, active_indices,
                best_tau, best_scale, n_expected=len(z_test),
            )
            print(f"    tau={best_tau:.2f}, emotion_scale={best_scale:.2f}")
            joblib.dump(gate_clf, out_dir / "gate_clf.joblib")
            if emo_clf is not None:
                joblib.dump(emo_clf, out_dir / "emo_clf.joblib")
            with open(out_dir / "two_stage_meta.json", "w") as f:
                json.dump({
                    "best_tau": best_tau,
                    "best_emotion_scale": best_scale,
                    "no_emotion_model_idx": no_emotion_model_idx,
                    "active_model_indices": active_indices,
                }, f, indent=2)
        else:
            print("    [WARN] Two-stage training failed; using direct DPAD predictions")
            z_pred_class = _z_pred_to_class(z_pred)
    else:
        if args.two_stage and no_emotion_model_idx is None:
            print("    [WARN] No 'no emotion' class (0) in data; --two-stage disabled")
        z_pred_class = _z_pred_to_class(z_pred)

    # ----- 6) Evaluate -----
    f1_macro = evaluate_decoding(z_test, z_pred_class, n_classes=nb_classes)
    print(f"[6] Test decoding macro F1: {f1_macro:.4f}")

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
    T_total = neural.shape[0]
    t1 = int(T_total * args.train_ratio)
    test_idx = np.arange(t1, T_total)
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
