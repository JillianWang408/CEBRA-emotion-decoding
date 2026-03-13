"""
Shared two-stage gating (gate + emotion heads) for DPAD decoding.
Uses StandardScaler on latent and grid search over decoder types for best F1.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def _ensure_samples_last(x: np.ndarray, n_expected: int) -> np.ndarray:
    """Return (T, D) with samples as rows for sklearn. DPAD returns latent (nx, T) or (T, nx)."""
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.shape[1] == n_expected and x.shape[0] != n_expected:
        return x.T
    return x


def _make_decoder(decoder_type: str):
    """Create decoder for gate (binary) or emotion (multi-class)."""
    common = {"class_weight": "balanced", "random_state": 42}
    if decoder_type == "logreg":
        return LogisticRegression(solver="lbfgs", C=1.0, max_iter=2000, **common)
    if decoder_type == "logreg_C01":
        return LogisticRegression(solver="lbfgs", C=0.1, max_iter=2000, **common)
    if decoder_type == "logreg_C10":
        return LogisticRegression(solver="lbfgs", C=10.0, max_iter=2000, **common)
    if decoder_type == "svc":
        return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, **common)
    if decoder_type == "svc_C10":
        return SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, **common)
    if decoder_type == "rf":
        return RandomForestClassifier(n_estimators=100, max_depth=10, **common)
    raise ValueError(f"Unknown decoder_type: {decoder_type}")


def train_two_stage_heads(
    x_train: np.ndarray,
    z_train: np.ndarray,
    no_emotion_model_idx: int,
    idx_to_label: dict,
    label_to_idx: dict,
) -> tuple:
    """
    Train gate (binary: no vs emotion) and emotion (active classes) classifiers on DPAD latent.
    Uses StandardScaler on latent and grid search over decoder types (LogReg, SVC, RF) for best F1.
    Returns (gate_clf, emo_clf, scaler, active_model_indices, best_tau, best_emotion_scale).
    """
    T = len(z_train)
    x = _ensure_samples_last(x_train, T)  # (T, D)

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    n_active = len(idx_to_label) - 1 if no_emotion_model_idx is not None else len(idx_to_label)
    active_model_indices = [i for i in idx_to_label if i != no_emotion_model_idx]
    if n_active == 0:
        return None, None, scaler, [], 0.5, 1.0, "logreg"

    y_gate = (z_train != no_emotion_model_idx).astype(np.int32)
    model_to_active = {mi: a for a, mi in enumerate(active_model_indices)}
    y_emo = np.array([model_to_active.get(int(z), -1) for z in z_train], dtype=np.int32)
    mask_emo = y_emo >= 0

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    try:
        tr_idx, cal_idx = next(sss.split(x, y_gate))
    except ValueError:
        tr_idx, cal_idx = np.arange(len(x)), np.array([], dtype=int)
    if len(cal_idx) < 10:
        gate_clf = LogisticRegression(solver="lbfgs", class_weight="balanced", C=1.0, max_iter=2000)
        gate_clf.fit(x, y_gate)
        emo_clf = None
        if mask_emo.sum() >= 2:
            emo_clf = LogisticRegression(solver="lbfgs", class_weight="balanced", C=1.0, max_iter=2000)
            emo_clf.fit(x[mask_emo], y_emo[mask_emo])
        return gate_clf, emo_clf, scaler, active_model_indices, 0.5, 1.0, "logreg"

    x_tr, y_gate_tr = x[tr_idx], y_gate[tr_idx]
    x_cal, z_cal = x[cal_idx], z_train[cal_idx]
    y_emo_tr = y_emo[tr_idx]
    mask_emo_tr = mask_emo[tr_idx]

    decoder_types = ["logreg_C01", "logreg", "logreg_C10", "svc", "svc_C10", "rf"]
    tau_grid = [0.1, 0.3, 0.4, 0.45, 0.5, 0.55]
    scale_grid = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    best_f1, best_tau, best_scale = -1.0, 0.5, 1.0
    best_gate_clf, best_emo_clf = None, None
    best_dec_type = "logreg"

    for dec_type in decoder_types:
        try:
            gate_clf = _make_decoder(dec_type)
            gate_clf.fit(x_tr, y_gate_tr)
            emo_clf = None
            if mask_emo_tr.sum() >= 2:
                emo_clf = _make_decoder(dec_type)
                emo_clf.fit(x_tr[mask_emo_tr], y_emo_tr[mask_emo_tr])
            gate_proba_cal = gate_clf.predict_proba(x_cal)[:, 1]
            emo_proba_cal = emo_clf.predict_proba(x_cal) if emo_clf is not None else np.ones((len(x_cal), n_active)) / n_active

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
                        best_gate_clf, best_emo_clf = gate_clf, emo_clf
                        best_dec_type = dec_type
        except Exception:
            continue

    if best_gate_clf is None:
        best_gate_clf = LogisticRegression(solver="lbfgs", class_weight="balanced", C=1.0, max_iter=2000)
        best_gate_clf.fit(x_tr, y_gate_tr)
        best_emo_clf = None
        if mask_emo_tr.sum() >= 2:
            best_emo_clf = LogisticRegression(solver="lbfgs", class_weight="balanced", C=1.0, max_iter=2000)
            best_emo_clf.fit(x_tr[mask_emo_tr], y_emo_tr[mask_emo_tr])

    return best_gate_clf, best_emo_clf, scaler, active_model_indices, best_tau, best_scale, best_dec_type


def predict_two_stage(
    x_pred: np.ndarray,
    gate_clf,
    emo_clf,
    scaler,
    no_emotion_model_idx: int,
    active_model_indices: list,
    tau: float,
    emotion_scale: float,
    n_expected: int,
    smooth_window: int | None = None,
) -> np.ndarray:
    """
    Two-stage prediction: gate prob * emotion softmax. Tau rule: if gate_prob < tau -> no_emotion.
    If smooth_window is set, applies temporal smoothing (uniform filter) to probabilities before tau/argmax.
    Returns class indices in model space.
    """
    x = _ensure_samples_last(x_pred, n_expected)
    x = scaler.transform(x)
    gate_proba = gate_clf.predict_proba(x)[:, 1]
    if emo_clf is None:
        emo_proba = np.ones((len(x), len(active_model_indices))) / len(active_model_indices)
    else:
        emo_proba = emo_clf.predict_proba(x)

    if smooth_window is not None and smooth_window > 1:
        gate_proba = uniform_filter1d(gate_proba.astype(np.float64), size=smooth_window, mode="nearest")
        emo_proba = uniform_filter1d(emo_proba.astype(np.float64), size=smooth_window, axis=0, mode="nearest")
        emo_proba = emo_proba / emo_proba.sum(axis=1, keepdims=True)

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
