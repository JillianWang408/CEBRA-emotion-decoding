"""
Data loading and transformation for the single-emotion pipeline.

Loads trial-based .mat files (stim/resp), concatenates trials for DPAD,
aggregates per-trial predictions, and reduces trials to feature vectors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io


def load_trial_data(neural_path: Path, emotion_path: Path):
    """
    Load trial-based neural and emotion data from .mat files.

    Expects:
      - stim: MATLAB cell array, each cell = one trial's neural data (T_i x F or F x T_i)
      - resp: emotion label per trial (cell array or flat array, same length as stim)

    Returns:
      - neural_trials: list of ndarray, each (T_i, F)
      - emotions: ndarray (n_trials,) of emotion labels
      - label_to_idx, idx_to_label: for remapping to contiguous 0..n-1
    """
    def _load_mat(path: Path, key: str):
        try:
            import mat73
            data = mat73.loadmat(str(path))[key]
        except Exception:
            data = scipy.io.loadmat(
                str(path), squeeze_me=True, struct_as_record=False
            )[key]
        return data

    stim = _load_mat(neural_path, "stim")
    resp = _load_mat(emotion_path, "resp")

    # Fallback: stim is (T, F) uniform matrix - only when convertible
    try:
        stim_arr = np.asarray(stim)
        if stim_arr.ndim == 2 and stim_arr.dtype != np.object_:
            neural_trials = [stim_arr[i : i + 1] for i in range(stim_arr.shape[0])]
            emotions_raw = np.atleast_1d(np.asarray(resp).flatten())[: len(neural_trials)]
            emotions_raw = emotions_raw.astype(np.int64)
            unique = np.unique(emotions_raw)
            label_to_idx = {int(v): i for i, v in enumerate(sorted(unique))}
            idx_to_label = {i: int(v) for v, i in label_to_idx.items()}
            emotions = np.array(
                [label_to_idx[int(v)] for v in emotions_raw], dtype=np.int64
            )
            return neural_trials, emotions, label_to_idx, idx_to_label
    except (ValueError, TypeError):
        pass  # inhomogeneous cell array, fall through

    # Handle cell array: variable-length trials (44 x (200, T_i))
    if hasattr(stim, "flat"):
        trials = [stim.flat[i] for i in range(stim.size)]
    elif isinstance(stim, (list, tuple)):
        trials = list(stim)
    else:
        trials = [stim]

    # Extract each trial's neural data
    # Data format: each cell is (200, T) = 200 rows (features) × time
    neural_trials = []
    for i, cell in enumerate(trials):
        arr = np.asarray(cell, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim == 2 and arr.shape[0] == 200:
            arr = arr.T  # (200, T) -> (T, 200)
        neural_trials.append(arr)

    # Extract emotion labels
    if hasattr(resp, "shape") and resp.size > 1:
        emotions_raw = np.atleast_1d(resp.flatten())
    elif isinstance(resp, (list, np.ndarray)):
        emotions_raw = np.atleast_1d(np.asarray(resp).flatten())
    else:
        emotions_raw = np.array([resp])

    n_trials = len(neural_trials)
    if len(emotions_raw) != n_trials:
        raise ValueError(
            f"Trial count mismatch: neural has {n_trials} trials, "
            f"emotion has {len(emotions_raw)}"
        )
    emotions_raw = emotions_raw[:n_trials].astype(np.int64)

    # Remap to contiguous 0..n-1
    unique = np.unique(emotions_raw)
    label_to_idx = {int(v): i for i, v in enumerate(sorted(unique))}
    idx_to_label = {i: int(v) for v, i in label_to_idx.items()}
    emotions = np.array([label_to_idx[int(v)] for v in emotions_raw], dtype=np.int64)

    return neural_trials, emotions, label_to_idx, idx_to_label


def concatenate_trials_for_dpad(
    neural_trials: list[np.ndarray],
    emotions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Concatenate trials into (T_total, F) neural and (T_total,) labels for DPAD.
    Each trial's timepoints get the same emotion label.
    """
    neural = np.concatenate(neural_trials, axis=0)
    labels = np.concatenate([np.full(t.shape[0], e) for t, e in zip(neural_trials, emotions)])
    return neural.astype(np.float64), labels.astype(np.int64)


def aggregate_dpad_predictions_per_trial(
    z_pred,
    trial_lengths: list[int],
) -> np.ndarray:
    """
    Aggregate DPAD per-timepoint predictions to one label per trial.
    Uses mean of probabilities when z_pred is (T, n_classes).
    """
    z_pred = np.asarray(z_pred).squeeze()
    if z_pred.ndim < 2 or z_pred.shape[-1] <= 1:
        raise ValueError(
            f"Expected z_pred (T, n_classes); got shape {z_pred.shape}"
        )
    pred_per_trial = []
    offset = 0
    for L in trial_lengths:
        trial_probs = z_pred[offset : offset + L]
        mean_probs = trial_probs.mean(axis=0)
        pred_per_trial.append(int(np.argmax(mean_probs)))
        offset += L
    return np.array(pred_per_trial, dtype=np.int64)


def reduce_trials_to_features(
    neural_trials: list[np.ndarray],
    reduction: str = "mean",
) -> np.ndarray:
    """
    Reduce each trial's (T_i, F) neural data to a single feature vector for sklearn.

    reduction:
      - mean: average over time -> (n_trials, F)
      - max: max over time
      - concat: flatten each trial, pad to max length (variable-length trials)
    """
    if reduction == "mean":
        return np.array([t.mean(axis=0) for t in neural_trials], dtype=np.float64)
    elif reduction == "max":
        return np.array([t.max(axis=0) for t in neural_trials], dtype=np.float64)
    elif reduction == "concat":
        max_t = max(t.shape[0] for t in neural_trials)
        F = neural_trials[0].shape[1]
        X = np.zeros((len(neural_trials), max_t * F), dtype=np.float64)
        for i, t in enumerate(neural_trials):
            flat = t.flatten()
            X[i, : len(flat)] = flat
        return X
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
