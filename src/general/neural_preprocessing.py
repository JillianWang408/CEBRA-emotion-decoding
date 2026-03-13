"""
Neural data preprocessing for ECoG/neural decoding pipelines.

Supports z-score, robust scaling, min-max, and baseline removal.
Fit on train, transform on test to avoid data leakage and handle session shifts.

Usage:
    from src.general.neural_preprocessing import NeuralPreprocessor, preprocess_neural

    # Option 1: Class API
    preproc = NeuralPreprocessor(method="zscore", axis=0)
    y_train = preproc.fit_transform(y_train)
    y_test = preproc.transform(y_test)

    # Option 2: Convenience function
    y_train, y_test, preproc = preprocess_neural(y_train, y_test, method="zscore")

    # DPAD integration: --preprocess zscore
    python -m src.DPAD.DPAD_valence --patient-id 9 --target arousal --preprocess zscore
"""

from __future__ import annotations

from typing import Literal

import numpy as np

__all__ = ["NeuralPreprocessor", "preprocess_neural"]


class NeuralPreprocessor:
    """
    Preprocess neural data (T×D): time points × features (channels, frequencies, etc.).
    Fits statistics on train, applies same transform to test for session generalization.
    """

    def __init__(
        self,
        method: Literal["zscore", "standardize", "robust", "minmax", "baseline", "none"] = "zscore",
        axis: int = 0,
        eps: float = 1e-8,
    ):
        """
        Args:
            method: Preprocessing method.
                - "zscore": (x - mean) / std per feature (axis)
                - "standardize": alias for zscore
                - "robust": (x - median) / IQR, robust to outliers
                - "minmax": (x - min) / (max - min), scale to [0, 1]
                - "baseline": subtract mean per feature (center only)
                - "none": no preprocessing
            axis: Axis along which to compute statistics. 0 = per feature (columns), 1 = per timepoint.
                For (T, D) neural data, axis=0 normalizes each channel/feature across time.
            eps: Small constant to avoid division by zero.
        """
        self.method = method
        self.axis = axis
        self.eps = eps
        self._fitted = False
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._median: np.ndarray | None = None
        self._q25: np.ndarray | None = None
        self._q75: np.ndarray | None = None
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "NeuralPreprocessor":
        """Compute statistics from training data. X: (T, D) or (T, D, ...)."""
        X = np.asarray(X, dtype=np.float64)
        if self.method in ("none",):
            self._fitted = True
            return self

        ax = self.axis if self.axis >= 0 else X.ndim + self.axis
        if self.method in ("zscore", "standardize", "baseline"):
            self._mean = np.nanmean(X, axis=ax, keepdims=True)
            if self.method in ("zscore", "standardize"):
                self._std = np.nanstd(X, axis=ax, keepdims=True)
                self._std = np.where(self._std < self.eps, 1.0, self._std)
        elif self.method == "robust":
            self._median = np.nanmedian(X, axis=ax, keepdims=True)
            self._q25 = np.nanpercentile(X, 25, axis=ax, keepdims=True)
            self._q75 = np.nanpercentile(X, 75, axis=ax, keepdims=True)
            iqr = self._q75 - self._q25
            self._std = np.where(iqr < self.eps, 1.0, iqr)
        elif self.method == "minmax":
            self._min = np.nanmin(X, axis=ax, keepdims=True)
            self._max = np.nanmax(X, axis=ax, keepdims=True)
            self._std = np.where(
                (self._max - self._min) < self.eps, 1.0, self._max - self._min
            )

        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted transform to data."""
        X = np.asarray(X, dtype=np.float64).copy()
        if self.method == "none":
            return X
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() or fit_transform() first.")

        if self.method in ("zscore", "standardize"):
            X = (X - self._mean) / self._std
        elif self.method == "baseline":
            X = X - self._mean
        elif self.method == "robust":
            X = (X - self._median) / self._std
        elif self.method == "minmax":
            X = (X - self._min) / self._std

        return X.astype(np.float64)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

    def get_params(self) -> dict:
        """Return fitted parameters for saving."""
        return {
            "method": self.method,
            "axis": self.axis,
            "eps": self.eps,
            "mean": self._mean,
            "std": self._std,
            "median": self._median,
            "q25": self._q25,
            "q75": self._q75,
            "min": self._min,
            "max": self._max,
        }


def preprocess_neural(
    y_train: np.ndarray,
    y_test: np.ndarray,
    method: str = "zscore",
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray, NeuralPreprocessor]:
    """
    Convenience function: fit preprocessor on train, transform both train and test.

    Args:
        y_train: Training neural data (T_train, D).
        y_test: Test neural data (T_test, D).
        method: One of "zscore", "robust", "minmax", "baseline", "none".
        axis: Axis for statistics (0 = per feature/channel).

    Returns:
        (y_train_processed, y_test_processed, preprocessor)
    """
    preproc = NeuralPreprocessor(method=method, axis=axis)
    y_train_out = preproc.fit_transform(y_train)
    y_test_out = preproc.transform(y_test)
    return y_train_out, y_test_out, preproc
