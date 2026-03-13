"""
Single-emotion trial-level decoding package.

Provides data loading, trial reduction, and DPAD aggregation utilities.
Re-exports core functions from data module for convenience.
"""

from src.single_emotion.data import (
    load_trial_data,
    concatenate_trials_for_dpad,
    aggregate_dpad_predictions_per_trial,
    reduce_trials_to_features,
)

__all__ = [
    "load_trial_data",
    "concatenate_trials_for_dpad",
    "aggregate_dpad_predictions_per_trial",
    "reduce_trials_to_features",
]
