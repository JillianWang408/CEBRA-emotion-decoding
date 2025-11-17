"""
Aggregate neural datasets across multiple patients, z-score each patient,
and generate sanity-check plots to confirm comparable feature statistics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import mat73
import scipy.io


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Copy of the patient configuration declared in `src.config`.
# We keep a local copy to avoid relying on the PATIENT_ID environment variable
# that `src.config` requires at import time.
PATIENT_CONFIG = {
    1: ("EC238", "238"),
    2: ("EC239", "239"),
    9: ("EC272", "272"),
    27: ("EC301", "301"),
    28: ("EC304", "304"),
    15: ("EC280", "280"),
    22: ("EC288", "288"),
    24: ("EC293", "293"),
    29: ("PR06", "PR06"),
    30: ("EC325", "325"),
    31: ("EC326", "326"),
}

# Subdirectory and file names follow the same convention as in `src.config`.
DATA_SUBDIR = (
    "nrcRF_stim_resp_5_Nfold_pairs_msBW_1000_wASpec16_v16_DC5_1   2   5   6   7   8   9  10  11  12"
    "__wASpec16_v16_DC5_1   2   5   6   7   8   9  10  11  12_5"
)
NEURAL_FILENAME = "nrcRF_calc_Stim_StimNum_5_Nr_1_msBW_1000_movHeldOut_1.mat"
EMOTION_FILENAME = "nrcRF_calc_Resp_chan_1_movHeldOut_1.mat"

AGGREGATED_FILENAME = "aggregated_patient_data.npz"
PLOT_FILENAME = "concatenated_heatmap.png"


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PatientPaths:
    patient_id: int
    patient_code: str
    data_dir: Path
    neural_path: Path
    emotion_path: Path


@dataclass
class PatientStats:
    patient_id: int
    patient_label: str
    n_samples: int
    n_features: int
    original_feature_means: np.ndarray
    original_feature_stds: np.ndarray
    z_feature_means: np.ndarray
    z_feature_stds: np.ndarray


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _resolve_patient_paths(patient_id: int) -> PatientPaths:
    if patient_id not in PATIENT_CONFIG:
        known = ", ".join(map(str, sorted(PATIENT_CONFIG)))
        raise KeyError(f"Unknown patient id {patient_id}. Available ids: {known}")

    ec_code, _ = PATIENT_CONFIG[patient_id]
    data_dir = PROJECT_ROOT / "data" / ec_code / DATA_SUBDIR
    neural_path = data_dir / NEURAL_FILENAME
    emotion_path = data_dir / EMOTION_FILENAME

    return PatientPaths(
        patient_id=patient_id,
        patient_code=ec_code,
        data_dir=data_dir,
        neural_path=neural_path,
        emotion_path=emotion_path,
    )


def _load_patient_arrays(patient_paths: PatientPaths) -> Tuple[np.ndarray, np.ndarray]:
    if not patient_paths.neural_path.exists():
        raise FileNotFoundError(
            f"Missing neural data for patient {patient_paths.patient_code}: {patient_paths.neural_path}"
        )

    if not patient_paths.emotion_path.exists():
        raise FileNotFoundError(
            f"Missing emotion data for patient {patient_paths.patient_code}: {patient_paths.emotion_path}"
        )

    neural = mat73.loadmat(str(patient_paths.neural_path))["stim"].T  # Shape (T, F)
    emotion = scipy.io.loadmat(str(patient_paths.emotion_path))["resp"].flatten()

    if neural.shape[0] != emotion.shape[0]:
        raise ValueError(
            f"Sample mismatch for patient {patient_paths.patient_code}: "
            f"neural ({neural.shape[0]}), emotion ({emotion.shape[0]})"
        )

    return neural.astype(np.float32), emotion.astype(np.int32)


def _zscore_features(neural: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_means = neural.mean(axis=0)
    feature_stds = neural.std(axis=0)

    adjusted_stds = np.where(feature_stds < eps, 1.0, feature_stds)
    z_scored = (neural - feature_means) / adjusted_stds

    return z_scored, feature_means, feature_stds


def _collect_stats(
    patient_id: int,
    patient_label: str,
    original_means: np.ndarray,
    original_stds: np.ndarray,
    z_neural: np.ndarray,
) -> PatientStats:
    return PatientStats(
        patient_id=patient_id,
        patient_label=patient_label,
        n_samples=z_neural.shape[0],
        n_features=z_neural.shape[1],
        original_feature_means=original_means,
        original_feature_stds=original_stds,
        z_feature_means=z_neural.mean(axis=0),
        z_feature_stds=z_neural.std(axis=0),
    )

def _plot_concatenated_timeseries(
    z_neural_list: Sequence[np.ndarray],
    patient_labels: Sequence[str],
    save_path: Path | None = None,
    show: bool = True,
) -> None:
    """
    Plot the concatenated z-scored neural signals across patients, preserving
    the original channel x time structure (1000 channels by total time).
    """
    # Each z array is (T_i, F). Concatenate along time to (sum_T, F), then transpose.
    lengths = [z.shape[0] for z in z_neural_list]
    boundaries = np.cumsum([0] + lengths)  # time boundaries in concatenated timeline
    concatenated_tf = np.concatenate(z_neural_list, axis=0)  # (sum_T, F)
    concatenated_ft = concatenated_tf.T  # (F, sum_T) -> channels x time

    plt.figure(figsize=(16, 6))
    im = plt.imshow(
        concatenated_ft,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        origin="upper",
        vmin=-3.0,
        vmax=3.0,
    )
    plt.colorbar(im, fraction=0.02, pad=0.02, label="z-score")
    for i in range(1, len(boundaries) - 1):
        plt.axvline(boundaries[i], color="white", linestyle="--", linewidth=0.8, alpha=0.9)
    # Label patient segments near the top
    for i, label in enumerate(patient_labels):
        start = boundaries[i]
        end = boundaries[i + 1]
        mid = (start + end) // 2
        plt.text(mid, 5, label, color="white", ha="center", va="bottom", fontsize=9, alpha=0.95)

    plt.title("Concatenated z-scored neural signals (channels x time)")
    plt.xlabel("Time (concatenated)")
    plt.ylabel("Channels")
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


def _aggregate_patients(
    patient_ids: Iterable[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[PatientStats]]:
    z_neural_list: List[np.ndarray] = []
    emotion_list: List[np.ndarray] = []
    patient_id_list: List[np.ndarray] = []
    stats: List[PatientStats] = []
    patient_labels: List[str] = []

    for patient_id in patient_ids:
        paths = _resolve_patient_paths(patient_id)
        neural, emotion = _load_patient_arrays(paths)
        
        # Exclude timepoints after 630 for patient 239 (patient_id 2)
        if paths.patient_code == "EC239" or patient_id == 2:
            max_timesteps = 630
            original_shape = neural.shape[0]
            if neural.shape[0] > max_timesteps:
                print(f"[INFO] Trimming patient 239: keeping first {max_timesteps} of {original_shape} timepoints")
                neural = neural[:max_timesteps]
                emotion = emotion[:max_timesteps]
                print(f"[INFO] Patient 239 shape after trimming: {neural.shape[0]} (was {original_shape})")
        
        z_neural, means, stds = _zscore_features(neural)

        stats.append(_collect_stats(patient_id, paths.patient_code, means, stds, z_neural))

        z_neural_list.append(z_neural.astype(np.float32))
        emotion_list.append(emotion.astype(np.int32))
        patient_id_list.append(np.full(z_neural.shape[0], patient_id, dtype=np.int32))
        patient_labels.append(paths.patient_code)

    combined_neural = np.concatenate(z_neural_list, axis=0)
    combined_emotion = np.concatenate(emotion_list, axis=0)
    combined_patient_ids = np.concatenate(patient_id_list, axis=0)

    print(f"[INFO] Final aggregated shape: neural {combined_neural.shape}, emotion {combined_emotion.shape}, patient_ids {combined_patient_ids.shape}")

    return combined_neural, combined_emotion, combined_patient_ids, stats


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def aggregate_patients(
    patient_ids: Sequence[int],
    output_dir: Path | str,
    save_arrays: bool = True,
    make_plot: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    if not patient_ids:
        raise ValueError("`patient_ids` must contain at least one patient identifier.")

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    combined_neural, combined_emotion, combined_patient_ids, stats = _aggregate_patients(patient_ids)

    if save_arrays:
        # Build a suffix using EC codes (e.g., 238_239_272)
        code_suffix = "_".join(PATIENT_CONFIG[pid][1] for pid in patient_ids)
        agg_name = f"aggregated_patient_data_{code_suffix}.npz"
        np.savez(
            output_path / agg_name,
            neural=combined_neural,
            emotion=combined_emotion,
            patient_ids=combined_patient_ids,
        )

    if make_plot:
        # Build per-patient z-neural list again for plotting continuity
        z_neural_segments: List[np.ndarray] = []
        labels: List[str] = []
        for pid in patient_ids:
            paths = _resolve_patient_paths(pid)
            neural, _ = _load_patient_arrays(paths)
            
            # Apply same trimming as in aggregation (exclude after 630 for patient 239)
            if paths.patient_code == "EC239" or pid == 2:
                max_timesteps = 630
                if neural.shape[0] > max_timesteps:
                    neural = neural[:max_timesteps]
            
            z_neural, _, _ = _zscore_features(neural)
            z_neural_segments.append(z_neural)
            labels.append(paths.patient_code)
        code_suffix = "_".join(PATIENT_CONFIG[pid][1] for pid in patient_ids)
        plot_filename = f"concatenated_heatmap_{code_suffix}.png"
        plot_path = output_path / plot_filename if save_arrays else None
        _plot_concatenated_timeseries(z_neural_segments, labels, save_path=plot_path, show=True)

    return combined_neural, combined_emotion, combined_patient_ids


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate neural datasets from multiple patients with per-patient z-scoring."
    )
    parser.add_argument(
        "--patient-ids",
        nargs="+",
        type=int,
        default=sorted(PATIENT_CONFIG.keys()),
        help="Patient IDs to include (default: all configured patients).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output_patient_aggregation",
        help="Where to store arrays and plot unless --no-save is set.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save aggregated arrays or plot.",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    aggregate_patients(
        patient_ids=args.patient_ids,
        output_dir=args.output_dir,
        save_arrays=not args.no_save,
        make_plot=True,
    )


if __name__ == "__main__":
    main()