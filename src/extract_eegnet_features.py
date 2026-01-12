"""
Extract features from EEGNet model for use as input to CEBRA.

This script:
1. Loads a trained EEGNet model
2. Processes neural data using sliding windows
3. Extracts features from EEGNet (before classification layer)
4. Aggregates features per timepoint (mean pooling across overlapping windows)
5. Saves features in format (T, F_eegnet) for CEBRA training
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import mat73
import scipy.io

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eegnet_pytorch import EEGNet
from src.train_eegnet import (
    PATIENT_CONFIG, DATA_SUBDIR, NEURAL_FILENAME, EMOTION_FILENAME,
    load_single_patient_data, load_aggregated_data, create_windows
)


class WindowDataset(Dataset):
    """Dataset for windowed neural data (for feature extraction only, no labels needed)."""
    def __init__(self, X: np.ndarray):
        self.X = torch.FloatTensor(X)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx]


def aggregate_window_features(features: np.ndarray, n_timepoints: int, 
                               window_size: int, stride: int = 1,
                               method: str = 'mean') -> np.ndarray:
    """
    Aggregate features from overlapping windows to per-timepoint features.
    
    Args:
        features: Window features of shape (N_windows, F_eegnet)
        n_timepoints: Original number of timepoints (T)
        window_size: Window size used for extraction
        stride: Stride used for window creation
        method: Aggregation method ('mean', 'center', 'max')
    
    Returns:
        Aggregated features of shape (T, F_eegnet)
    """
    n_windows, n_features = features.shape
    aggregated = np.zeros((n_timepoints, n_features), dtype=features.dtype)
    counts = np.zeros(n_timepoints, dtype=np.int32)
    
    # For each window, assign its features to the timepoints it covers
    for window_idx in range(n_windows):
        start_timepoint = window_idx * stride
        end_timepoint = min(start_timepoint + window_size, n_timepoints)
        
        if method == 'center':
            # Use only the center timepoint
            center_idx = start_timepoint + window_size // 2
            if center_idx < n_timepoints:
                aggregated[center_idx] = features[window_idx]
                counts[center_idx] = 1
        else:
            # Assign to all timepoints in window
            for t in range(start_timepoint, end_timepoint):
                if method == 'mean':
                    aggregated[t] += features[window_idx]
                    counts[t] += 1
                elif method == 'max':
                    aggregated[t] = np.maximum(aggregated[t], features[window_idx])
                    counts[t] = max(counts[t], 1)
    
    # Normalize for mean pooling
    if method == 'mean':
        # Avoid division by zero (timepoints not covered by any window)
        counts = np.where(counts == 0, 1, counts)
        aggregated = aggregated / counts[:, np.newaxis]
    
    return aggregated


def extract_features_from_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    feature_dim: Optional[int] = None
) -> np.ndarray:
    """
    Extract features from EEGNet model for all windows in data loader.
    
    Args:
        model: Trained EEGNet model (in eval mode)
        data_loader: DataLoader with windowed data
        device: PyTorch device
        feature_dim: Expected feature dimension (for validation)
    
    Returns:
        Extracted features of shape (N_windows, F_eegnet)
    """
    model.eval()
    all_features = []
    
    with torch.no_grad():
        for X_batch in data_loader:
            X_batch = X_batch.to(device)
            
            # Extract features (before classification)
            if hasattr(model, 'extract_features'):
                features = model.extract_features(X_batch)
            else:
                # Fallback: manually extract features
                # This mimics the forward pass but stops before dense layer
                x = X_batch
                if x.dim() == 4 and x.shape[-1] == 1:
                    x = x.permute(0, 3, 1, 2)
                elif x.dim() == 3:
                    x = x.unsqueeze(1)
                
                x = model.block1_conv(x)
                x = model.block1_bn1(x)
                x = model.block1_depthwise(x)
                x = model.block1_bn2(x)
                x = torch.nn.functional.elu(x)
                x = model.block1_pool(x)
                x = model.dropout1(x)
                
                x = model.block2_conv(x)
                x = model.block2_bn(x)
                x = torch.nn.functional.elu(x)
                x = model.block2_pool(x)
                x = model.dropout2(x)
                
                features = x.view(x.size(0), -1)
            
            all_features.append(features.cpu().numpy())
    
    features_array = np.concatenate(all_features, axis=0)
    
    if feature_dim is not None and features_array.shape[1] != feature_dim:
        print(f"[WARN] Feature dimension mismatch: expected {feature_dim}, got {features_array.shape[1]}")
    
    return features_array


def load_model_and_config(model_path: Path, device: torch.device) -> Tuple[nn.Module, dict]:
    """
    Load EEGNet model and its configuration.
    
    Args:
        model_path: Path to model checkpoint (.pt file)
        device: PyTorch device
    
    Returns:
        model: Loaded EEGNet model
        config: Model configuration dict
    """
    # Try to load config from same directory
    config_path = model_path.parent / "eegnet_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Try to infer from model path or use defaults
        print(f"[WARN] Config file not found at {config_path}, using defaults")
        config = {
            "nb_classes": 10,
            "Chans": None,  # Will be inferred from data
            "Samples": 32,
            "dropoutRate": 0.5,
            "kernLength": None,
            "F1": 8,
            "D": 2,
            "F2": 16,
        }
    
    # Load model state dict to get actual parameters
    state_dict = torch.load(model_path, map_location=device)
    
    # Infer model parameters from state dict if not in config
    # Try to get Chans from block1_depthwise weight shape
    if config.get("Chans") is None:
        for key in state_dict.keys():
            if "block1_depthwise.weight" in key:
                # Shape: (F1*D, 1, Chans, 1)
                chans = state_dict[key].shape[2]
                config["Chans"] = chans
                break
    
    # Infer F1 and D from state dict
    if config.get("F1") is None or config.get("D") is None:
        for key in state_dict.keys():
            if "block1_depthwise.weight" in key:
                f1_times_d = state_dict[key].shape[0]
                # Try to infer F1 from block1_conv
                for key2 in state_dict.keys():
                    if "block1_conv.weight" in key2:
                        f1 = state_dict[key2].shape[0]
                        d = f1_times_d // f1
                        config["F1"] = f1
                        config["D"] = d
                        break
                break
    
    # Infer F2 from state dict
    if config.get("F2") is None:
        for key in state_dict.keys():
            if "block2_conv.weight" in key:
                config["F2"] = state_dict[key].shape[0]
                break
    
    # Infer nb_classes from dense layer
    if config.get("nb_classes") is None:
        for key in state_dict.keys():
            if "dense.weight" in key:
                config["nb_classes"] = state_dict[key].shape[0]
                break
    
    # Set defaults for missing parameters
    if config.get("kernLength") is None:
        config["kernLength"] = max(8, config.get("Samples", 32) // 4)
    
    print(f"[MODEL] Loaded config: {config}")
    
    # Create model
    model = EEGNet(**config)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model, config


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from trained EEGNet model for CEBRA training."
    )
    
    # Data source
    parser.add_argument(
        "--patient-id",
        type=int,
        default=None,
        help="Patient ID (1, 2, 9, 27, 28, etc.). If specified, loads single patient data."
    )
    parser.add_argument(
        "--aggregated-npz",
        type=Path,
        default=None,
        help="Path to aggregated .npz file. Use this OR --patient-id, not both."
    )
    
    # Model
    parser.add_argument(
        "--eegnet-model",
        type=Path,
        required=True,
        help="Path to trained EEGNet model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        required=True,
        help="Window size used for EEGNet training (must match!)"
    )
    
    # Feature extraction
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride for sliding window (default: 1)"
    )
    parser.add_argument(
        "--aggregation-method",
        type=str,
        default="mean",
        choices=["mean", "center", "max"],
        help="Method to aggregate overlapping window features (default: mean)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for feature extraction"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as model directory or output_eegnet_features/<patient_id>)"
    )
    parser.add_argument(
        "--high-gamma-only",
        action="store_true",
        help="Use only high gamma band (must match EEGNet training)"
    )
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using device: {device}")
    
    # Validate arguments
    if args.patient_id is None and args.aggregated_npz is None:
        parser.error("Must specify either --patient-id OR --aggregated-npz")
    if args.patient_id is not None and args.aggregated_npz is not None:
        parser.error("Cannot specify both --patient-id and --aggregated-npz")
    
    if not args.eegnet_model.exists():
        parser.error(f"Model file not found: {args.eegnet_model}")
    
    # Load model
    print(f"\n[MODEL] Loading EEGNet model from {args.eegnet_model}")
    model, model_config = load_model_and_config(args.eegnet_model, device)
    
    # Determine output directory
    if args.output_dir is None:
        if args.patient_id is not None:
            _, numeric_code = PATIENT_CONFIG[args.patient_id]
            args.output_dir = PROJECT_ROOT / "output_eegnet_features" / numeric_code
        else:
            args.output_dir = args.eegnet_model.parent / "features"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUTPUT] Features will be saved to: {args.output_dir}")
    
    # Load data
    extract_band = 4 if args.high_gamma_only else None
    
    if args.patient_id is not None:
        print(f"\n[DATA] Loading patient {args.patient_id} data...")
        neural, emotion = load_single_patient_data(args.patient_id, extract_band=extract_band)
        patient_code, _ = PATIENT_CONFIG[args.patient_id]
    else:
        print(f"\n[DATA] Loading aggregated data from {args.aggregated_npz}")
        neural, emotion = load_aggregated_data(args.aggregated_npz, extract_band=extract_band)
        patient_code = "aggregated"
    
    print(f"[DATA] Neural shape: {neural.shape}, Emotion shape: {emotion.shape}")
    
    # Verify Chans matches model
    expected_chans = model_config.get("Chans")
    if expected_chans is not None and neural.shape[1] != expected_chans:
        raise ValueError(
            f"Feature dimension mismatch: data has {neural.shape[1]} features, "
            f"but model expects {expected_chans} channels (Chans)"
        )
    
    # Create windows
    print(f"\n[WINDOWING] Creating windows (window_size={args.window_size}, stride={args.stride})...")
    X_windows, _ = create_windows(neural, emotion, args.window_size, args.stride)
    print(f"[WINDOWING] Created {len(X_windows)} windows")
    
    # Create data loader
    window_dataset = WindowDataset(X_windows)
    window_loader = DataLoader(window_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Extract features
    print(f"\n[EXTRACTION] Extracting features from EEGNet model...")
    window_features = extract_features_from_model(model, window_loader, device)
    print(f"[EXTRACTION] Extracted features shape: {window_features.shape}")
    
    # Aggregate features to per-timepoint
    print(f"\n[AGGREGATION] Aggregating window features to timepoints (method={args.aggregation_method})...")
    timepoint_features = aggregate_window_features(
        window_features, 
        n_timepoints=len(neural),
        window_size=args.window_size,
        stride=args.stride,
        method=args.aggregation_method
    )
    print(f"[AGGREGATION] Aggregated features shape: {timepoint_features.shape}")
    
    # Save features
    features_path = args.output_dir / "eegnet_features.npy"
    np.save(features_path, timepoint_features)
    print(f"[SAVE] Features saved to {features_path}")
    
    # Save emotion labels (for convenience)
    emotion_path = args.output_dir / "emotion_labels.npy"
    np.save(emotion_path, emotion)
    print(f"[SAVE] Emotion labels saved to {emotion_path}")
    
    # Save configuration
    config = {
        "patient_code": patient_code,
        "model_path": str(args.eegnet_model),
        "model_config": model_config,
        "window_size": args.window_size,
        "stride": args.stride,
        "aggregation_method": args.aggregation_method,
        "feature_dim": timepoint_features.shape[1],
        "n_timepoints": timepoint_features.shape[0],
        "original_neural_shape": list(neural.shape),
        "high_gamma_only": args.high_gamma_only
    }
    config_path = args.output_dir / "eegnet_features_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"[SAVE] Configuration saved to {config_path}")
    
    print(f"\n[COMPLETE] Feature extraction complete!")
    print(f"[INFO] Features shape: {timepoint_features.shape}")
    print(f"[INFO] Feature dimension: {timepoint_features.shape[1]}")
    print(f"[INFO] Use these features with CEBRA by loading: {features_path}")


if __name__ == "__main__":
    main()
