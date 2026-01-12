"""
CEBRA training using EEGNet-extracted features as input.

This script demonstrates how to use EEGNet features (extracted using extract_eegnet_features.py)
as input to CEBRA instead of raw neural data.

Usage:
    # First, extract EEGNet features:
    python src/extract_eegnet_features.py --patient-id 1 --eegnet-model output_eegnet/238/eegnet_best_model.pt --window-size 32
    
    # Then, train CEBRA on EEGNet features:
    PATIENT_ID=1 python src/full_encoding_eegnet_features.py --eegnet-features-dir output_eegnet_features/238
"""

import os
import argparse
from pathlib import Path
import json

import torch
import numpy as np
import scipy.io
import cebra

from cebra.data import DatasetxCEBRA, ContrastiveMultiObjectiveLoader
from cebra.models import init as init_model

from src.config import (
    MODEL_DIR, EMOTION_PATH, PATIENT_CONFIG
)
from src.utils_training import (
    build_cebra_config_supervised, build_cebra_config_unsupervised, train_and_save
)


def load_eegnet_features(features_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load EEGNet features and configuration.
    
    Args:
        features_dir: Directory containing eegnet_features.npy and eegnet_features_config.json
    
    Returns:
        features: EEGNet features array (T, F_eegnet)
        emotion_labels: Emotion labels (T,)
        config: Configuration dictionary
    """
    features_path = features_dir / "eegnet_features.npy"
    labels_path = features_dir / "emotion_labels.npy"
    config_path = features_dir / "eegnet_features_config.json"
    
    if not features_path.exists():
        raise FileNotFoundError(f"EEGNet features not found: {features_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load features and labels
    features = np.load(features_path)
    if labels_path.exists():
        emotion_labels = np.load(labels_path)
    else:
        # Fallback: load from emotion path (if available)
        emotion_labels = scipy.io.loadmat(str(EMOTION_PATH))['resp'].flatten()
        if len(emotion_labels) != len(features):
            raise ValueError(f"Label length mismatch: {len(emotion_labels)} vs {len(features)}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"[LOAD] Loaded EEGNet features: shape {features.shape}")
    print(f"[LOAD] Feature dimension: {config.get('feature_dim', features.shape[1])}")
    print(f"[LOAD] Number of timepoints: {len(features)}")
    
    return features, emotion_labels, config


def main():
    parser = argparse.ArgumentParser(
        description="Train CEBRA on EEGNet-extracted features."
    )
    parser.add_argument(
        "--eegnet-features-dir",
        type=Path,
        required=True,
        help="Directory containing eegnet_features.npy and eegnet_features_config.json"
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=16,
        help="CEBRA latent dimension (default: 16)"
    )
    parser.add_argument(
        "--unsup-steps",
        type=int,
        default=2000,
        help="Unsupervised training steps (default: 2000)"
    )
    parser.add_argument(
        "--sup-steps",
        type=int,
        default=1500,
        help="Supervised training steps (default: 1500)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size (default: 512)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: MODEL_DIR/xcebra_eegnet_features)"
    )
    
    args = parser.parse_args()
    
    # Setup
    pid = int(float(os.environ.get("PATIENT_ID", "1")))
    _, patient_id = PATIENT_CONFIG[pid]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.output_dir is None:
        args.output_dir = MODEL_DIR / "xcebra_eegnet_features"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[SETUP] Patient: {patient_id}")
    print(f"[SETUP] Device: {device}")
    print(f"[SETUP] Output directory: {args.output_dir}")
    
    # Load EEGNet features
    print(f"\n[DATA] Loading EEGNet features from {args.eegnet_features_dir}")
    eegnet_features, emotion_labels, eegnet_config = load_eegnet_features(args.eegnet_features_dir)
    
    # Convert to tensors
    neural_tensor = torch.tensor(eegnet_features, dtype=torch.float32)  # (T, F_eegnet)
    label_tensor = torch.tensor(emotion_labels, dtype=torch.float32).unsqueeze(1)  # (T, 1)
    
    print(f"[DATA] Neural tensor shape: {neural_tensor.shape}")
    print(f"[DATA] Label tensor shape: {label_tensor.shape}")
    
    # Split train/test
    T = neural_tensor.shape[0]
    split_point = int(0.8 * T)
    train_idx = torch.arange(split_point)
    test_idx = torch.arange(split_point, T)
    
    neural_train = neural_tensor[train_idx]
    neural_test = neural_tensor[test_idx]
    label_train = label_tensor[train_idx]
    label_test = label_tensor[test_idx]
    
    print(f"[SPLIT] Train: {len(neural_train)} timepoints, Test: {len(neural_test)} timepoints")
    
    # Initialize CEBRA model
    # Important: num_neurons must match EEGNet feature dimension
    feature_dim = neural_train.shape[1]
    print(f"\n[MODEL] Initializing CEBRA model with {feature_dim} input features")
    
    encoder_model = init_model(
        name="offset10-model",
        num_neurons=feature_dim,  # EEGNet feature dimension
        num_units=256,
        num_output=args.latent_dim
    ).to(device)
    
    # Create datasets
    train_dataset = DatasetxCEBRA(neural=neural_train, position=label_train)
    test_dataset = DatasetxCEBRA(neural=neural_test, position=label_test)
    train_dataset.configure_for(encoder_model)
    test_dataset.configure_for(encoder_model)
    
    # Create loaders
    unsupervised_loader = ContrastiveMultiObjectiveLoader(
        dataset=train_dataset, batch_size=args.batch_size, num_steps=args.unsup_steps
    )
    supervised_loader = ContrastiveMultiObjectiveLoader(
        dataset=train_dataset, batch_size=args.batch_size, num_steps=args.sup_steps
    )
    
    # Build configs
    BEHAVIOR_INDICES = (0, args.latent_dim)
    unsupervised_config = build_cebra_config_unsupervised(
        unsupervised_loader, BEHAVIOR_INDICES
    )
    supervised_config = build_cebra_config_supervised(
        supervised_loader, BEHAVIOR_INDICES
    )
    
    # Unsupervised pretraining (CEBRA-Time)
    print(f"\n[TRAIN] Phase A: Unsupervised pretraining ({args.unsup_steps} steps)")
    unsup_dir = args.output_dir / "xcebra_unsupervised"
    unsup_dir.mkdir(parents=True, exist_ok=True)
    
    _ = train_and_save(
        model=encoder_model,
        loader=unsupervised_loader,
        config=unsupervised_config,
        out_dir=unsup_dir,
        full_neural_tensor=neural_tensor,
        meta={
            "phase": "unsupervised",
            "latent_dim": args.latent_dim,
            "eegnet_config": eegnet_config,
            "feature_source": "eegnet"
        },
        device=device,
        num_steps=args.unsup_steps
    )
    print(f"[DONE] Unsupervised training complete")
    
    # Supervised fine-tuning (CEBRA-TimeDelta)
    print(f"\n[TRAIN] Phase B: Supervised fine-tuning ({args.sup_steps} steps)")
    sup_dir = args.output_dir / "xcebra_supervised"
    sup_dir.mkdir(parents=True, exist_ok=True)
    
    _ = train_and_save(
        model=encoder_model,
        loader=supervised_loader,
        config=supervised_config,
        out_dir=sup_dir,
        full_neural_tensor=neural_tensor,
        meta={
            "phase": "supervised",
            "latent_dim": args.latent_dim,
            "eegnet_config": eegnet_config,
            "feature_source": "eegnet"
        },
        device=device,
        num_steps=args.sup_steps
    )
    print(f"[DONE] Supervised training complete")
    
    print(f"\n[COMPLETE] CEBRA training on EEGNet features complete!")
    print(f"[OUTPUT] Results saved to: {args.output_dir}")
    print(f"  - Unsupervised: {unsup_dir}")
    print(f"  - Supervised: {sup_dir}")


if __name__ == "__main__":
    main()
