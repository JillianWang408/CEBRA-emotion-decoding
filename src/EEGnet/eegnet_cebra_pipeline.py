"""
Unified pipeline: EEGNet feature extraction + CEBRA training.

This script combines:
1. EEGNet training (or loading existing model)
2. EEGNet feature extraction
3. CEBRA training (unsupervised + supervised) on EEGNet features

Output structure matches output_eegnet but with additional CEBRA outputs.
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
from torch.utils.data import Dataset, DataLoader
import cebra
from cebra.data import DatasetxCEBRA, ContrastiveMultiObjectiveLoader
from cebra.models import init as init_model

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eegnet_pytorch import EEGNet
from src.train_eegnet import (
    PATIENT_CONFIG, DATA_SUBDIR, NEURAL_FILENAME, EMOTION_FILENAME,
    EMOTION_MAP,
    load_single_patient_data, create_windows, train_eegnet,
    analyze_class_distribution, compute_class_weights
)
from src.utils_training import (
    build_cebra_config_supervised, build_cebra_config_unsupervised, train_and_save
)
from src.utils_training import plot_embedding_split


def align_embedding_labels(Z, y_full):
    """Align embedding with labels, handling temporal offset.
    
    With valid padding (padding=0), PyTorch Conv1d output aligns with START of input.
    So we trim labels from the END (use first T_emb labels).
    """
    T_emb = Z.shape[0]
    T_full = len(y_full)
    offset = T_full - T_emb
    assert offset >= 0, f"Embedding longer than labels: {T_emb} > {T_full}"
    # Trim from END: use first T_emb labels (last 'offset' labels are lost)
    y_aligned = y_full[:T_emb]

    # 80/20 split on the embedding timeline
    split = int(0.8 * T_emb)
    print(f"split at {split} / {T_emb} (offset={offset}, trimmed from END)")
    return y_aligned, offset, split
from src.utils_visualization import collect_decoding_timecourse, save_decoding_timecourse, plot_decoding_timecourses
# Define _split_train_test locally to avoid config.py dependency
def _split_train_test(embedding, y_full):
    """Align embedding with labels, then return aligned train/test splits."""
    y_aligned, offset, split = align_embedding_labels(embedding, y_full)
    X = embedding[:len(y_aligned)]
    y = np.squeeze(y_aligned).astype(int)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, X_test, y_train, y_test, offset, split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class WindowDatasetFeatures(Dataset):
    """Dataset for windowed neural data (for feature extraction only)."""
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
    
    for window_idx in range(n_windows):
        start_timepoint = window_idx * stride
        end_timepoint = min(start_timepoint + window_size, n_timepoints)
        
        if method == 'center':
            center_idx = start_timepoint + window_size // 2
            if center_idx < n_timepoints:
                aggregated[center_idx] = features[window_idx]
                counts[center_idx] = 1
        else:
            for t in range(start_timepoint, end_timepoint):
                if method == 'mean':
                    aggregated[t] += features[window_idx]
                    counts[t] += 1
                elif method == 'max':
                    aggregated[t] = np.maximum(aggregated[t], features[window_idx])
                    counts[t] = max(counts[t], 1)
    
    if method == 'mean':
        counts = np.where(counts == 0, 1, counts)
        aggregated = aggregated / counts[:, np.newaxis]
    
    return aggregated


def extract_eegnet_features(
    model: nn.Module,
    neural_data: np.ndarray,
    emotion_labels: np.ndarray,
    window_size: int,
    stride: int,
    batch_size: int,
    device: torch.device,
    aggregation_method: str = 'mean'
) -> Tuple[np.ndarray, dict]:
    """
    Extract EEGNet features from neural data.
    
    Returns:
        features: Extracted features (T, F_eegnet)
        feature_info: Dictionary with feature dimension info
    """
    print(f"\n[EEGNET FEATURES] Extracting features...")
    
    # Create windows
    X_windows, _ = create_windows(neural_data, emotion_labels, window_size, stride)
    print(f"  Created {len(X_windows)} windows")
    
    # Create data loader
    window_dataset = WindowDatasetFeatures(X_windows)
    window_loader = DataLoader(window_dataset, batch_size=batch_size, shuffle=False)
    
    # Extract features
    model.eval()
    all_features = []
    
    with torch.no_grad():
        for X_batch in window_loader:
            X_batch = X_batch.to(device)
            if hasattr(model, 'extract_features'):
                features = model.extract_features(X_batch)
            else:
                # Fallback: manually extract
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
    
    window_features = np.concatenate(all_features, axis=0)
    print(f"  Window features shape: {window_features.shape}")
    
    # Aggregate to per-timepoint
    timepoint_features = aggregate_window_features(
        window_features,
        n_timepoints=len(neural_data),
        window_size=window_size,
        stride=stride,
        method=aggregation_method
    )
    print(f"  Timepoint features shape: {timepoint_features.shape}")
    
    feature_info = {
        'feature_dim': timepoint_features.shape[1],
        'n_timepoints': timepoint_features.shape[0],
        'window_features_shape': list(window_features.shape),
        'aggregation_method': aggregation_method
    }
    
    return timepoint_features, feature_info


def main():
    parser = argparse.ArgumentParser(
        description="Unified pipeline: EEGNet feature extraction + CEBRA training"
    )
    
    # Data
    parser.add_argument(
        "--patient-id",
        type=int,
        required=True,
        help="Patient ID (1, 2, 9, 27, 28, etc.)"
    )
    parser.add_argument(
        "--high-gamma-only",
        action="store_true",
        help="Use only high gamma band"
    )
    
    # EEGNet options
    parser.add_argument(
        "--eegnet-model-path",
        type=Path,
        default=None,
        help="Path to existing EEGNet model. If not provided, will train new model."
    )
    parser.add_argument(
        "--train-eegnet",
        action="store_true",
        help="Train EEGNet (even if model path provided, will retrain)"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=32,
        help="Window size for EEGNet (default: 32)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride for sliding windows (default: 1)"
    )
    
    # EEGNet training parameters (if training)
    parser.add_argument(
        "--eegnet-F1",
        type=int,
        default=8,
        help="EEGNet F1 parameter (default: 8)"
    )
    parser.add_argument(
        "--eegnet-D",
        type=int,
        default=2,
        help="EEGNet D parameter (default: 2)"
    )
    parser.add_argument(
        "--eegnet-epochs",
        type=int,
        default=100,
        help="EEGNet training epochs (default: 100)"
    )
    parser.add_argument(
        "--eegnet-batch-size",
        type=int,
        default=32,
        help="EEGNet batch size (default: 32)"
    )
    parser.add_argument(
        "--eegnet-learning-rate",
        type=float,
        default=1e-3,
        help="EEGNet learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--eegnet-patience",
        type=int,
        default=15,
        help="EEGNet early stopping patience (default: 15)"
    )
    
    # Feature extraction
    parser.add_argument(
        "--feature-aggregation",
        type=str,
        default="mean",
        choices=["mean", "center", "max"],
        help="Feature aggregation method (default: mean)"
    )
    parser.add_argument(
        "--feature-batch-size",
        type=int,
        default=64,
        help="Batch size for feature extraction (default: 64)"
    )
    
    # CEBRA parameters
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
        "--cebra-batch-size",
        type=int,
        default=512,
        help="CEBRA batch size (default: 512)"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: output_EEGNet+xCEBRA/<patient_id>)"
    )
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using device: {device}")
    
    # Validate patient ID
    if args.patient_id not in PATIENT_CONFIG:
        known = ", ".join(map(str, sorted(PATIENT_CONFIG.keys())))
        parser.error(f"Unknown patient id {args.patient_id}. Available ids: {known}")
    
    ec_code, numeric_code = PATIENT_CONFIG[args.patient_id]
    print(f"\n[MODE] Single patient mode: {ec_code} (ID: {args.patient_id})")
    
    # Determine output directory
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / "output_EEGNet+xCEBRA" / numeric_code
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUTPUT] Results will be saved to: {args.output_dir}")
    
    # Load data
    extract_band = 4 if args.high_gamma_only else None
    print(f"\n[DATA] Loading patient {ec_code} data...")
    neural, emotion = load_single_patient_data(args.patient_id, extract_band=extract_band)
    print(f"[DATA] Neural shape: {neural.shape}, Emotion shape: {emotion.shape}")
    
    # ========== EEGNet Training/Loading ==========
    eegnet_model = None
    eegnet_config = None
    
    if args.train_eegnet or args.eegnet_model_path is None:
        print(f"\n{'='*60}")
        print(f"PHASE 1: Training EEGNet")
        print(f"{'='*60}")
        
        # Prepare data for EEGNet training
        unique_emotions = np.unique(emotion)
        nb_classes = len(unique_emotions)
        
        label_to_idx = {int(orig_label): idx for idx, orig_label in enumerate(sorted(unique_emotions))}
        idx_to_label = {idx: int(orig_label) for orig_label, idx in label_to_idx.items()}
        emotion_original = emotion.copy()
        emotion_remapped = np.array([label_to_idx[int(label)] for label in emotion], dtype=np.int32)
        emotion = emotion_remapped
        
        # Split train/test
        split_idx = int(len(neural) * 0.8)
        neural_train_eegnet = neural[:split_idx]
        emotion_train_eegnet = emotion[:split_idx]
        neural_test_eegnet = neural[split_idx:]
        emotion_test_eegnet = emotion[split_idx:]
        
        # Create windows
        X_train, y_train = create_windows(neural_train_eegnet, emotion_train_eegnet, args.window_size, args.stride)
        X_test, y_test = create_windows(neural_test_eegnet, emotion_test_eegnet, args.window_size, args.stride)
        
        Chans, Samples = X_train.shape[1], X_train.shape[2]
        kern_length = max(8, Samples // 4)
        
        # Create model
        eegnet_model = EEGNet(
            nb_classes=nb_classes,
            Chans=Chans,
            Samples=Samples,
            dropoutRate=0.5,
            kernLength=kern_length,
            F1=args.eegnet_F1,
            D=args.eegnet_D,
            F2=args.eegnet_F1 * args.eegnet_D,
            dropoutType='Dropout'
        ).to(device)
        
        # Save config
        eegnet_config = {
            "nb_classes": nb_classes,
            "Chans": Chans,
            "Samples": Samples,
            "dropoutRate": 0.5,
            "kernLength": kern_length,
            "F1": args.eegnet_F1,
            "D": args.eegnet_D,
            "F2": args.eegnet_F1 * args.eegnet_D,
            "dropoutType": "Dropout",
            "window_size": args.window_size,
            "stride": args.stride
        }
        
        # Train (need WindowDataset with labels for training)
        from src.train_eegnet import WindowDataset as WindowDatasetWithLabels
        train_dataset = WindowDatasetWithLabels(X_train, y_train)
        test_dataset = WindowDatasetWithLabels(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=args.eegnet_batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.eegnet_batch_size, shuffle=False)
        
        # Create EEGNet subfolder
        eegnet_dir = args.output_dir / "eegnet"
        eegnet_dir.mkdir(parents=True, exist_ok=True)
        
        model, history, test_labels, test_preds = train_eegnet(
            model=eegnet_model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            out_dir=eegnet_dir,
            epochs=args.eegnet_epochs,
            learning_rate=args.eegnet_learning_rate,
            patience=args.eegnet_patience
        )
        eegnet_model = model
        
        # Save config
        config_path = eegnet_dir / "eegnet_config.json"
        with open(config_path, 'w') as f:
            json.dump(eegnet_config, f, indent=2)
        
        print(f"[EEGNET] Training complete, model saved to {eegnet_dir}")
        
    else:
        print(f"\n{'='*60}")
        print(f"PHASE 1: Loading EEGNet model")
        print(f"{'='*60}")
        
        if not args.eegnet_model_path.exists():
            parser.error(f"Model file not found: {args.eegnet_model_path}")
        
        # Load config
        config_path = args.eegnet_model_path.parent / "eegnet_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                eegnet_config = json.load(f)
        else:
            parser.error(f"Config file not found: {config_path}")
        
        # Load model
        eegnet_model = EEGNet(**{k: v for k, v in eegnet_config.items() 
                                  if k not in ['window_size', 'stride']})
        eegnet_model.load_state_dict(torch.load(args.eegnet_model_path, map_location=device))
        eegnet_model = eegnet_model.to(device)
        eegnet_model.eval()
        
        print(f"[EEGNET] Model loaded from {args.eegnet_model_path}")
    
    # ========== Feature Extraction ==========
    print(f"\n{'='*60}")
    print(f"PHASE 2: Extracting EEGNet Features")
    print(f"{'='*60}")
    
    # Use original emotion labels (before remapping if training was done)
    if 'emotion_original' in locals():
        emotion_for_features = emotion_original
    else:
        emotion_for_features = emotion
    
    eegnet_features, feature_info = extract_eegnet_features(
        model=eegnet_model,
        neural_data=neural,
        emotion_labels=emotion_for_features,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.feature_batch_size,
        device=device,
        aggregation_method=args.feature_aggregation
    )
    
    # Save features in eegnet subfolder
    eegnet_dir = args.output_dir / "eegnet"
    eegnet_dir.mkdir(parents=True, exist_ok=True)
    features_path = eegnet_dir / "eegnet_features.npy"
    np.save(features_path, eegnet_features)
    print(f"[FEATURES] Saved to {features_path}")
    
    # ========== CEBRA Training ==========
    print(f"\n{'='*60}")
    print(f"PHASE 3: CEBRA Training on EEGNet Features")
    print(f"{'='*60}")
    
    # Convert to tensors
    neural_tensor = torch.tensor(eegnet_features, dtype=torch.float32)  # (T, F_eegnet)
    label_tensor = torch.tensor(emotion_for_features, dtype=torch.float32).unsqueeze(1)  # (T, 1)
    
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
    feature_dim = neural_train.shape[1]
    print(f"[CEBRA] Initializing model with {feature_dim} input features, {args.latent_dim} latent dim")
    
    encoder_model = init_model(
        name="offset10-model",
        num_neurons=feature_dim,
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
        dataset=train_dataset, batch_size=args.cebra_batch_size, num_steps=args.unsup_steps
    )
    supervised_loader = ContrastiveMultiObjectiveLoader(
        dataset=train_dataset, batch_size=args.cebra_batch_size, num_steps=args.sup_steps
    )
    
    # Build configs
    BEHAVIOR_INDICES = (0, args.latent_dim)
    unsupervised_config = build_cebra_config_unsupervised(unsupervised_loader, BEHAVIOR_INDICES)
    supervised_config = build_cebra_config_supervised(supervised_loader, BEHAVIOR_INDICES)
    
    # Unsupervised training
    print(f"\n[CEBRA] Phase A: Unsupervised pretraining ({args.unsup_steps} steps)")
    unsup_dir = args.output_dir / "xcebra_unsupervised"
    unsup_dir.mkdir(parents=True, exist_ok=True)
    
    solver_unsup = train_and_save(
        model=encoder_model,
        loader=unsupervised_loader,
        config=unsupervised_config,
        out_dir=unsup_dir,
        full_neural_tensor=neural_tensor,
        meta={
            "phase": "unsupervised",
            "latent_dim": args.latent_dim,
            "eegnet_config": eegnet_config,
            "feature_info": feature_info,
            "feature_source": "eegnet"
        },
        device=device,
        num_steps=args.unsup_steps
    )
    print(f"[CEBRA] Unsupervised training complete")
    
    # Supervised training
    print(f"\n[CEBRA] Phase B: Supervised fine-tuning ({args.sup_steps} steps)")
    sup_dir = args.output_dir / "xcebra_supervised"
    sup_dir.mkdir(parents=True, exist_ok=True)
    
    solver_sup = train_and_save(
        model=encoder_model,
        loader=supervised_loader,
        config=supervised_config,
        out_dir=sup_dir,
        full_neural_tensor=neural_tensor,
        meta={
            "phase": "supervised",
            "latent_dim": args.latent_dim,
            "eegnet_config": eegnet_config,
            "feature_info": feature_info,
            "feature_source": "eegnet"
        },
        device=device,
        num_steps=args.sup_steps
    )
    print(f"[CEBRA] Supervised training complete")
    
    # Load embeddings and plot
    print(f"\n[PLOTTING] Generating embedding plots...")
    Z_unsup_full = torch.load(unsup_dir / "embedding.pt")
    Z_sup_full = torch.load(sup_dir / "embedding.pt")
    
    Z_unsup = Z_unsup_full.squeeze(0).T  # (T, K)
    Z_sup = Z_sup_full.squeeze(0).T      # (T, K)
    
    y_full = label_tensor.squeeze(1).cpu().numpy()
    
    # Align and plot
    y_aligned_unsup, offset_unsup, split_unsup = align_embedding_labels(Z_unsup, y_full)
    plot_embedding_split(Z_unsup, y_aligned_unsup, split_unsup, unsup_dir, "emb_unsup", "Unsupervised")
    
    y_aligned_sup, offset_sup, split_sup = align_embedding_labels(Z_sup, y_full)
    plot_embedding_split(Z_sup, y_aligned_sup, split_sup, sup_dir, "emb_sup", "Supervised")
    
    # ========== Decoding (after CEBRA) ==========
    print(f"\n{'='*60}")
    print(f"PHASE 4: Decoding from CEBRA Embeddings")
    print(f"{'='*60}")
    
    # Load supervised embedding
    embedding = Z_sup  # Already loaded as (T, K)
    
    # Prepare labels (use emotion_for_features which matches the feature timepoints)
    y_full_decoding = emotion_for_features.astype(float)
    
    # Decode using existing full_decoding.py logic
    DECODERS = ["knn", "logreg"]
    dec_dir = args.output_dir / "decoding"
    dec_dir.mkdir(parents=True, exist_ok=True)
    all_timecourse = []
    rows = []
    
    for decoder in DECODERS:
        X_train, X_test, y_train, y_test, offset, split = _split_train_test(
            embedding, y_full_decoding
        )
        
        # Compute metrics
        coef, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
        y_pred_lin = X_test @ coef
        R2_behavior = r2_score(y_test, y_pred_lin)
        
        log_reg = LogisticRegression(max_iter=2000, solver="lbfgs")
        log_reg.fit(X_train, y_train)
        y_pred_logreg = log_reg.predict(X_test)
        acc_logreg = accuracy_score(y_test, y_pred_logreg)
        
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        acc_knn = accuracy_score(y_test, y_pred_knn)
        
        if decoder == "knn":
            acc = acc_knn
            y_pred = y_pred_knn
        elif decoder == "logreg":
            acc = acc_logreg
            y_pred = y_pred_logreg
        else:
            continue
        
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        
        # Save metrics
        result = {
            "patient": numeric_code,
            "decoder": decoder,
            "R2_behavior": f"{R2_behavior:.4f}",
            "accuracy": f"{acc:.4f}",
            "macroF1": f"{macro_f1:.4f}",
        }
        rows.append(result)
        print(f"[DECODING] [{decoder}] acc={acc:.3f}, R²={R2_behavior:.3f}, F1={macro_f1:.3f}")
        
        # Confusion matrix
        unique_labels = np.unique(np.concatenate([y_test, y_pred]))
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        label_names = [EMOTION_MAP.get(int(l), f"Class {int(l)}") for l in unique_labels]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
        disp.plot(xticks_rotation=90, cmap="Blues")
        plt.title(f"{decoder} confusion matrix (patient {numeric_code})")
        plt.tight_layout()
        plt.savefig(dec_dir / f"cm_{decoder}_{numeric_code}.png", dpi=150)
        plt.close()
        
        test_idx_local = np.arange(split, split + len(y_test))
        
        df_pair = collect_decoding_timecourse(
            pair_name=f"{decoder}_supervised",
            y_true=y_test,
            y_pred=y_pred,
            test_idx=test_idx_local,
        )
        all_timecourse.append(df_pair)
    
    # Save decoding summary
    df_summary = pd.DataFrame(rows)
    summary_path = dec_dir / "decoding_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"[DECODING] Summary saved to {summary_path}")
    
    # Timecourse visualization
    df_all = save_decoding_timecourse(all_timecourse, dec_dir / "decoding_timecourse.csv")
    if df_all is not None:
        plot_decoding_timecourses(
            csv_path=dec_dir / "decoding_timecourse.csv",
            out_path=dec_dir / "decoding_timecourse_grid.png",
            emotion_map=EMOTION_MAP,
            n_cols=2,
        )
        print(f"[DECODING] Timecourse plot saved to {dec_dir / 'decoding_timecourse_grid.png'}")
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: Pipeline finished successfully!")
    print(f"{'='*60}")
    print(f"[OUTPUT] Results saved to: {args.output_dir}")
    print(f"  - EEGNet outputs: {eegnet_dir}")
    print(f"    - Model: {eegnet_dir / 'eegnet_best_model.pt'}")
    print(f"    - Features: {features_path}")
    print(f"    - Metrics: {eegnet_dir / 'eegnet_results.json'} (F1, accuracy)")
    print(f"  - CEBRA unsupervised: {unsup_dir}")
    print(f"  - CEBRA supervised: {sup_dir}")
    print(f"  - Decoding: {dec_dir}")
    print(f"    - Summary: {dec_dir / 'decoding_summary.csv'} (F1, accuracy)")
    print(f"    - Timecourse: {dec_dir / 'decoding_timecourse_grid.png'}")


if __name__ == "__main__":
    main()
