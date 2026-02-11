"""
Train EEGNet on single-patient or aggregated multi-patient data for emotion classification.

This script:
1. Loads neural data (single patient from .mat files OR aggregated from .npz)
2. Creates sliding windows for EEGNet input (Chans, Samples)
3. Trains EEGNet model for emotion classification
4. Evaluates on test set and generates predictions

For single patient: trains on 80% of data, tests on 20%
For aggregated: trains on 80% of windows, validates on 20%
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import mat73
import scipy.io
import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eegnet_pytorch import EEGNet
from src.utils_visualization import collect_decoding_timecourse, save_decoding_timecourse, plot_decoding_timecourses

# Emotion map (from config.py, made local to avoid PATIENT_ID dependency)
EMOTION_MAP = {
    0: "No emotion",
    1: "Amusement",
    2: "Embarrassment",
    3: "Anger",
    4: "Confused",
    5: "Awe",
    6: "Disgust",
    7: "Fear",
    8: "Affection",
    9: "Sadness"
}

# Patient configuration (from other modules)
PATIENT_CONFIG = {
    1:    ("EC238", "238"),
    2:    ("EC239", "239"),
    9:    ("EC272", "272"),
    27:   ("EC301", "301"),
    28:   ("EC304", "304"),
    15:   ("EC280", "280"),
    22:   ("EC288", "288"),
    24:   ("EC293", "293"),
    29:   ("PR06", "PR06"),
    30:   ("EC325", "325"),
    31:   ("EC326", "326"),
}

DATA_SUBDIR = "nrcRF_stim_resp_5_Nfold_pairs_msBW_1000_wASpec16_v16_DC5_1   2   5   6   7   8   9  10  11  12__wASpec16_v16_DC5_1   2   5   6   7   8   9  10  11  12_5"
NEURAL_FILENAME = "nrcRF_calc_Stim_StimNum_5_Nr_1_msBW_1000_movHeldOut_1.mat"
EMOTION_FILENAME = "nrcRF_calc_Resp_chan_1_movHeldOut_1.mat"


class WindowDataset(Dataset):
    """Dataset for windowed neural data."""
    def __init__(self, X: np.ndarray, y: np.ndarray, window_groups: Optional[np.ndarray] = None):
        """
        Args:
            X: Windowed data of shape (N, Chans, Samples, 1)
            y: Labels of shape (N,) - window-level labels
            window_groups: Optional array of shape (N,) indicating which windows belong to same trial.
                          If None, each window is treated as independent.
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.window_groups = torch.LongTensor(window_groups) if window_groups is not None else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.window_groups is not None:
            return self.X[idx], self.y[idx], self.window_groups[idx]
        else:
            return self.X[idx], self.y[idx]


def create_windows(data: np.ndarray, labels: np.ndarray, window_size: int, 
                   stride: int = 1, group_by_trial: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Create sliding windows from time series data.
    
    Args:
        data: Neural data of shape (T, F) where T=timepoints, F=features
        labels: Emotion labels of shape (T,)
        window_size: Number of timepoints per window
        stride: Stride for sliding window
        group_by_trial: If True, group windows into trials based on contiguous emotion segments.
                       A trial is defined as a contiguous segment with the same emotion label.
        
    Returns:
        X: Windowed data of shape (N, F, window_size, 1) for EEGNet
        y: Labels of shape (N,) - uses majority label in each window
        window_groups: Optional array of shape (N,) indicating trial ID for each window.
                      Only returned if group_by_trial=True.
    """
    T, F = data.shape
    windows = []
    window_labels = []
    window_groups = [] if group_by_trial else None
    
    if group_by_trial:
        # Identify trial boundaries: contiguous segments with same emotion
        trial_id = 0
        current_trial_start = 0
        current_emotion = labels[0]
        
        for i in range(0, T - window_size + 1, stride):
            window = data[i:i+window_size]  # (window_size, F)
            window_label = labels[i:i+window_size]
            
            # Reshape to (F, window_size) for EEGNet (Chans, Samples)
            window = window.T  # (F, window_size)
            window = np.expand_dims(window, axis=-1)  # (F, window_size, 1)
            
            # Use majority label in window
            unique, counts = np.unique(window_label, return_counts=True)
            label = unique[np.argmax(counts)]
            
            # Check if we've moved to a new trial (emotion changed)
            center_idx = i + window_size // 2
            if center_idx < T:
                center_emotion = labels[center_idx]
                if center_emotion != current_emotion:
                    # New trial started
                    trial_id += 1
                    current_emotion = center_emotion
            
            windows.append(window)
            window_labels.append(label)
            window_groups.append(trial_id)
        
        window_groups = np.array(window_groups)
    else:
        # Original behavior: no trial grouping
        for i in range(0, T - window_size + 1, stride):
            window = data[i:i+window_size]  # (window_size, F)
            window_label = labels[i:i+window_size]
            
            # Reshape to (F, window_size) for EEGNet (Chans, Samples)
            window = window.T  # (F, window_size)
            window = np.expand_dims(window, axis=-1)  # (F, window_size, 1)
            
            # Use majority label in window (treat ALL classes equally, including 0 = no emotion)
            # This ensures "no emotion" is learned properly
            unique, counts = np.unique(window_label, return_counts=True)
            label = unique[np.argmax(counts)]  # Most common label in window
            
            windows.append(window)
            window_labels.append(label)
    
    X = np.array(windows)  # (N, F, window_size, 1)
    y = np.array(window_labels)
    
    return X, y, window_groups


def extract_high_gamma_band(data: np.ndarray, n_lags: int = 5, n_electrodes: int = 40, 
                             n_bands: int = 5, band_idx: int = 4) -> np.ndarray:
    """
    Extract features for a specific frequency band across all lags and electrodes.
    
    Feature structure: 5 lags * 40 electrodes * 5 bands = 1000 features
    Within each lag: 40 electrodes * 5 bands = 200 features
    For each electrode: 5 bands (indices 0-4, where 4 = high gamma/5th band)
    
    Args:
        data: Neural data of shape (T, 1000) 
        n_lags: Number of time lags (default: 5)
        n_electrodes: Number of electrodes (default: 40)
        n_bands: Number of frequency bands (default: 5)
        band_idx: Band index to extract (default: 4 = high gamma/5th band)
        
    Returns:
        Extracted band features of shape (T, n_lags * n_electrodes)
        Example: (T, 200) for high gamma (5 lags * 40 electrodes)
    """
    f_per_lag = n_electrodes * n_bands  # 200
    
    # For each lag, extract band_idx for all electrodes
    # Pattern: For lag l, electrode e, band b: index = l * f_per_lag + e * n_bands + b
    # We want band_idx (4) for all electrodes across all lags
    indices = []
    for lag in range(n_lags):
        base_lag = lag * f_per_lag
        for electrode in range(n_electrodes):
            idx = base_lag + electrode * n_bands + band_idx
            indices.append(idx)
    
    # Extract features
    extracted = data[:, indices]  # (T, n_lags * n_electrodes)
    
    return extracted


def load_single_patient_data(patient_id: int, extract_band: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and z-score a single patient's data.
    
    Args:
        patient_id: Patient ID (e.g., 1, 2, 9, 27, 28, etc.)
        extract_band: If specified (0-4), extract only this frequency band.
                     4 = high gamma (5th band). If None, use all features.
    
    Returns:
        neural: Z-scored neural data (T, F)
        emotion: Emotion labels (T,)
    """
    if patient_id not in PATIENT_CONFIG:
        known = ", ".join(map(str, sorted(PATIENT_CONFIG.keys())))
        raise KeyError(f"Unknown patient id {patient_id}. Available ids: {known}")
    
    ec_code, _ = PATIENT_CONFIG[patient_id]
    data_dir = PROJECT_ROOT / "data" / ec_code / DATA_SUBDIR
    neural_path = data_dir / NEURAL_FILENAME
    emotion_path = data_dir / EMOTION_FILENAME
    
    if not neural_path.exists():
        raise FileNotFoundError(f"Missing neural data: {neural_path}")
    if not emotion_path.exists():
        raise FileNotFoundError(f"Missing emotion data: {emotion_path}")
    
    print(f"[DATA] Loading patient {ec_code} (ID: {patient_id})...")
    neural = mat73.loadmat(str(neural_path))["stim"].T  # (T, F)
    emotion = scipy.io.loadmat(str(emotion_path))["resp"].flatten()
    
    # Apply same trimming as in aggregation (patient 239)
    if ec_code == "EC239" or patient_id == 2:
        max_timesteps = 630
        if neural.shape[0] > max_timesteps:
            print(f"[INFO] Trimming patient 239: keeping first {max_timesteps} of {neural.shape[0]} timepoints")
            neural = neural[:max_timesteps]
            emotion = emotion[:max_timesteps]
    
    if neural.shape[0] != emotion.shape[0]:
        raise ValueError(f"Sample mismatch: neural ({neural.shape[0]}), emotion ({emotion.shape[0]})")
    
    # Z-score per patient
    feature_means = neural.mean(axis=0)
    feature_stds = neural.std(axis=0)
    eps = 1e-6
    adjusted_stds = np.where(feature_stds < eps, 1.0, feature_stds)
    z_neural = (neural - feature_means) / adjusted_stds
    
    if extract_band is not None:
        print(f"[FILTER] Extracting band {extract_band} (0=band1, 1=band2, ..., 4=high_gamma)")
        original_shape = z_neural.shape
        z_neural = extract_high_gamma_band(z_neural, band_idx=extract_band)
        print(f"[FILTER] Filtered neural shape: {z_neural.shape} (was {original_shape})")
    
    return z_neural.astype(np.float32), emotion.astype(np.int32)


def load_aggregated_data(npz_path: Path, extract_band: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load aggregated neural data and emotion labels.
    
    Args:
        npz_path: Path to aggregated .npz file
        extract_band: If specified (0-4), extract only this frequency band.
                     4 = high gamma (5th band). If None, use all features.
    """
    data = np.load(npz_path)
    neural = data["neural"]  # (T, F) where F=1000 (all features)
    emotion = data["emotion"]  # (T,)
    
    if extract_band is not None:
        print(f"[FILTER] Extracting band {extract_band} (0=band1, 1=band2, ..., 4=high_gamma)")
        neural = extract_high_gamma_band(neural, band_idx=extract_band)
        print(f"[FILTER] Filtered neural shape: {neural.shape} (was {data['neural'].shape})")
    
    return neural.astype(np.float32), emotion.astype(np.int32)


def split_train_test(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, 
                     random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets.
    
    NOTE: This function is deprecated for time series data with overlapping windows.
    Use temporal split at timepoint level instead (before windowing) to avoid data leakage.
    
    For single patient: 80% train, 20% test
    For aggregated: 80% train, 20% test (validation)
    """
    np.random.seed(random_seed)
    indices = np.random.permutation(len(X))
    split_idx = int(len(X) * (1 - test_ratio))
    
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def train_eegnet(model: nn.Module,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 device: torch.device,
                 out_dir: Path,
                 epochs: int = 100,
                 learning_rate: float = 1e-3,
                 patience: int = 15,
                 idx_to_label: Optional[dict] = None,
                 class_weights: Optional[torch.Tensor] = None):
    """
    Train EEGNet model.
    
    Args:
        model: EEGNet model
        train_loader: Training data loader
        test_loader: Test data loader
        device: PyTorch device
        out_dir: Output directory
        epochs: Maximum epochs
        learning_rate: Initial learning rate
        patience: Early stopping patience
        class_weights: Optional class weights for imbalanced data
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model = model.to(device)
    
    # Use class weights if provided (for imbalanced data)
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"[TRAIN] Using class weights: {class_weights.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    
    # Track best VALIDATION loss (not training loss) for model selection
    best_test_loss = float('inf')
    best_test_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    print(f"[TRAIN] Starting training...")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Max epochs: {epochs}, Patience: {patience}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch in pbar:
            if len(batch) == 3:
                X_batch, y_batch, window_groups_batch = batch
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                window_groups_batch = window_groups_batch.to(device)
            else:
                X_batch, y_batch = batch
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                window_groups_batch = None
            
            optimizer.zero_grad()
            # With attention, we still get per-window predictions (enhanced with context)
            # So y_batch remains window-level labels
            if model.use_attention and window_groups_batch is not None:
                outputs = model(X_batch, window_groups=window_groups_batch)
            else:
                outputs = model(X_batch)
            
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*train_correct/train_total:.2f}%'})
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Test phase (evaluation during training)
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 3:
                    X_batch, y_batch, window_groups_batch = batch
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    window_groups_batch = window_groups_batch.to(device)
                else:
                    X_batch, y_batch = batch
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    window_groups_batch = None
                
                # With attention, we still get per-window predictions
                if model.use_attention and window_groups_batch is not None:
                    outputs = model(X_batch, window_groups=window_groups_batch)
                else:
                    outputs = model(X_batch)
                
                loss = criterion(outputs, y_batch)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += y_batch.size(0)
                test_correct += (predicted == y_batch).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        test_loss /= len(test_loader)
        test_acc = 100 * test_correct / test_total
        
        # Update learning rate based on validation loss
        scheduler.step(test_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        
        # Save best model based on VALIDATION loss (prevents overfitting)
        # Also track best accuracy as secondary metric
        improved = False
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            improved = True
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            if not improved:
                improved = True
        
        if improved:
            patience_counter = 0
            torch.save(model.state_dict(), out_dir / "eegnet_best_model.pt")
            print(f"  ✓ New best model saved (test loss: {test_loss:.4f}, test acc: {test_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[EARLY STOP] No improvement in test loss for {patience} epochs. Stopping.")
                break
    
    # Load best model
    if (out_dir / "eegnet_best_model.pt").exists():
        model.load_state_dict(torch.load(out_dir / "eegnet_best_model.pt"))
        print(f"\n[SAVE] Best model loaded from {out_dir / 'eegnet_best_model.pt'}")
    else:
        print(f"\n[WARNING] Best model checkpoint not found, using current model state")
    
    # Final evaluation
    print(f"\n[EVALUATION]")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                X_batch, y_batch, window_groups_batch = batch
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                window_groups_batch = window_groups_batch.to(device)
            else:
                X_batch, y_batch = batch
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                window_groups_batch = None
            
            # With attention, we still get per-window predictions
            if model.use_attention and window_groups_batch is not None:
                outputs = model(X_batch, window_groups=window_groups_batch)
            else:
                outputs = model(X_batch)
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    accuracy = 100 * (all_preds == all_labels).sum() / len(all_labels)
    
    print(f"  Test Accuracy: {accuracy:.2f}%")
    print(f"  Test F1 (macro): {f1_macro:.4f}")
    print(f"  Test F1 (weighted): {f1_weighted:.4f}")
    
    # Classification report
    print(f"\n[CLASSIFICATION REPORT]")
    class_report = classification_report(all_labels, all_preds, zero_division=0)
    print(class_report)
    
    # Save results as JSON
    results = {
        'test_acc': float(accuracy),
        'test_f1_macro': float(f1_macro),
        'test_f1_weighted': float(f1_weighted),
        'best_test_loss': float(best_test_loss),
        'best_test_acc': float(best_test_acc),
        'history': history
    }
    
    import json
    with open(out_dir / "eegnet_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save metrics as text file
    metrics_txt_path = out_dir / "eegnet_metrics.txt"
    with open(metrics_txt_path, 'w') as f:
        f.write("EEGNet Classification Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Accuracy: {accuracy:.2f}%\n")
        f.write(f"Test F1 (macro): {f1_macro:.4f}\n")
        f.write(f"Test F1 (weighted): {f1_weighted:.4f}\n")
        f.write(f"Best Test Loss: {best_test_loss:.4f}\n")
        f.write(f"Best Test Accuracy: {best_test_acc:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write("-" * 50 + "\n")
        f.write(class_report)
    print(f"[SAVE] Metrics saved to {metrics_txt_path}")
    
    # Plot training curves
    plot_training_curves(history, out_dir)
    
    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, len(np.unique(all_labels)), out_dir)
    
    return model, history, all_labels, all_preds


def plot_training_curves(history: dict, out_dir: Path):
    """Plot training and validation curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0].plot(epochs, history['test_loss'], label='Test Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Test Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], label='Train Acc')
    axes[1].plot(epochs, history['test_acc'], label='Test Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Test Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "eegnet_training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Training curves saved to {out_dir / 'eegnet_training_curves.png'}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                          nb_classes: int, out_dir: Path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=range(nb_classes))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(nb_classes),
                yticklabels=range(nb_classes))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_dir / "eegnet_confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Confusion matrix saved to {out_dir / 'eegnet_confusion_matrix.png'}")


def compute_class_weights(y: np.ndarray, method: str = 'balanced') -> Optional[torch.Tensor]:
    """
    Compute class weights for imbalanced data.
    
    Args:
        y: Array of class labels
        method: 'balanced' (sklearn style) or 'inverse_freq'
    
    Returns:
        Class weights tensor or None
    """
    unique, counts = np.unique(y, return_counts=True)
    n_classes = len(unique)
    
    if method == 'balanced':
        # sklearn style: n_samples / (n_classes * np.bincount(y))
        total = len(y)
        weights = total / (n_classes * counts)
    elif method == 'inverse_freq':
        # Inverse frequency
        weights = 1.0 / counts
    else:
        return None
    
    # Normalize weights
    weights = weights / weights.sum() * n_classes
    
    # Create weight tensor (indexed by class label)
    weight_dict = {int(label): float(weight) for label, weight in zip(unique, weights)}
    weight_tensor = torch.zeros(max(unique) + 1)
    for label, weight in weight_dict.items():
        weight_tensor[label] = weight
    
    print(f"[CLASS WEIGHTS] Method: {method}")
    print(f"  Class distribution: {dict(zip(unique, counts))}")
    print(f"  Weights: {weight_dict}")
    
    return weight_tensor


def analyze_class_distribution(y_train: np.ndarray, y_test: np.ndarray, 
                               idx_to_label: dict, emotion_map: dict, out_dir: Path):
    """Analyze and save class distribution information."""
    import json
    
    # Get class counts
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    test_unique, test_counts = np.unique(y_test, return_counts=True)
    
    # Create distribution report
    distribution = {
        'train': {int(label): int(count) for label, count in zip(train_unique, train_counts)},
        'test': {int(label): int(count) for label, count in zip(test_unique, test_counts)},
        'train_percentages': {int(label): float(100*count/len(y_train)) 
                             for label, count in zip(train_unique, train_counts)},
        'test_percentages': {int(label): float(100*count/len(y_test)) 
                            for label, count in zip(test_unique, test_counts)},
        'label_mapping': {int(model_idx): emotion_map.get(int(orig_label), f"Unknown({orig_label})")
                         for model_idx, orig_label in idx_to_label.items()}
    }
    
    # Save to JSON
    dist_path = out_dir / "class_distribution.json"
    with open(dist_path, 'w') as f:
        json.dump(distribution, f, indent=2)
    
    # Print summary
    print(f"\n[CLASS DISTRIBUTION]")
    print(f"  Train set: {len(y_train)} samples")
    for label in sorted(train_unique):
        orig_label = idx_to_label[int(label)]
        emotion_name = emotion_map.get(int(orig_label), f"Unknown({orig_label})")
        count = distribution['train'][int(label)]
        pct = distribution['train_percentages'][int(label)]
        print(f"    Class {label} ({emotion_name}): {count} ({pct:.1f}%)")
    
    print(f"  Test set: {len(y_test)} samples")
    for label in sorted(test_unique):
        orig_label = idx_to_label[int(label)]
        emotion_name = emotion_map.get(int(orig_label), f"Unknown({orig_label})")
        count = distribution['test'][int(label)]
        pct = distribution['test_percentages'][int(label)]
        print(f"    Class {label} ({emotion_name}): {count} ({pct:.1f}%)")
    
    print(f"[SAVE] Class distribution saved to {dist_path}")
    
    return distribution


def main():
    parser = argparse.ArgumentParser(
        description="Train EEGNet on single-patient or aggregated multi-patient data for emotion classification."
    )
    
    # Data source: choose one
    parser.add_argument(
        "--patient-id",
        type=int,
        default=None,
        help="Patient ID (1, 2, 9, 27, 28, etc.). If specified, loads single patient data from .mat files."
    )
    parser.add_argument(
        "--aggregated-npz",
        type=Path,
        default=None,
        help="Path to aggregated .npz file. Use this OR --patient-id, not both."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: output_eegnet/<patient_code> for single patient, or <npz_dir>/eegnet_results for aggregated)"
    )
    
    # Data parameters
    parser.add_argument(
        "--window-size",
        type=int,
        default=32,
        help="Number of timepoints per window (Samples parameter for EEGNet). Each timepoint = 1 second. Default 32 captures ~1 emotion (30 sec) with small context."
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride for sliding window creation"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--high-gamma-only",
        action="store_true",
        help="Use only high gamma band (5th band, index 4). Extracts 200 features (5 lags * 40 electrodes) instead of 1000."
    )
    
    # Model parameters
    parser.add_argument(
        "--F1",
        type=int,
        default=8,
        help="Number of temporal filters (F1)"
    )
    parser.add_argument(
        "--D",
        type=int,
        default=2,
        help="Number of spatial filters (D)"
    )
    parser.add_argument(
        "--kern-length",
        type=int,
        default=None,
        help="Temporal kernel length (default: window_size // 4)"
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.5,
        help="Dropout rate"
    )
    
    # Training parameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Use class weights to handle imbalanced data"
    )
    parser.add_argument(
        "--class-weight-method",
        type=str,
        default="balanced",
        choices=["balanced", "inverse_freq"],
        help="Method for computing class weights"
    )
    
    # Attention parameters
    parser.add_argument(
        "--use-attention",
        action="store_true",
        help="Use self-attention to aggregate window-level predictions into trial-level predictions"
    )
    parser.add_argument(
        "--attention-heads",
        type=int,
        default=8,
        help="Number of attention heads (default: 8)"
    )
    parser.add_argument(
        "--attention-dim",
        type=int,
        default=None,
        help="Dimension for attention layer (default: flat_size from EEGNet features)"
    )
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using device: {device}")
    
    # Validate arguments
    if args.patient_id is None and args.aggregated_npz is None:
        parser.error("Must specify either --patient-id OR --aggregated-npz")
    if args.patient_id is not None and args.aggregated_npz is not None:
        parser.error("Cannot specify both --patient-id and --aggregated-npz. Choose one.")
    
    # Load data
    extract_band = 4 if args.high_gamma_only else None
    
    if args.patient_id is not None:
        # Single patient mode
        if args.patient_id not in PATIENT_CONFIG:
            known = ", ".join(map(str, sorted(PATIENT_CONFIG.keys())))
            parser.error(f"Unknown patient id {args.patient_id}. Available ids: {known}")
        
        ec_code, numeric_code = PATIENT_CONFIG[args.patient_id]
        print(f"[MODE] Single patient mode: {ec_code} (ID: {args.patient_id})")
        
        neural, emotion = load_single_patient_data(args.patient_id, extract_band=extract_band)
        
        # Determine output directory
        if args.output_dir is None:
            args.output_dir = PROJECT_ROOT / "output_eegnet" / numeric_code
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[OUTPUT] Results will be saved to: {args.output_dir}")
        
    else:
        # Aggregated mode
        if not args.aggregated_npz.exists():
            parser.error(f"Aggregated file not found: {args.aggregated_npz}")
        
        print(f"[MODE] Aggregated multi-patient mode")
        print(f"[DATA] Loading aggregated data from {args.aggregated_npz}")
        neural, emotion = load_aggregated_data(args.aggregated_npz, extract_band=extract_band)
        
        # Determine output directory
        if args.output_dir is None:
            args.output_dir = args.aggregated_npz.parent / "eegnet_results"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[OUTPUT] Results will be saved to: {args.output_dir}")
    
    print(f"[DATA] Neural shape: {neural.shape}, Emotion shape: {emotion.shape}")
    
    # Get unique emotions (should be 0-9, where 0 = no emotion)
    # Note: Some emotion classes may be missing from this patient's data
    unique_emotions = np.unique(emotion)
    expected_classes = set(range(10))  # 0-9: 0=no emotion, 1-9=emotions
    missing_classes = expected_classes - set(unique_emotions)
    
    nb_classes = len(unique_emotions)
    print(f"[DATA] Number of emotion classes in data: {nb_classes} (should be 10: 0=no emotion, 1-9=emotions)")
    print(f"[DATA] Present emotion classes: {sorted(unique_emotions)}")
    if missing_classes:
        print(f"[DATA] Missing emotion classes (not in this patient's data): {sorted(missing_classes)}")
    
    # Remap labels to consecutive indices (0, 1, 2, ..., nb_classes-1)
    # This is necessary because labels have gaps (missing classes 2, 3 in your case)
    # Model expects labels in range [0, nb_classes-1] where nb_classes = number of classes present
    label_to_idx = {int(orig_label): idx for idx, orig_label in enumerate(sorted(unique_emotions))}
    idx_to_label = {idx: int(orig_label) for orig_label, idx in label_to_idx.items()}
    
    # Save original emotion labels BEFORE remapping (for timecourse visualization)
    emotion_original = emotion.copy()
    
    # Apply mapping to emotion labels
    emotion_remapped = np.array([label_to_idx[int(label)] for label in emotion], dtype=np.int32)
    print(f"[DATA] Remapped emotion classes (for model): {sorted(np.unique(emotion_remapped))}")
    print(f"[DATA] Label mapping (original -> model_index): {label_to_idx}")
    
    # Use remapped labels
    emotion = emotion_remapped
    
    # Split at timepoint level FIRST (before windowing) to avoid data leakage
    # This ensures overlapping windows don't end up in both train and test
    print(f"\n[SPLIT] Splitting timepoints into train/test (80/20)...")
    test_ratio = args.val_ratio if hasattr(args, 'val_ratio') else 0.2
    split_idx = int(len(neural) * (1 - test_ratio))
    
    neural_train = neural[:split_idx]
    emotion_train = emotion[:split_idx]
    neural_test = neural[split_idx:]
    emotion_test = emotion[split_idx:]
    
    print(f"  Train timepoints: {len(neural_train)} ({100*(1-test_ratio):.0f}%)")
    print(f"  Test timepoints: {len(neural_test)} ({100*test_ratio:.0f}%)")
    
    # Create windows separately for train and test
    print(f"\n[WINDOWING] Creating sliding windows...")
    print(f"  Window size: {args.window_size}, Stride: {args.stride}")
    group_by_trial = getattr(args, 'use_attention', False)
    X_train, y_train, window_groups_train = create_windows(
        neural_train, emotion_train, args.window_size, args.stride, 
        group_by_trial=group_by_trial
    )
    X_test, y_test, window_groups_test = create_windows(
        neural_test, emotion_test, args.window_size, args.stride,
        group_by_trial=group_by_trial
    )
    
    print(f"[WINDOWING] Created windows:")
    print(f"  Train: {len(X_train)} windows")
    print(f"  Test: {len(X_test)} windows")
    print(f"  X shape: (N, Chans, Samples, 1)")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    Chans, Samples = X_train.shape[1], X_train.shape[2]
    print(f"  Channels: {Chans}, Samples per window: {Samples}")
    
    # Create data loaders
    train_dataset = WindowDataset(X_train, y_train, window_groups_train)
    test_dataset = WindowDataset(X_test, y_test, window_groups_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Default kernel length
    if args.kern_length is None:
        args.kern_length = max(8, Samples // 4)
    
    # Create model
    print(f"\n[MODEL] Creating EEGNet model...")
    print(f"  Chans={Chans}, Samples={Samples}, Classes={nb_classes}")
    print(f"  F1={args.F1}, D={args.D}, kern_length={args.kern_length}, dropout={args.dropout_rate}")
    
    use_attention = getattr(args, 'use_attention', False)
    attention_heads = getattr(args, 'attention_heads', 8)
    attention_dim = getattr(args, 'attention_dim', None)
    
    model = EEGNet(
        nb_classes=nb_classes,
        Chans=Chans,
        Samples=Samples,
        dropoutRate=args.dropout_rate,
        kernLength=args.kern_length,
        F1=args.F1,
        D=args.D,
        F2=args.F1 * args.D,
        dropoutType='Dropout',
        use_attention=use_attention,
        attention_heads=attention_heads,
        attention_dim=attention_dim
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    
    # Save model configuration for later use (e.g., feature extraction)
    import json
    model_config = {
        "nb_classes": nb_classes,
        "Chans": Chans,
        "Samples": Samples,
        "dropoutRate": args.dropout_rate,
        "kernLength": args.kern_length,
        "F1": args.F1,
        "D": args.D,
        "F2": args.F1 * args.D,
        "dropoutType": "Dropout",
        "window_size": args.window_size,
        "stride": args.stride
    }
    config_path = args.output_dir / "eegnet_config.json"
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    print(f"[INFO] Model configuration saved to {config_path}")
    
    # Save label mapping for interpretation later
    label_mapping_path = args.output_dir / "label_mapping.json"
    with open(label_mapping_path, 'w') as f:
        json.dump({
            'original_to_model': label_to_idx,
            'model_to_original': idx_to_label
        }, f, indent=2)
    print(f"[INFO] Label mapping saved to {label_mapping_path}")
    
    # Analyze class distribution
    emotion_map = {int(orig_label): EMOTION_MAP.get(int(orig_label), f"Unknown({orig_label})") 
                   for orig_label in idx_to_label.values()}
    class_distribution = analyze_class_distribution(
        y_train, y_test, idx_to_label, emotion_map, args.output_dir
    )
    
    # Compute class weights if requested
    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(y_train, method=args.class_weight_method)
    
    # Train model
    print(f"\n[TRAINING] Starting training...")
    model, history, test_labels, test_preds = train_eegnet(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        out_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        class_weights=class_weights
    )
    
    # Create timecourse visualization using ORIGINAL timepoint-level labels
    # This shows the actual emotion labels at each timepoint, including "no emotion" (0)
    print(f"\n[TIMECOURSE] Creating timecourse with original timepoint-level labels...")
    
    # Get original emotion labels for test timepoints (before remapping/windowing)
    emotion_test_original = emotion_original[split_idx:]
    test_timepoint_indices = split_idx + np.arange(len(emotion_test_original))
    test_timepoint_labels = emotion_test_original
    
    # Map window predictions to timepoints
    # Each window prediction applies to its center timepoint
    test_timepoint_preds = np.zeros_like(emotion_test_original, dtype=np.int32)
    for window_idx in range(len(test_preds)):
        # Map window prediction back to original label
        pred_original = idx_to_label[int(test_preds[window_idx])]
        # Assign to center timepoint of this window
        center_idx = window_idx + args.window_size // 2
        if center_idx < len(test_timepoint_preds):
            test_timepoint_preds[center_idx] = pred_original
    
    # Forward-fill predictions to fill gaps between windows (optional - comment out if you want sparse predictions)
    # for i in range(1, len(test_timepoint_preds)):
    #     if test_timepoint_preds[i] == 0 and test_timepoint_preds[i-1] != 0:
    #         test_timepoint_preds[i] = test_timepoint_preds[i-1]
    
    print(f"\n[TIMECOURSE] Creating decoding timecourse visualization...")
    
    # emotion_map already created above
    print(f"[TIMECOURSE] Using emotion labels: {emotion_map}")
    
    # Collect timecourse data using timepoint-level labels
    df_timecourse = collect_decoding_timecourse(
        pair_name="EEGNet",
        y_true=test_timepoint_labels,
        y_pred=test_timepoint_preds,
        test_idx=test_timepoint_indices
    )
    
    # Save and plot timecourse
    timecourse_csv_path = args.output_dir / "decoding_timecourse.csv"
    df_all = save_decoding_timecourse([df_timecourse], timecourse_csv_path)
    
    if df_all is not None:
        plot_decoding_timecourses(
            csv_path=timecourse_csv_path,
            out_path=args.output_dir / "decoding_timecourse_grid.png",
            emotion_map=emotion_map,
            n_cols=1
        )
    
    print(f"\n[COMPLETE] Training finished!")
    print(f"[RESULTS] Check output directory: {args.output_dir}")
    print(f"  - Model weights: {args.output_dir / 'eegnet_best_model.pt'}")
    print(f"  - Training curves: {args.output_dir / 'eegnet_training_curves.png'}")
    print(f"  - Confusion matrix: {args.output_dir / 'eegnet_confusion_matrix.png'}")
    print(f"  - Results JSON: {args.output_dir / 'eegnet_results.json'}")
    print(f"  - Metrics (txt): {args.output_dir / 'eegnet_metrics.txt'}")
    print(f"  - Timecourse CSV: {args.output_dir / 'decoding_timecourse.csv'}")
    print(f"  - Timecourse plot: {args.output_dir / 'decoding_timecourse_grid.png'}")
    
    print(f"\n[DONE] Training complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
