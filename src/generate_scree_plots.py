"""
Generate scree plots from existing training checkpoints/embeddings.

This script loads pre-trained embeddings and generates scree plots to analyze
variance captured by each principal component. Useful for determining optimal
latent dimensions without retraining.

Usage:
    python src/generate_scree_plots.py \
        --output-dir output_patient_aggregation/238_239_272_301 \
        [--test-patient-id 28] \
        [--latent-dim 16]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Optional
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import constants and helper functions
import mat73
import scipy.io
import torch.nn as nn

# Patient configuration
PATIENT_CONFIG = {
    1:    ("EC238", "238"),
    2:    ("EC239", "239"),
    9:    ("EC272", "272"),
    27:    ("EC301", "301"),
    28:    ("EC304", "304"),
    15: ("EC280", "280"),
    22: ("EC288", "288"),
    24: ("EC293", "293"),
    29: ("PR06", "PR06"),
    30: ("EC325", "325"),
    31: ("EC326", "326"),
}

# Data paths (from patient_aggreagation_encoding_finetune.py)
DATA_SUBDIR = "nrcRF_stim_resp_5_Nfold_pairs_msBW_1000_wASpec16_v16_DC5_1   2   5   6   7   8   9  10  11  12__wASpec16_v16_DC5_1   2   5   6   7   8   9  10  11  12_5"
NEURAL_FILENAME = "nrcRF_calc_Stim_StimNum_5_Nr_1_msBW_1000_movHeldOut_1.mat"
EMOTION_FILENAME = "nrcRF_calc_Resp_chan_1_movHeldOut_1.mat"

def load_test_patient_data(patient_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load and z-score a single patient's data for testing."""
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
    
    # Z-score per patient (same as aggregation)
    feature_means = neural.mean(axis=0)
    feature_stds = neural.std(axis=0)
    eps = 1e-6
    adjusted_stds = np.where(feature_stds < eps, 1.0, feature_stds)
    z_neural = (neural - feature_means) / adjusted_stds
    
    return z_neural.astype(np.float32), emotion.astype(np.int32)

def generate_embedding_for_patient(encoder: nn.Module, neural_data: np.ndarray, device: torch.device) -> np.ndarray:
    """Generate embedding for a patient's neural data using the encoder."""
    encoder.eval()
    with torch.no_grad():
        X = torch.tensor(neural_data, dtype=torch.float32).to(device)  # (T, F)
        X_bct = X.transpose(0, 1).unsqueeze(0)  # (1, F, T)
        Z_bdt = encoder(X_bct)  # (1, D, T')
        Z = Z_bdt.squeeze(0).T.cpu().numpy()  # (T', D)
    return Z

def plot_scree_analysis(Z_train: np.ndarray, Z_test: np.ndarray, 
                        out_dir: Path, prefix: str, title_prefix: str,
                        latent_dim: int):
    """
    Generate scree plots to analyze variance captured by each principal component
    of the learned embeddings. This helps assess if the latent dimension is appropriate.
    
    Args:
        Z_train: Training embeddings (T_train, D)
        Z_test: Test embeddings (T_test, D)
        out_dir: Output directory
        prefix: Prefix for output filenames
        title_prefix: Prefix for plot titles
        latent_dim: Current latent dimension used
    """
    # Compute PCA on all dimensions (limited by actual embedding dim)
    max_components = min(Z_train.shape[1], Z_train.shape[0], Z_test.shape[0])
    if max_components < 2:
        print(f"[WARN] Insufficient data for scree plot: max_components={max_components}")
        return
    
    # Fit PCA on training data, transform both
    pca_full = PCA(n_components=max_components)
    Z_train_pca = pca_full.fit_transform(Z_train)
    Z_test_pca = pca_full.transform(Z_test)
    
    # Get explained variance ratios
    explained_var_ratio = pca_full.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var_ratio)
    
    # Create scree plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    pc_numbers = np.arange(1, len(explained_var_ratio) + 1)
    
    # 1. Individual variance explained (bar plot)
    ax1 = axes[0, 0]
    ax1.bar(pc_numbers, explained_var_ratio, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(x=latent_dim, color='red', linestyle='--', linewidth=2, 
                label=f'Current latent_dim={latent_dim}')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Variance Explained Ratio')
    ax1.set_title(f'{title_prefix}: Variance Explained per PC\n(Individual)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xticks(pc_numbers[::max(1, len(pc_numbers)//10)])
    
    # 2. Individual variance explained (line plot, easier to see elbow)
    ax2 = axes[0, 1]
    ax2.plot(pc_numbers, explained_var_ratio, marker='o', linewidth=2, markersize=6, color='steelblue')
    ax2.axvline(x=latent_dim, color='red', linestyle='--', linewidth=2, 
                label=f'Current latent_dim={latent_dim}')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Variance Explained Ratio')
    ax2.set_title(f'{title_prefix}: Variance Explained per PC\n(Line Plot - Look for "Elbow")')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xticks(pc_numbers[::max(1, len(pc_numbers)//10)])
    
    # 3. Cumulative variance explained
    ax3 = axes[1, 0]
    ax3.plot(pc_numbers, cumulative_var, marker='o', linewidth=2, markersize=6, color='darkgreen')
    ax3.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='80% variance')
    ax3.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='90% variance')
    ax3.axhline(y=0.95, color='purple', linestyle='--', alpha=0.7, label='95% variance')
    ax3.axvline(x=latent_dim, color='red', linestyle='--', linewidth=2, alpha=0.5,
                label=f'Current latent_dim={latent_dim}')
    ax3.set_xlabel('Number of Principal Components')
    ax3.set_ylabel('Cumulative Variance Explained')
    ax3.set_title(f'{title_prefix}: Cumulative Variance Explained')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='lower right')
    ax3.set_ylim([0, 1.05])
    ax3.set_xticks(pc_numbers[::max(1, len(pc_numbers)//10)])
    
    # 4. Comparison: Training vs Test variance
    # Compute PCA separately on test data for comparison
    pca_test = PCA(n_components=max_components)
    Z_test_pca_separate = pca_test.fit_transform(Z_test)
    explained_var_test = pca_test.explained_variance_ratio_
    
    ax4 = axes[1, 1]
    ax4.plot(pc_numbers, explained_var_ratio[:len(pc_numbers)], marker='o', 
             linewidth=2, markersize=6, label='Training', color='steelblue')
    ax4.plot(pc_numbers, explained_var_test[:len(pc_numbers)], marker='s', 
             linewidth=2, markersize=6, label='Test', color='coral', alpha=0.7)
    ax4.axvline(x=latent_dim, color='red', linestyle='--', linewidth=2, alpha=0.5,
                label=f'Current latent_dim={latent_dim}')
    ax4.set_xlabel('Principal Component')
    ax4.set_ylabel('Variance Explained Ratio')
    ax4.set_title(f'{title_prefix}: Training vs Test Variance\n(Should be similar for generalization)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_xticks(pc_numbers[::max(1, len(pc_numbers)//10)])
    
    plt.tight_layout()
    
    # Save plot
    out_dir.mkdir(parents=True, exist_ok=True)
    scree_path = out_dir / f"{prefix}_scree_plot.png"
    plt.savefig(scree_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SCREE] Saved scree plot → {scree_path}")
    
    # Print analysis summary
    print(f"\n{'='*60}")
    print(f"[SCREE ANALYSIS] {title_prefix}")
    print(f"{'='*60}")
    print(f"Current latent_dim: {latent_dim}")
    print(f"Total embedding dimensions: {Z_train.shape[1]}")
    print(f"\nTop 5 Principal Components (Training):")
    for i in range(min(5, len(explained_var_ratio))):
        print(f"  PC{i+1}: {explained_var_ratio[i]:.4f} ({explained_var_ratio[i]*100:.2f}%)")
    
    print(f"\nCumulative Variance Explained (Training):")
    for n_components in [2, 3, 5, 8, 10, 16, 32]:
        if n_components <= len(cumulative_var):
            print(f"  First {n_components} PCs: {cumulative_var[n_components-1]:.4f} ({cumulative_var[n_components-1]*100:.2f}%)")
    
    # Find elbow (simple method: where the drop becomes small)
    if len(explained_var_ratio) >= 3:
        drops = explained_var_ratio[:-1] - explained_var_ratio[1:]
        if len(drops) > 0:
            # Find where drops become consistently small (less than median drop)
            median_drop = np.median(drops)
            small_drops = np.where(drops < median_drop * 0.5)[0]
            if len(small_drops) > 0:
                elbow_estimate = small_drops[0] + 2  # +2 because drop[i] is between PC i+1 and i+2
                print(f"\nEstimated 'Elbow' (where variance drops become small): ~{elbow_estimate} PCs")
    
    # Recommendations
    print(f"\n[RECOMMENDATIONS]:")
    var_at_latent_dim = cumulative_var[min(latent_dim, len(cumulative_var))-1] if latent_dim <= len(cumulative_var) else cumulative_var[-1]
    if var_at_latent_dim < 0.8:
        print(f"  ⚠️  Only {var_at_latent_dim*100:.1f}% variance captured by {latent_dim} dims.")
        print(f"     Consider increasing --latent-dim if validation F1 is low.")
    elif var_at_latent_dim > 0.95:
        print(f"  ✓  {var_at_latent_dim*100:.1f}% variance captured by {latent_dim} dims.")
        print(f"     Current dimension may be sufficient. If overfitting, consider reducing.")
    else:
        print(f"  →  {var_at_latent_dim*100:.1f}% variance captured by {latent_dim} dims.")
        print(f"     Balanced representation. Monitor validation F1 to guide further changes.")
    
    if explained_var_ratio[0] + explained_var_ratio[1] > 0.7:
        print(f"  ✓  First 2 PCs capture {explained_var_ratio[0]*100 + explained_var_ratio[1]*100:.1f}% - Good for 2D visualization!")
    print(f"{'='*60}\n")


def load_embedding_TxD(emb_path: Path) -> np.ndarray:
    """Load .pt embedding and return [T, D] numpy array."""
    embedding = torch.load(emb_path, map_location="cpu")
    if embedding.ndim == 3:
        # Shape: (1, D, T) -> (T, D)
        if embedding.shape[0] == 1:
            embedding = embedding.squeeze(0).T  # (D, T) -> (T, D)
        else:
            raise ValueError(f"Unexpected 3D shape: {tuple(embedding.shape)}")
    elif embedding.ndim == 2:
        # Already (T, D) or (D, T) - check which dimension is larger
        if embedding.shape[0] < embedding.shape[1]:
            embedding = embedding.T  # Assume (D, T) -> (T, D)
        # Otherwise assume (T, D) is correct
    else:
        raise ValueError(f"Unexpected embedding shape {tuple(embedding.shape)}")
    return embedding.numpy()


def load_latent_dim_from_metadata(output_dir: Path) -> Optional[int]:
    """Try to load latent_dim from finetune_meta.pt"""
    meta_path = output_dir / "finetune_meta.pt"
    if meta_path.exists():
        try:
            meta = torch.load(meta_path, map_location="cpu", weights_only=False)
            if "hyperparams" in meta and "latent_dim" in meta["hyperparams"]:
                return int(meta["hyperparams"]["latent_dim"])
        except Exception as e:
            print(f"[WARN] Could not load latent_dim from metadata: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate scree plots from existing training checkpoints."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory containing embeddings (e.g., output_patient_aggregation/238_239_272_301)"
    )
    parser.add_argument(
        "--test-patient-id",
        type=int,
        default=None,
        help="Optional: Patient ID to include in scree analysis (e.g., 28 for EC304)"
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=None,
        help="Latent dimension used. If not provided, will try to load from finetune_meta.pt"
    )
    parser.add_argument(
        "--phases",
        type=str,
        nargs="+",
        default=["unsupervised", "supervised", "finetuned"],
        choices=["unsupervised", "supervised", "finetuned"],
        help="Which phases to generate scree plots for"
    )
    
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    # Determine latent_dim
    latent_dim = args.latent_dim
    if latent_dim is None:
        latent_dim = load_latent_dim_from_metadata(output_dir)
        if latent_dim is None:
            print("[WARN] Could not determine latent_dim. Using default 16.")
            latent_dim = 16
        else:
            print(f"[INFO] Loaded latent_dim={latent_dim} from metadata")
    else:
        print(f"[INFO] Using provided latent_dim={latent_dim}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate scree plots for each requested phase
    for phase in args.phases:
        print(f"\n{'='*70}")
        print(f"Processing phase: {phase.upper()}")
        print(f"{'='*70}")
        
        if phase == "unsupervised":
            unsup_dir = output_dir / "xcebra_unsupervised"
            embedding_path = unsup_dir / "embedding.pt"
            prefix = "emb_unsup"
            title = "Unsupervised Embeddings"
            
        elif phase == "supervised":
            sup_dir = output_dir / "xcebra_supervised"
            embedding_path = sup_dir / "embedding.pt"
            prefix = "emb_sup"
            title = "Supervised Embeddings"
            
        elif phase == "finetuned":
            embedding_path = output_dir / "embedding_finetuned.pt"
            prefix = "emb_finetuned"
            title = "Finetuned Embeddings"
            unsup_dir = output_dir  # Save plot in main output dir
        
        if not embedding_path.exists():
            print(f"[SKIP] Embedding not found: {embedding_path}")
            continue
        
        print(f"[LOAD] Loading embedding from {embedding_path}")
        Z_train = load_embedding_TxD(embedding_path)
        print(f"[LOAD] Loaded embedding shape: {Z_train.shape}")
        
        # Get test embeddings if test patient is specified
        Z_test = None
        encoder = None
        
        if args.test_patient_id is not None:
            print(f"[LOAD] Generating embeddings for test patient {args.test_patient_id}...")
            
            # Load test patient data
            test_neural, _ = load_test_patient_data(args.test_patient_id)
            test_ec_code, test_patient_code = PATIENT_CONFIG[args.test_patient_id]
            
            # Find aggregated .npz to get input dimensions
            aggregated_npz = output_dir / "aggregated_patient_data_238_239_272_301.npz"
            if not aggregated_npz.exists():
                # Try to find any .npz in the directory
                npz_files = list(output_dir.glob("*.npz"))
                if npz_files:
                    aggregated_npz = npz_files[0]
                    print(f"[INFO] Using aggregated data: {aggregated_npz.name}")
            
            if not aggregated_npz.exists():
                print(f"[WARN] Cannot find aggregated .npz file. Using training split instead.")
                split_idx = len(Z_train) // 5
                Z_test = Z_train[-split_idx:]
                Z_train = Z_train[:-split_idx]
            else:
                # Load encoder based on phase
                from cebra.models import init as init_model
                data = np.load(aggregated_npz)
                num_neurons = data["neural"].shape[1]
                
                if phase == "finetuned":
                    encoder_path = output_dir / "encoder_finetuned.pt"
                    if encoder_path.exists():
                        encoder_data = torch.load(encoder_path, map_location="cpu", weights_only=False)
                        encoder_latent_dim = encoder_data.get("latent_dim", latent_dim)
                        encoder = init_model(
                            name="offset10-model",
                            num_neurons=num_neurons,
                            num_units=256,
                            num_output=encoder_latent_dim
                        ).to(device)
                        encoder.load_state_dict(encoder_data["state_dict"])
                        encoder.eval()
                else:
                    # For unsupervised/supervised phases
                    if phase == "unsupervised":
                        phase_dir = output_dir / "xcebra_unsupervised"
                    else:
                        phase_dir = output_dir / "xcebra_supervised"
                    
                    model_path = phase_dir / "model_weights.pt"
                    if model_path.exists():
                        encoder = init_model(
                            name="offset10-model",
                            num_neurons=num_neurons,
                            num_units=256,
                            num_output=latent_dim
                        ).to(device)
                        encoder.load_state_dict(torch.load(model_path, map_location=device))
                        encoder.eval()
                
                if encoder is not None:
                    Z_test = generate_embedding_for_patient(encoder, test_neural, device)
                    print(f"[LOAD] Test embedding shape: {Z_test.shape}")
                else:
                    print(f"[WARN] Could not load encoder. Using training split instead.")
                    split_idx = len(Z_train) // 5
                    Z_test = Z_train[-split_idx:]
                    Z_train = Z_train[:-split_idx]
        else:
            # No test patient - use training data split
            print(f"[INFO] No test patient specified. Using training data split for comparison.")
            split_idx = len(Z_train) // 5
            Z_test = Z_train[-split_idx:]
            Z_train = Z_train[:-split_idx]
            print(f"[INFO] Split: train={Z_train.shape}, test={Z_test.shape}")
        
        # Determine output directory for plots
        if phase == "finetuned":
            plot_dir = output_dir
        else:
            plot_dir = output_dir / f"xcebra_{phase}"
        
        # Generate scree plot
        plot_scree_analysis(
            Z_train=Z_train,
            Z_test=Z_test,
            out_dir=plot_dir,
            prefix=prefix,
            title_prefix=title,
            latent_dim=latent_dim
        )
    
    print(f"\n{'='*70}")
    print("✓ Scree plot generation complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

