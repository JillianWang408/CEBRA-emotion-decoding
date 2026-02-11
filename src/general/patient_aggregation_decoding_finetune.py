"""
Decoding for aggregated multi-patient training results.

- Loads encoder + heads from aggregated training folder
- Tests on a single patient's data (specified by patient_id)
- Follows same decoding pipeline as full_decoding_finetune.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
import mat73

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.calibration import CalibratedClassifierCV

import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Inline functions to avoid dependency on src.config
def load_embedding_TxD(emb_path: Path):
    """Load .pt embedding and return [T, D] numpy array."""
    embedding = torch.load(emb_path, map_location="cpu")
    if embedding.ndim == 3 and embedding.shape[0] == 1:
        embedding = embedding.squeeze(0).T
    elif embedding.ndim != 2:
        raise ValueError(f"Unexpected embedding shape {tuple(embedding.shape)}")
    return embedding.numpy()

def align_embedding_labels(Z, y_full):
    """Align embedding with labels, handling temporal offset.
    
    With valid padding (padding=0), PyTorch Conv1d reduces length from the END.
    So we trim labels from the END to match embedding length.
    """
    T_emb = Z.shape[0]
    T_full = len(y_full)
    offset = T_full - T_emb
    assert offset >= 0, f"Embedding longer than labels: {T_emb} > {T_full}"
    # Trim from END (last 'offset' timesteps are lost)
    y_aligned = y_full[:T_emb]  # Take first T_emb labels
    print(f"Aligned labels: {len(y_aligned)} / {T_full} (offset={offset}, trimmed from END)")
    return y_aligned, offset

# Import visualization utilities (these don't depend on config)
try:
    from src.utils_visualization import (
        collect_decoding_timecourse,
        save_decoding_timecourse,
        plot_decoding_timecourses,
    )
except ImportError:
    # Fallback if visualization utils not available
    def collect_decoding_timecourse(*args, **kwargs):
        return None
    def save_decoding_timecourse(*args, **kwargs):
        return None
    def plot_decoding_timecourses(*args, **kwargs):
        pass

# Patient configuration (same as in patient_aggregation.py)
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

EMOTION_MAP = {
    0: "No emotion",
    1: "Happy",
    2: "Sad",
    3: "Angry",
    4: "Fear",
    5: "Disgust",
    6: "Surprise",
    7: "Neutral",
    8: "Contempt",
    9: "Other",
}

# Data paths
DATA_SUBDIR = (
    "nrcRF_stim_resp_5_Nfold_pairs_msBW_1000_wASpec16_v16_DC5_1   2   5   6   7   8   9  10  11  12"
    "__wASpec16_v16_DC5_1   2   5   6   7   8   9  10  11  12_5"
)
NEURAL_FILENAME = "nrcRF_calc_Stim_StimNum_5_Nr_1_msBW_1000_movHeldOut_1.mat"
EMOTION_FILENAME = "nrcRF_calc_Resp_chan_1_movHeldOut_1.mat"

# Hyperparameter grids (same as full_decoding_finetune.py)
LOGREG_C_GRID = [0.01, 0.05, 0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 20.0]
HMM_STAY_GRID = [0.75, 0.80, 0.85, 0.90, 0.95]
HMM_EMO_TO_NONE_GRID = [0.02, 0.05, 0.10, 0.15, 0.20]
HMM_BETA_GRID = [0.4, 0.6, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1]
HMM_BETA_TEST_GRID = [0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1]
HEADS_EMOTION_SCALE_GRID = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
GLOBAL_NO_EMO = 0
ALL_ACTIVE_GLOBALS = list(range(1, 10))

# ---------------------------------------------------------------------
# Utilities (from full_decoding_finetune.py)
# ---------------------------------------------------------------------
def l2_normalize_rows(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (nrm + eps)

def compute_dwell_times(labels: np.ndarray) -> np.ndarray:
    if labels.size == 0:
        return np.array([])
    dwell, cur = [], 1
    for i in range(1, len(labels)):
        if labels[i] == labels[i - 1]:
            cur += 1
        else:
            dwell.append(cur)
            cur = 1
    dwell.append(cur)
    return np.array(dwell)

def estimate_transition_matrix_hub_spoke(
    n_classes: int,
    local_no_emo_idx: int,
    stay_p: float = 0.9,
    emo_to_none_p: float = 0.1,
    none_to_emo_p: float = 0.1,
) -> np.ndarray:
    A = np.full((n_classes, n_classes), 1e-12, dtype=np.float64)
    np.fill_diagonal(A, stay_p)
    for c in range(n_classes):
        if c != local_no_emo_idx:
            A[c, local_no_emo_idx] = emo_to_none_p
    for c in range(n_classes):
        if c != local_no_emo_idx:
            A[local_no_emo_idx, c] = none_to_emo_p
    A /= A.sum(axis=1, keepdims=True)
    return A

def log_clip(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(p, eps, 1.0))

def viterbi_decode_logprobs(log_emissions, log_A, log_pi):
    T, n_emissions = log_emissions.shape
    n_states = log_A.shape[0]
    if n_emissions != n_states:
        n_min = min(n_emissions, n_states)
        log_emissions = log_emissions[:, :n_min]
        log_A = log_A[:n_min, :n_min]
        log_pi = log_pi[:n_min]
        n_states = n_min
    dp = np.full((T, n_states), -np.inf)
    ptr = np.zeros((T, n_states), dtype=int)
    dp[0] = log_pi + log_emissions[0]
    for t in range(1, T):
        for j in range(n_states):
            scores = dp[t-1] + log_A[:, j]
            ptr[t, j] = np.argmax(scores)
            dp[t, j] = scores[ptr[t, j]] + log_emissions[t, j]
    states = np.zeros(T, dtype=int)
    states[-1] = np.argmax(dp[-1])
    for t in reversed(range(1, T)):
        states[t-1] = ptr[t, states[t]]
    return states

# ---------------------------------------------------------------------
# Heads loader + inference
# ---------------------------------------------------------------------
class GateHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)
    def forward(self, z: torch.Tensor):
        return self.fc(z).squeeze(-1)

class EmotionHead(nn.Module):
    def __init__(self, in_dim: int, n_active: int = 9):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_active)
    def forward(self, z: torch.Tensor):
        return self.fc(z)

def torch_load_safe(path, map_location="cpu", likely_weights=True):
    try:
        return torch.load(path, map_location=map_location, weights_only=True if likely_weights else False)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)

def load_heads(enc_dir: Path, D: int, device: torch.device):
    gate_path = enc_dir / "gate_head.pt"
    emo_path = enc_dir / "emo_head.pt"
    meta_path = enc_dir / "finetune_meta.pt"
    
    if not gate_path.exists() or not emo_path.exists():
        return None, None, 0.5
    
    gate_sd = torch_load_safe(gate_path, map_location="cpu", likely_weights=True)
    emo_sd = torch_load_safe(emo_path, map_location="cpu", likely_weights=True)
    meta = torch_load_safe(meta_path, map_location="cpu", likely_weights=False) if meta_path.exists() else {}
    
    gate = GateHead(D).to(device)
    emo = EmotionHead(D, n_active=len(ALL_ACTIVE_GLOBALS)).to(device)
    
    gate.load_state_dict(gate_sd.get("state_dict", gate_sd))
    emo.load_state_dict(emo_sd.get("state_dict", emo_sd))
    
    best_tau = float(meta.get("hyperparams", {}).get("best_tau", meta.get("best_results", {}).get("best_tau", 0.5)))
    print(f"[INFO] Loaded best_tau: {best_tau:.2f} from finetune_meta.pt")
    gate.eval(); emo.eval()
    return gate, emo, best_tau

@torch.no_grad()
def heads_predict_proba(X_TxD: np.ndarray, gate: GateHead, emo: EmotionHead, device, emotion_scale: float = 1.0) -> np.ndarray:
    X = torch.from_numpy(X_TxD).to(device=device, dtype=torch.float32)
    gl = gate(X)
    pa = torch.sigmoid(gl).unsqueeze(-1)
    el = emo(X)
    pe = F.softmax(el, dim=-1)
    
    if emotion_scale == 1.0:
        p_no = 1.0 - pa
        p_act = pa * pe
        P = torch.cat([p_no, p_act], dim=-1)
    else:
        p_act_scaled = (pa * emotion_scale) * pe
        p_no = 1.0 - (pa * emotion_scale)
        p_no = torch.clamp(p_no, min=1e-6)
        p_act_scaled = torch.clamp(p_act_scaled, min=1e-6)
        total = p_no + p_act_scaled.sum(dim=-1, keepdim=True)
        p_no = p_no / total
        p_act = p_act_scaled / total
        P = torch.cat([p_no, p_act], dim=-1)
    return P.cpu().numpy()

# ---------------------------------------------------------------------
# Load test patient data
# ---------------------------------------------------------------------
def load_test_patient_data(patient_id: int) -> tuple[np.ndarray, np.ndarray]:
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

# ---------------------------------------------------------------------
# Generate embeddings from encoder
# ---------------------------------------------------------------------
def generate_embeddings(encoder: nn.Module, neural_data: np.ndarray, device: torch.device) -> np.ndarray:
    """Generate embeddings from neural data using the encoder."""
    encoder.eval()
    with torch.no_grad():
        # Convert to tensor: (T, F) -> (1, F, T)
        X = torch.from_numpy(neural_data).to(device=device, dtype=torch.float32)
        X_bct = X.transpose(0, 1).unsqueeze(0)  # (1, F, T)
        
        # Get embeddings: (1, D, T') -> (T', D) -> (T', D)
        Z_bdt = encoder(X_bct)  # (1, D, T')
        Z = Z_bdt.squeeze(0).transpose(0, 1).cpu().numpy()  # (T', D)
        
    return Z

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Decode using aggregated multi-patient training results on a single test patient."
    )
    parser.add_argument(
        "--aggregated-result-dir",
        type=Path,
        required=True,
        help="Path to aggregated training result folder (e.g., output_patient_aggregation/238_239_272_301/)"
    )
    parser.add_argument(
        "--test-patient-id",
        type=int,
        required=True,
        help="Patient ID to use as test set (e.g., 1 for EC238, 2 for EC239, etc.)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <aggregated_result_dir>/decoding_test_<patient_code>)"
    )
    parser.add_argument(
        "--timecourse-cols",
        type=int,
        default=1,
        help="Number of columns for the decoding timecourse grid (default 1 column per row).",
    )
    
    args = parser.parse_args()
    
    # Validate and resolve paths
    enc_dir = Path(args.aggregated_result_dir).resolve()
    if not enc_dir.exists():
        raise FileNotFoundError(f"Aggregated result directory not found: {enc_dir}")
    
    encoder_path = enc_dir / "encoder_finetuned.pt"
    embedding_path = enc_dir / "embedding_finetuned.pt"
    
    # Find aggregated .npz file
    aggregated_npz_pattern = "aggregated_patient_data_*.npz"
    aggregated_npz_files = list(enc_dir.glob(aggregated_npz_pattern))
    if not aggregated_npz_files:
        # Try parent directory
        aggregated_npz_files = list(enc_dir.parent.glob(aggregated_npz_pattern))
    if not aggregated_npz_files:
        raise FileNotFoundError(f"Could not find aggregated .npz file in {enc_dir} or parent directory")
    aggregated_npz_path = aggregated_npz_files[0]
    print(f"[INFO] Found aggregated data: {aggregated_npz_path.name}")
    
    # Get test patient info
    if args.test_patient_id not in PATIENT_CONFIG:
        known = ", ".join(map(str, sorted(PATIENT_CONFIG.keys())))
        raise ValueError(f"Unknown test patient ID {args.test_patient_id}. Available: {known}")
    
    test_ec_code, test_patient_code = PATIENT_CONFIG[args.test_patient_id]
    print(f"[INFO] Test patient: {test_ec_code} (ID: {args.test_patient_id})")
    
    # Setup output directory
    if args.output_dir is None:
        out_dir = enc_dir / f"decoding_test_{test_patient_code}"
    else:
        out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # ========= Load aggregated training data =========
    print(f"[INFO] Loading aggregated training data...")
    data_train = np.load(aggregated_npz_path)
    neural_train_agg = data_train["neural"]  # (T_train, F)
    emotion_train_agg = data_train["emotion"]  # (T_train,)
    print(f"[INFO] Aggregated training data shape: neural {neural_train_agg.shape}, emotion {emotion_train_agg.shape}")
    
    # Get emotions present in training set
    emotions_train = np.unique(emotion_train_agg)
    emotions_train_sorted = sorted(emotions_train.tolist())
    emotion_names_train = [EMOTION_MAP.get(int(e), f"Unknown({int(e)})") for e in emotions_train_sorted]
    
    # ========= Load test patient data =========
    print(f"[INFO] Loading test patient data...")
    neural_test, emotion_test = load_test_patient_data(args.test_patient_id)
    print(f"[INFO] Test data shape: neural {neural_test.shape}, emotion {emotion_test.shape}")
    
    # Get emotions present in test set
    emotions_test = np.unique(emotion_test)
    emotions_test_sorted = sorted(emotions_test.tolist())
    emotion_names_test = [EMOTION_MAP.get(int(e), f"Unknown({int(e)})") for e in emotions_test_sorted]
    
    # ========= Save emotion information =========
    emotion_info_path = out_dir / "emotion_info.txt"
    with open(emotion_info_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("EMOTION PRESENCE INFORMATION\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Aggregated Training Set:\n")
        f.write(f"  Patient codes: {enc_dir.name}\n")
        f.write(f"  Total timesteps: {len(emotion_train_agg)}\n")
        f.write(f"  Emotions present: {emotions_train_sorted}\n")
        f.write(f"  Emotion names: {emotion_names_train}\n")
        f.write(f"  Counts:\n")
        for e in emotions_train_sorted:
            count = int((emotion_train_agg == e).sum())
            pct = 100.0 * count / len(emotion_train_agg)
            f.write(f"    {EMOTION_MAP.get(int(e), f'Unknown({int(e)})')}: {count} ({pct:.1f}%)\n")
        f.write(f"\n")
        f.write(f"Test Patient:\n")
        f.write(f"  Patient code: {test_ec_code} (ID: {args.test_patient_id})\n")
        f.write(f"  Total timesteps: {len(emotion_test)}\n")
        f.write(f"  Emotions present: {emotions_test_sorted}\n")
        f.write(f"  Emotion names: {emotion_names_test}\n")
        f.write(f"  Counts:\n")
        for e in emotions_test_sorted:
            count = int((emotion_test == e).sum())
            pct = 100.0 * count / len(emotion_test)
            f.write(f"    {EMOTION_MAP.get(int(e), f'Unknown({int(e)})')}: {count} ({pct:.1f}%)\n")
        f.write(f"\n")
        f.write(f"Emotions in training but NOT in test: {set(emotions_train_sorted) - set(emotions_test_sorted)}\n")
        f.write(f"Emotions in test but NOT in training: {set(emotions_test_sorted) - set(emotions_train_sorted)}\n")
        f.write(f"Emotions in both: {set(emotions_train_sorted) & set(emotions_test_sorted)}\n")
    print(f"[INFO] Saved emotion information → {emotion_info_path}")
    
    # ========= Load encoder =========
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder not found: {encoder_path}")
    
    encoder_data = torch_load_safe(encoder_path, map_location="cpu", likely_weights=False)
    latent_dim = encoder_data.get("latent_dim", 16)
    
    from cebra.models import init as init_model
    encoder = init_model(
        name="offset10-model",
        num_neurons=neural_train_agg.shape[1],  # Use training data shape
        num_units=256,
        num_output=latent_dim
    ).to(device)
    
    encoder.load_state_dict(encoder_data["state_dict"])
    encoder.eval()
    
    # ========= Generate embeddings for training data =========
    print(f"[INFO] Generating embeddings for aggregated training data...")
    X_train_agg = generate_embeddings(encoder, neural_train_agg, device)
    print(f"[INFO] Training embeddings shape: {X_train_agg.shape}")
    
    # Align training embeddings with labels (handle temporal offset)
    y_train_aligned, offset_train = align_embedding_labels(X_train_agg, emotion_train_agg)
    X_train_agg = X_train_agg[:len(y_train_aligned)]
    y_train_agg = np.squeeze(y_train_aligned).astype(int)
    print(f"[INFO] After alignment: training embeddings {X_train_agg.shape}, labels {y_train_agg.shape}")
    
    # ========= Generate embeddings for test patient (ENTIRE dataset) =========
    print(f"[INFO] Generating embeddings for test patient (entire dataset)...")
    X_test_all = generate_embeddings(encoder, neural_test, device)
    print(f"[INFO] Test embeddings shape: {X_test_all.shape}")
    
    # Align test embeddings with labels (handle temporal offset)
    y_test_aligned, offset_test = align_embedding_labels(X_test_all, emotion_test)
    X_test_all = X_test_all[:len(y_test_aligned)]
    y_test_all = np.squeeze(y_test_aligned).astype(int)
    print(f"[INFO] After alignment: test embeddings {X_test_all.shape}, labels {y_test_all.shape}")
    
    # L2-normalize
    X_train_agg = l2_normalize_rows(X_train_agg)
    X_test_all = l2_normalize_rows(X_test_all)
    
    # LOCAL mapping for kNN/LogReg (based on union of training and test emotions)
    present = np.unique(np.concatenate([y_train_agg, y_test_all]))
    n_local = len(present)
    g2l = {g: i for i, g in enumerate(present)}
    l2g = {i: g for g, i in g2l.items()}
    y_train = np.array([g2l[g] for g in y_train_agg], dtype=int)
    y_test = np.array([g2l[g] for g in y_test_all], dtype=int)
    
    # Train/cal split for logreg (from aggregated training data)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tr_idx, cal_idx = next(sss.split(X_train_agg, y_train))
    X_tr, y_tr = X_train_agg[tr_idx], y_train[tr_idx]
    X_cal, y_cal = X_train_agg[cal_idx], y_train[cal_idx]
    
    # Use entire test set (no split)
    X_test_g = X_test_all
    y_test_g = y_test_all
    
    # Linear R² sanity check (train on aggregated, test on test patient)
    coef, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
    y_pred_lin = X_test_g @ coef
    R2_behavior = r2_score(y_test, y_pred_lin)
    
    # HMM base matrix
    if GLOBAL_NO_EMO in present:
        local_no_emo = g2l[GLOBAL_NO_EMO]
        A_base = estimate_transition_matrix_hub_spoke(
            n_classes=n_local,
            local_no_emo_idx=local_no_emo,
            stay_p=0.9, emo_to_none_p=0.1, none_to_emo_p=0.1
        )
    else:
        print("[WARN] 'No emotion' (global 0) not present; using strong self-transitions.")
        A_base = np.full((n_local, n_local), 1e-12, dtype=np.float64)
        np.fill_diagonal(A_base, 0.95)
        A_base += (1 - np.eye(n_local)) * (0.05 / (n_local - 1))
        A_base /= A_base.sum(axis=1, keepdims=True)
        local_no_emo = -1
    
    log_A_base = log_clip(A_base)
    
    # Class prior
    pi = np.bincount(y_tr, minlength=n_local).astype(np.float64)
    pi = pi / pi.sum()
    log_pi = log_clip(pi)
    
    rows = []
    all_timecourse = []
    
    def local_to_global(arr_local: np.ndarray) -> np.ndarray:
        return np.array([l2g[int(a)] for a in arr_local], dtype=int)
    
    def evaluate_and_log(variant_tag: str, decoder_name: str, y_true_g: np.ndarray, y_pred_g: np.ndarray):
        acc = accuracy_score(y_true_g, y_pred_g)
        macro_f1 = f1_score(y_true_g, y_pred_g, average="macro")
        dwell_pred = compute_dwell_times(y_pred_g).mean() if y_pred_g.size else 0.0
        
        result = {
            "patient": test_patient_code,
            "decoder": decoder_name,
            "variant": variant_tag,
            "R2_behavior": round(R2_behavior, 4),
            "accuracy": round(acc, 4),
            "macroF1": round(macro_f1, 4),
            "mean_dwell_pred": round(dwell_pred, 2),
        }
        rows.append(result)
        print(f"[ok] [{decoder_name} | {variant_tag}] acc={acc:.3f}, R²={R2_behavior:.3f}, F1={macro_f1:.3f}, dwell={dwell_pred:.1f}")
        
        # Test indices: entire test set (offset accounts for embedding temporal offset)
        test_idx_local = np.arange(offset_test, offset_test + len(y_true_g))
        df_pair = collect_decoding_timecourse(
            pair_name=f"{decoder_name}_{variant_tag}",
            y_true=y_true_g,
            y_pred=y_pred_g,
            test_idx=test_idx_local,
        )
        return df_pair
    
    # =========================================================
    # LogReg (LOCAL): C tuning + calibration; Raw / HMM
    # =========================================================
    print(f"[LogReg] Searching C over {LOGREG_C_GRID}")
    best_lr, best_C, best_val = None, None, -1.0
    for C in LOGREG_C_GRID:
        lr = LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            C=C,
            max_iter=5000,
            n_jobs=-1,
        )
        lr.fit(X_tr, y_tr)
        y_cal_hat = lr.predict(X_cal)
        f1_cal = f1_score(y_cal, y_cal_hat, average="macro")
        print(f"  C={C:>4}: F1_cal={f1_cal:.4f}")
        if f1_cal > best_val:
            best_val, best_C, best_lr = f1_cal, C, lr
    
    cal = CalibratedClassifierCV(best_lr, method="isotonic", cv="prefit")
    cal.fit(X_cal, y_cal)
    P_test_lr_local = cal.predict_proba(X_test_g)
    y_lr_raw_local = np.argmax(P_test_lr_local, axis=1)
    y_lr_raw_global = local_to_global(y_lr_raw_local)
    all_timecourse.append(evaluate_and_log(f"Raw_C{best_C}", "logreg", y_test_g, y_lr_raw_global))
    
    # =========================================================
    # HMM Grid Search (calibration only to avoid leakage)
    # =========================================================
    print("\n[HMM GRID SEARCH] Optimizing transition parameters (calibration set only)...")
    
    if local_no_emo != -1:
        stay_p_values = HMM_STAY_GRID
        emo_to_none_values = HMM_EMO_TO_NONE_GRID
        beta_values = HMM_BETA_GRID
        
        best_stay, best_emo2none, best_beta_hmm = 0.9, 0.1, 0.9
        best_hmm_f1 = -1.0
        
        P_cal_lr = cal.predict_proba(X_cal)
        
        for stay_p in stay_p_values:
            for emo_to_none in emo_to_none_values:
                A_test = estimate_transition_matrix_hub_spoke(
                    n_classes=n_local,
                    local_no_emo_idx=local_no_emo,
                    stay_p=stay_p,
                    emo_to_none_p=emo_to_none,
                    none_to_emo_p=emo_to_none
                )
                log_A_test = log_clip(A_test)
                
                for beta in beta_values:
                    log_A = log_A_test.copy()
                    diag = np.eye(n_local, dtype=bool)
                    log_A[diag] += beta
                    A_boost = np.exp(log_A - log_A.max(axis=1, keepdims=True))
                    A_boost /= A_boost.sum(axis=1, keepdims=True)
                    log_A_boost = log_clip(A_boost)
                    
                    log_emiss_cal = log_clip(P_cal_lr)
                    y_cal_hmm = viterbi_decode_logprobs(log_emiss_cal, log_A_boost, log_pi)
                    f1_hmm = f1_score(y_cal, y_cal_hmm, average="macro")
                    print(
                        f"    stay={stay_p:.2f}, emo↔none={emo_to_none:.2f}, beta={beta:.2f} -> F1_cal={f1_hmm:.4f}"
                    )
                    
                    if f1_hmm > best_hmm_f1:
                        best_hmm_f1 = f1_hmm
                        best_stay, best_emo2none, best_beta_hmm = stay_p, emo_to_none, beta
        
        print(f"  [BEST HMM] stay_p={best_stay:.2f}, emo↔none={best_emo2none:.2f}, beta={best_beta_hmm:.2f}, F1_cal={best_hmm_f1:.4f}")
        
        A_base = estimate_transition_matrix_hub_spoke(
            n_classes=n_local,
            local_no_emo_idx=local_no_emo,
            stay_p=best_stay,
            emo_to_none_p=best_emo2none,
            none_to_emo_p=best_emo2none
        )
        log_A_base = log_clip(A_base)
    else:
        best_beta_hmm = 0.9
        print(f"  Using default beta={best_beta_hmm:.2f}")
    
    log_A = log_A_base.copy()
    diag = np.eye(n_local, dtype=bool)
    log_A[diag] += best_beta_hmm
    A_boost = np.exp(log_A - log_A.max(axis=1, keepdims=True))
    A_boost /= A_boost.sum(axis=1, keepdims=True)
    log_A_boost = log_clip(A_boost)
    
    log_emiss_lr = log_clip(P_test_lr_local)
    y_lr_hmm_local = viterbi_decode_logprobs(log_emiss_lr, log_A_boost, log_pi)
    y_lr_hmm_global = local_to_global(y_lr_hmm_local)
    all_timecourse.append(evaluate_and_log(f"HMM_b{best_beta_hmm:.2f}_C{best_C}", "logreg", y_test_g, y_lr_hmm_global))
    
    # =========================================================
    # Heads-based decoding (GLOBAL 10-class)
    # =========================================================
    gate, emo, best_tau = load_heads(enc_dir, D=X_test_all.shape[1], device=device)
    if gate is not None and emo is not None:
        X_test_torch = torch.from_numpy(X_test_g).to(device=device, dtype=torch.float32)
        gate_logits = gate(X_test_torch).cpu().detach().numpy()
        gate_probs = torch.sigmoid(torch.from_numpy(gate_logits)).numpy()
        
        # Grid search over emotion_scale on calibration set
        emotion_scales = HEADS_EMOTION_SCALE_GRID
        print(f"\n[GRID SEARCH] Testing emotion_scale values on calibration set...")
        best_scale, best_f1 = 1.0, 0.0
        
        for scale in emotion_scales:
            P_cal_heads_global = heads_predict_proba(X_cal, gate, emo, device, emotion_scale=scale)
            P_cal_heads_local = np.zeros((len(P_cal_heads_global), n_local))
            for local_idx, global_idx in l2g.items():
                if global_idx < P_cal_heads_global.shape[1]:
                    P_cal_heads_local[:, local_idx] = P_cal_heads_global[:, global_idx]
            row_sums_cal = P_cal_heads_local.sum(axis=1, keepdims=True)
            row_sums_cal[row_sums_cal == 0] = 1.0
            P_cal_heads_local = P_cal_heads_local / row_sums_cal
            
            y_pred_cal_local = np.argmax(P_cal_heads_local, axis=1)
            f1 = f1_score(y_cal, y_pred_cal_local, average="macro")
            print(f"  scale={scale:.1f}: F1_cal={f1:.4f}")
            if f1 > best_f1:
                best_f1, best_scale = f1, scale
        
        print(f"[BEST] emotion_scale={best_scale:.1f} with F1={best_f1:.4f}\n")
        
        P_test_heads = heads_predict_proba(X_test_g, gate, emo, device, emotion_scale=best_scale)
        
        # LogReg + Heads Features
        print("\n[LogReg+Features] Training LogReg with embeddings + gate + heads probabilities...")
        
        with torch.no_grad():
            X_train_torch = torch.from_numpy(X_train_agg).to(device=device, dtype=torch.float32)
            gate_train = torch.sigmoid(gate(X_train_torch)).cpu().detach().numpy().reshape(-1, 1)
            P_train_heads = heads_predict_proba(X_train_agg, gate, emo, device, emotion_scale=best_scale)
            gate_test = gate_probs.reshape(-1, 1)
            P_test_heads_feat = P_test_heads
        
        X_train_augmented = np.concatenate([X_train_agg, gate_train, P_train_heads], axis=1)
        X_test_augmented = np.concatenate([X_test_g, gate_test, P_test_heads_feat], axis=1)
        
        tr_idx_aug, cal_idx_aug = next(sss.split(X_train_augmented, y_train))
        X_tr_aug, y_tr_aug = X_train_augmented[tr_idx_aug], y_train[tr_idx_aug]
        X_cal_aug, y_cal_aug = X_train_augmented[cal_idx_aug], y_train[cal_idx_aug]
        
        best_lr_aug, best_C_aug, best_val_aug = None, None, -1.0
        for C in LOGREG_C_GRID:
            lr_aug = LogisticRegression(
                solver="lbfgs",
                class_weight="balanced",
                C=C,
                max_iter=5000,
                n_jobs=-1,
            )
            lr_aug.fit(X_tr_aug, y_tr_aug)
            y_cal_hat_aug = lr_aug.predict(X_cal_aug)
            f1_cal_aug = f1_score(y_cal_aug, y_cal_hat_aug, average="macro")
            print(f"  [LogReg+Feat] C={C:>4}: F1_cal={f1_cal_aug:.4f}")
            if f1_cal_aug > best_val_aug:
                best_val_aug, best_C_aug, best_lr_aug = f1_cal_aug, C, lr_aug
        
        cal_aug = CalibratedClassifierCV(best_lr_aug, method="isotonic", cv="prefit")
        cal_aug.fit(X_cal_aug, y_cal_aug)
        P_test_lr_aug_local = cal_aug.predict_proba(X_test_augmented)
        y_lr_aug_raw_local = np.argmax(P_test_lr_aug_local, axis=1)
        y_lr_aug_raw_global = local_to_global(y_lr_aug_raw_local)
        all_timecourse.append(evaluate_and_log(f"Raw_C{best_C_aug}", "logreg+features", y_test_g, y_lr_aug_raw_global))
        
        # HMM variant using best beta from calibration search
        log_A = log_A_base.copy()
        diag = np.eye(n_local, dtype=bool)
        log_A[diag] += best_beta_hmm
        A_boost = np.exp(log_A - log_A.max(axis=1, keepdims=True))
        A_boost /= A_boost.sum(axis=1, keepdims=True)
        log_A_boost = log_clip(A_boost)

        log_emiss_aug = log_clip(P_test_lr_aug_local)
        y_aug_hmm_local = viterbi_decode_logprobs(log_emiss_aug, log_A_boost, log_pi)
        y_aug_hmm_global = local_to_global(y_aug_hmm_local)
        all_timecourse.append(
            evaluate_and_log(f"HMM_b{best_beta_hmm:.2f}_C{best_C_aug}", "logreg+features", y_test_g, y_aug_hmm_global)
        )
        
        # Ensemble
        print("\n[Ensemble] Combining LogReg and Heads predictions...")
        P_test_lr_local = cal.predict_proba(X_test_g)
        lr_classes = cal.classes_
        n_classes_lr = len(lr_classes)
        
        P_test_heads_local = np.zeros((len(P_test_heads), n_classes_lr))
        for lr_idx, local_class in enumerate(lr_classes):
            global_class = l2g[local_class]
            if global_class < P_test_heads.shape[1]:
                P_test_heads_local[:, lr_idx] = P_test_heads[:, global_class]
        row_sums_test = P_test_heads_local.sum(axis=1, keepdims=True)
        row_sums_test[row_sums_test == 0] = 1.0
        P_test_heads_local = P_test_heads_local / row_sums_test
        
        P_cal_lr = cal.predict_proba(X_cal)
        P_cal_heads_global = heads_predict_proba(X_cal, gate, emo, device, emotion_scale=best_scale)
        P_cal_heads_local = np.zeros((len(P_cal_heads_global), n_classes_lr))
        for lr_idx, local_class in enumerate(lr_classes):
            global_class = l2g[local_class]
            if global_class < P_cal_heads_global.shape[1]:
                P_cal_heads_local[:, lr_idx] = P_cal_heads_global[:, global_class]
        row_sums_cal = P_cal_heads_local.sum(axis=1, keepdims=True)
        row_sums_cal[row_sums_cal == 0] = 1.0
        P_cal_heads_local = P_cal_heads_local / row_sums_cal
        
        best_weight, best_ensemble_f1 = 0.5, -1.0
        for w_lr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            w_heads = 1.0 - w_lr
            P_cal_ensemble = w_lr * P_cal_lr + w_heads * P_cal_heads_local
            y_cal_pred = np.argmax(P_cal_ensemble, axis=1)
            f1_ens = f1_score(y_cal, y_cal_pred, average="macro")
            print(f"    w_lr={w_lr:.1f}, w_heads={w_heads:.1f}: F1_cal={f1_ens:.4f}")
            if f1_ens > best_ensemble_f1:
                best_ensemble_f1, best_weight = f1_ens, w_lr
        
        print(f"  [BEST] w_lr={best_weight:.1f}, w_heads={1-best_weight:.1f}, F1={best_ensemble_f1:.4f}\n")
        
        w_lr_best = best_weight
        w_heads_best = 1.0 - best_weight
        P_test_ensemble_local = w_lr_best * P_test_lr_local + w_heads_best * P_test_heads_local
        
        y_ensemble_raw_local = np.argmax(P_test_ensemble_local, axis=1)
        y_ensemble_raw_global = local_to_global(y_ensemble_raw_local)
        all_timecourse.append(evaluate_and_log(f"Raw_w{w_lr_best:.1f}", "ensemble", y_test_g, y_ensemble_raw_global))
        
        log_A = log_A_base.copy()
        diag = np.eye(n_local, dtype=bool)
        log_A[diag] += best_beta_hmm
        A_boost = np.exp(log_A - log_A.max(axis=1, keepdims=True))
        A_boost /= A_boost.sum(axis=1, keepdims=True)
        log_A_boost = log_clip(A_boost)
        
        log_emiss = log_clip(P_test_ensemble_local)
        y_ensemble_hmm_local = viterbi_decode_logprobs(log_emiss, log_A_boost, log_pi)
        y_ensemble_hmm_global = local_to_global(y_ensemble_hmm_local)
        all_timecourse.append(
            evaluate_and_log(f"HMM_b{best_beta_hmm:.2f}_w{w_lr_best:.1f}", "ensemble", y_test_g, y_ensemble_hmm_global)
        )
    else:
        print("[INFO] Heads not found; skipping heads-based decoding.")
    
    # Save results
    df_summary = pd.DataFrame(rows)
    summary_path = out_dir / "decoding_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"[done] wrote decoding summary → {summary_path}")
    
    # Timecourse visualization
    df_all = save_decoding_timecourse(all_timecourse, out_dir / "decoding_timecourse.csv")
    if df_all is not None:
        plot_decoding_timecourses(
            csv_path=out_dir / "decoding_timecourse.csv",
            out_path=out_dir / "decoding_timecourse_grid.png",
            emotion_map=EMOTION_MAP,
            n_cols=max(1, args.timecourse_cols),
        )
    
    print(f"[FINISHED] Decoding complete for test patient {test_patient_code} using aggregated model from {enc_dir.name}")

if __name__ == "__main__":
    main()

