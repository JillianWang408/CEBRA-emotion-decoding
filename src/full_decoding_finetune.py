# full_decoding_phaseA_localHMM.py
# Phase A (kNN / LogReg) + Heads-based decoding.
# - Original LOCAL-class HMM stays unchanged for kNN/LogReg.
# - NEW: Load finetuned two-heads (gate + emotion) from models_finetune/,
#        produce GLOBAL (10-class) probabilities + Raw/EMA/HMM variants.
# Assumes EMOTION_MAP[0] == "No emotion" (class 0 globally).

import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier  # Commented out - kNN underperforms LogReg
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    # confusion_matrix,  # Commented out - not generating confusion matrices
    # ConfusionMatrixDisplay,
    r2_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.calibration import CalibratedClassifierCV

import matplotlib.pyplot as plt

from src.config import FULL_EMOTION_PATH, MODEL_DIR, PATIENT_CONFIG, EMOTION_MAP
from src.utils_decoding import load_embedding_TxD, _split_train_test
from src.utils_visualization import (
    collect_decoding_timecourse,
    save_decoding_timecourse,
    plot_decoding_timecourses,
)
# --- add near the imports ---
import torch.serialization as torch_serial

def torch_load_safe(path, map_location="cpu", likely_weights=True):
    """
    Try the PyTorch 2.6 default (weights_only=True) first.
    If it fails (meta files with numpy scalars), retry with weights_only=False.
    """
    try:
        return torch.load(path, map_location=map_location,
                          weights_only=True if likely_weights else False)
    except Exception:
        # fallback for trusted local checkpoints containing python/numpy objects
        return torch.load(path, map_location=map_location, weights_only=False)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
DECODERS = ["knn", "logreg", "heads"]  # <-- NEW family added
GLOBAL_NO_EMO = 0  # EMOTION_MAP[0] == "No emotion"
ALL_ACTIVE_GLOBALS = list(range(1, 10))  # 1..9 are emotions

# ---------------------------------------------------------------------
# Utilities
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

def ema_probs(P: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    Q = P.copy()
    for t in range(1, len(P)):
        Q[t] = alpha * P[t] + (1 - alpha) * Q[t - 1]
    return Q

def estimate_transition_matrix_hub_spoke(
    n_classes: int,
    local_no_emo_idx: int,
    stay_p: float = 0.9,
    emo_to_none_p: float = 0.1,
    none_to_emo_p: float = 0.1,
) -> np.ndarray:
    """
    Hub-and-spoke transitions:
      - High self-transition on all classes
      - emotion -> no_emo allowed
      - no_emo -> any emotion allowed
      - emotion -> other emotion discouraged (≈0)
    Rows normalized.
    """
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
    """
    Viterbi decoding in log-space, dimension-safe version.
    log_emissions: [T, N_states] log probabilities of emissions at each timestep
    log_A: [N_states, N_states] log transition probabilities
    log_pi: [N_states] log initial probabilities
    """
    T, n_emissions = log_emissions.shape
    n_states = log_A.shape[0]

    # Align dims if needed
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
# Heads loader + inference (NEW)
# ---------------------------------------------------------------------
class GateHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)
    def forward(self, z: torch.Tensor):
        return self.fc(z).squeeze(-1)  # (T,)

class EmotionHead(nn.Module):
    def __init__(self, in_dim: int, n_active: int = 9):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_active)
    def forward(self, z: torch.Tensor):
        return self.fc(z)  # (T, 9)

def load_heads(enc_dir: Path, D: int, device: torch.device):
    gate_path = enc_dir / "gate_head.pt"
    emo_path  = enc_dir / "emo_head.pt"
    meta_path = enc_dir / "finetune_meta.pt"

    # weights are pure tensors -> likely_weights=True
    gate_sd = torch_load_safe(gate_path, map_location="cpu", likely_weights=True)
    emo_sd  = torch_load_safe(emo_path,  map_location="cpu", likely_weights=True)

    # meta can contain numpy scalars -> likely_weights=False
    meta = torch_load_safe(meta_path, map_location="cpu", likely_weights=False)

    # build modules
    from src.full_encoding_finetune import GateHead, EmotionHead, ALL_ACTIVE_GLOBALS
    gate = GateHead(D).to(device)
    emo  = EmotionHead(D, n_active=len(ALL_ACTIVE_GLOBALS)).to(device)

    # accept both {"state_dict": ...} and raw state_dict
    gate.load_state_dict(gate_sd.get("state_dict", gate_sd))
    emo.load_state_dict(emo_sd.get("state_dict", emo_sd))

    best_tau = float(meta.get("hyperparams", {}).get("best_tau", 0.5))
    print(f"[INFO] Loaded best_tau: {best_tau:.2f} from finetune_meta.pt")
    gate.eval(); emo.eval()
    return gate, emo, best_tau

@torch.no_grad()
def heads_predict_proba(X_TxD: np.ndarray, gate: GateHead, emo: EmotionHead, device, emotion_scale: float = 1.0) -> np.ndarray:
    """
    Turn head logits into GLOBAL 10-class probabilities:
    P(no) = 1 - sigmoid(gate), P(g in 1..9) = sigmoid(gate) * softmax(emo)[g-1]
    
    Args:
        emotion_scale: Factor to scale emotion probabilities (default 1.0 = original formulation)
    """
    X = torch.from_numpy(X_TxD).to(device=device, dtype=torch.float32)  # (T,D)
    gl = gate(X)                              # (T,)
    pa = torch.sigmoid(gl).unsqueeze(-1)      # (T,1)
    el = emo(X)                                # (T,9)
    pe = F.softmax(el, dim=-1)                 # (T,9)

    if emotion_scale == 1.0:
        # Original formulation
        p_no = 1.0 - pa                            # (T,1)
        p_act = pa * pe                            # (T,9)
        P = torch.cat([p_no, p_act], dim=-1)       # (T,10)
    else:
        # Scale emotions to improve argmax behavior
        p_act_scaled = (pa * emotion_scale) * pe   # (T, 9)
        p_no = 1.0 - (pa * emotion_scale)          # (T, 1)
        
        # Clamp to ensure valid probabilities
        p_no = torch.clamp(p_no, min=1e-6)
        p_act_scaled = torch.clamp(p_act_scaled, min=1e-6)
        
        # Renormalize to sum to 1
        total = p_no + p_act_scaled.sum(dim=-1, keepdim=True)  # (T, 1)
        p_no = p_no / total
        p_act = p_act_scaled / total
        
        P = torch.cat([p_no, p_act], dim=-1)       # (T, 10)
    return P.cpu().numpy()

def apply_tau_rule(P_10: np.ndarray, tau: float) -> np.ndarray:
    """
    Deterministic τ-threshold decision:
    if sigmoid(gate) < τ -> class 0, else argmax over 1..9.
    We approximate gate prob as 1 - P[:,0].
    """
    pa = 1.0 - P_10[:, 0]          # (T,)
    y = np.zeros(P_10.shape[0], dtype=int)
    mask = pa >= tau
    if np.any(mask):
        y_em = P_10[mask, 1:].argmax(axis=1) + 1
        y[mask] = y_em
    return y

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    pid = int(float(os.environ["PATIENT_ID"]))
    _, patient_id = PATIENT_CONFIG[pid]

    enc_dir = MODEL_DIR / "models_finetune"            # finetuned encoder/heads
    out_dir = MODEL_DIR.parent / "full_decoding_finetune"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_timecourse = []

    print(f"[INFO] Starting decoding for patient {patient_id}")

    # Load embedding and labels
    emb_path = enc_dir / "embedding_finetuned.pt"
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing embedding file: {emb_path}")

    X_all = load_embedding_TxD(emb_path)  # [T, D]
    y_tensor = torch.load(FULL_EMOTION_PATH, map_location="cpu")
    if y_tensor.ndim > 1:
        y_tensor = y_tensor.squeeze(-1)
    y_all_global = y_tensor.long().contiguous().numpy().ravel()

    # Train/test split
    X_train_g, X_test_g, y_train_g, y_test_g, offset, split = _split_train_test(X_all, y_all_global)

    # L2-normalize
    X_train_g = l2_normalize_rows(X_train_g)
    X_test_g  = l2_normalize_rows(X_test_g)

    # LOCAL mapping for kNN/LogReg
    present = np.unique(np.concatenate([y_train_g, y_test_g]))
    n_local = len(present)
    g2l = {g: i for i, g in enumerate(present)}
    l2g = {i: g for g, i in g2l.items()}
    y_train = np.array([g2l[g] for g in y_train_g], dtype=int)
    y_test  = np.array([g2l[g] for g in y_test_g], dtype=int)

    # Train/cal split for logreg
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tr_idx, cal_idx = next(sss.split(X_train_g, y_train))
    X_tr, y_tr = X_train_g[tr_idx], y_train[tr_idx]
    X_cal, y_cal = X_train_g[cal_idx], y_train[cal_idx]

    # (Optional) linear R² sanity (on LOCAL labels for continuity)
    coef, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
    y_pred_lin = X_test_g @ coef
    R2_behavior = r2_score(y_test, y_pred_lin)

    # HMM base matrix (will be optimized later after LogReg is trained)
    if GLOBAL_NO_EMO in present:
        local_no_emo = g2l[GLOBAL_NO_EMO]
        A_base = estimate_transition_matrix_hub_spoke(
            n_classes=n_local,
            local_no_emo_idx=local_no_emo,
            stay_p=0.9, emo_to_none_p=0.1, none_to_emo_p=0.1
        )
    else:
        print("[WARN] 'No emotion' (global 0) not present for this patient; using strong self-transitions.")
        A_base = np.full((n_local, n_local), 1e-12, dtype=np.float64)
        np.fill_diagonal(A_base, 0.95)
        A_base += (1 - np.eye(n_local)) * (0.05 / (n_local - 1))
        A_base /= A_base.sum(axis=1, keepdims=True)
        local_no_emo = -1  # Not present
    
    log_A_base = log_clip(A_base)

    # Class prior (LOCAL)
    pi = np.bincount(y_tr, minlength=n_local).astype(np.float64)
    pi = pi / pi.sum()
    log_pi = log_clip(pi)

    rows = []

    def local_to_global(arr_local: np.ndarray) -> np.ndarray:
        return np.array([l2g[int(a)] for a in arr_local], dtype=int)

    def evaluate_and_log(variant_tag: str, decoder_name: str, y_true_g: np.ndarray,
                         y_pred_g: np.ndarray):
        acc = accuracy_score(y_true_g, y_pred_g)
        macro_f1 = f1_score(y_true_g, y_pred_g, average="macro")
        dwell_pred = compute_dwell_times(y_pred_g).mean() if y_pred_g.size else 0.0

        result = {
            "patient": patient_id,
            "decoder": decoder_name,
            "variant": variant_tag,
            "R2_behavior": f"{R2_behavior:.4f}",
            "accuracy": f"{acc:.4f}",
            "macroF1": f"{macro_f1:.4f}",
            "mean_dwell_pred": f"{dwell_pred:.2f}",
        }
        rows.append(result)
        print(f"[ok] [{decoder_name} | {variant_tag}] acc={acc:.3f}, R²={R2_behavior:.3f}, F1={macro_f1:.3f}, dwell={dwell_pred:.1f}")

        # Confusion matrix commented out for cleaner output
        # unique_labels = np.unique(np.concatenate([y_true_g, y_pred_g]))
        # cm = confusion_matrix(y_true_g, y_pred_g, labels=unique_labels)
        # label_names = [EMOTION_MAP[int(l)] for l in unique_labels]
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
        # disp.plot(xticks_rotation=90, cmap="Blues")
        # plt.title(f"{decoder_name} ({variant_tag}) CM (patient {patient_id})")
        # plt.tight_layout()
        # out_dir_cm = out_dir / f"cm_{decoder_name}_{variant_tag}_{patient_id}.png"
        # plt.savefig(out_dir_cm, dpi=150)
        # plt.close()

        # Timecourse row for multi-panel figure
        test_idx_local = np.arange(split, split + len(y_true_g))
        df_pair = collect_decoding_timecourse(
            pair_name=f"{decoder_name}_{variant_tag}",
            y_true=y_true_g,
            y_pred=y_pred_g,
            test_idx=test_idx_local,
        )
        return df_pair

    # =========================================================
    # kNN (LOCAL): Raw / EMA / HMM
    # COMMENTED OUT: kNN underperforms LogReg due to:
    #   1. Curse of dimensionality in high-dimensional embedding space
    #   2. No learned decision boundaries (just memorizes training data)
    #   3. Poor probability calibration (vote counts ≠ confidence)
    #   4. CEBRA embeddings are optimized for linear classification
    # =========================================================
    # knn = KNeighborsClassifier(n_neighbors=9, metric="cosine", weights="distance")
    # knn.fit(X_tr, y_tr)
    # P_test_knn_local = knn.predict_proba(X_test_g)  # [T_test, n_local]
    # y_knn_raw_local  = np.argmax(P_test_knn_local, axis=1)
    # y_knn_raw_global = local_to_global(y_knn_raw_local)
    all_timecourse = []
    # all_timecourse.append(evaluate_and_log("Raw", "knn", y_test_g, y_knn_raw_global))

    # for alpha in [0.3, 0.5, 0.7]:
    #     Q_local = ema_probs(P_test_knn_local, alpha=alpha)
    #     y_knn_ema_local = np.argmax(Q_local, axis=1)
    #     y_knn_ema_global = local_to_global(y_knn_ema_local)
    #     all_timecourse.append(evaluate_and_log(f"EMA_a{alpha}", "knn", y_test_g, y_knn_ema_global))

    # for beta in [0.2, 0.5, 0.8, 0.9, 1.0]:
    #     log_A = log_A_base.copy()
    #     diag = np.eye(n_local, dtype=bool)
    #     log_A[diag] += beta
    #     A_boost = np.exp(log_A - log_A.max(axis=1, keepdims=True))
    #     A_boost /= A_boost.sum(axis=1, keepdims=True)
    #     log_A_boost = log_clip(A_boost)

    #     log_emiss_knn = log_clip(P_test_knn_local)
    #     y_knn_hmm_local = viterbi_decode_logprobs(log_emiss_knn, log_A_boost, log_pi)
    #     y_knn_hmm_global = local_to_global(y_knn_hmm_local)
    #     all_timecourse.append(evaluate_and_log(f"HMM_b{beta}", "knn", y_test_g, y_knn_hmm_global))

    # =========================================================
    # LogReg (LOCAL): C tuning + calibration; Raw / EMA / HMM
    # =========================================================
    best_lr, best_C, best_val = None, None, -1.0
    for C in [0.1, 1.0, 3.0, 10.0]:
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
        if f1_cal > best_val:
            best_val, best_C, best_lr = f1_cal, C, lr

    cal = CalibratedClassifierCV(best_lr, method="isotonic", cv="prefit")
    cal.fit(X_cal, y_cal)
    P_test_lr_local = cal.predict_proba(X_test_g)
    y_lr_raw_local  = np.argmax(P_test_lr_local, axis=1)
    y_lr_raw_global = local_to_global(y_lr_raw_local)
    all_timecourse.append(evaluate_and_log(f"Raw_C{best_C}", "logreg", y_test_g, y_lr_raw_global))

    # =========================================================
    # Grid Search: HMM Transition Parameters + Beta (HIGH IMPACT)
    # =========================================================
    print("\n[HMM GRID SEARCH] Optimizing transition parameters on calibration set...")
    
    if local_no_emo != -1:  # If "no emotion" class is present
        # Grid search over HMM parameters
        stay_p_values = [0.85, 0.90, 0.95]
        emo_to_none_values = [0.05, 0.10, 0.15]
        beta_values = [0.8, 0.85, 0.9, 0.95, 1.0]
        
        best_stay, best_emo2none, best_beta_hmm = 0.9, 0.1, 0.9
        best_hmm_f1 = -1.0
        
        # Test on calibration set with LogReg probabilities
        P_cal_lr = cal.predict_proba(X_cal)
        
        for stay_p in stay_p_values:
            for emo_to_none in emo_to_none_values:
                A_test = estimate_transition_matrix_hub_spoke(
                    n_classes=n_local,
                    local_no_emo_idx=local_no_emo,
                    stay_p=stay_p,
                    emo_to_none_p=emo_to_none,
                    none_to_emo_p=emo_to_none  # Keep symmetric for simplicity
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
                    
                    if f1_hmm > best_hmm_f1:
                        best_hmm_f1 = f1_hmm
                        best_stay, best_emo2none, best_beta_hmm = stay_p, emo_to_none, beta
        
        print(f"  [BEST HMM] stay_p={best_stay:.2f}, emo↔none={best_emo2none:.2f}, beta={best_beta_hmm:.2f}, F1={best_hmm_f1:.4f}")
        
        # Rebuild A_base with optimized parameters
        A_base = estimate_transition_matrix_hub_spoke(
            n_classes=n_local,
            local_no_emo_idx=local_no_emo,
            stay_p=best_stay,
            emo_to_none_p=best_emo2none,
            none_to_emo_p=best_emo2none
        )
        log_A_base = log_clip(A_base)
    else:
        # No optimization if "no emotion" not present
        best_beta_hmm = 0.9
        print(f"  Using default beta={best_beta_hmm:.2f} (no emotion class not present)")
    
    # Fine-grained beta grid for testing
    beta_test_values = [0.8, 0.85, 0.9, 0.95, 1.0]

    # EMA commented out - HMM performs better (learns optimal state transitions vs simple smoothing)
    # for alpha in [0.3, 0.5, 0.7]:
    #     Q_local = ema_probs(P_test_lr_local, alpha=alpha)
    #     y_lr_ema_local = np.argmax(Q_local, axis=1)
    #     y_lr_ema_global = local_to_global(y_lr_ema_local)
    #     all_timecourse.append(evaluate_and_log(f"EMA_a{alpha}_C{best_C}", "logreg", y_test_g, y_lr_ema_global))

    # Use fine-grained beta grid with optimized HMM parameters
    beta_test_values = [0.8, 0.85, 0.9, 0.95, 1.0]
    for beta in beta_test_values:
        log_A = log_A_base.copy()
        diag = np.eye(n_local, dtype=bool)
        log_A[diag] += beta
        A_boost = np.exp(log_A - log_A.max(axis=1, keepdims=True))
        A_boost /= A_boost.sum(axis=1, keepdims=True)
        log_A_boost = log_clip(A_boost)

        log_emiss_lr = log_clip(P_test_lr_local)
        y_lr_hmm_local = viterbi_decode_logprobs(log_emiss_lr, log_A_boost, log_pi)
        y_lr_hmm_global = local_to_global(y_lr_hmm_local)
        
        # Add marker if this is the best beta from grid search
        marker = "*" if beta == best_beta_hmm else ""
        all_timecourse.append(evaluate_and_log(f"HMM_b{beta}{marker}_C{best_C}", "logreg", y_test_g, y_lr_hmm_global))

    # =========================================================
    # Heads-based decoding (GLOBAL 10-class): Raw / τ / EMA / HMM  (NEW)
    # =========================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gate, emo, best_tau = load_heads(enc_dir, D=X_all.shape[1], device=device)
    if gate is not None and emo is not None:
        # Heads produce 10-class global probabilities directly
        X_test_torch = torch.from_numpy(X_test_g).to(device=device, dtype=torch.float32)
        gate_logits = gate(X_test_torch).cpu().detach().numpy()
        gate_probs = torch.sigmoid(torch.from_numpy(gate_logits)).numpy()
        
        # Grid search over emotion_scale - use CALIBRATION set to avoid data leakage
        print(f"\n[GRID SEARCH] Testing emotion_scale values on calibration set:")
        emotion_scales = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5]
        best_scale, best_f1 = 1.0, 0.0
        
        for scale in emotion_scales:
            # Get heads probabilities on calibration set
            P_cal_heads_global = heads_predict_proba(X_cal, gate, emo, device, emotion_scale=scale)
            
            # Map from global 10-class to local n_local classes
            P_cal_heads_local = np.zeros((len(P_cal_heads_global), n_local))
            for local_idx, global_idx in l2g.items():
                if global_idx < P_cal_heads_global.shape[1]:
                    P_cal_heads_local[:, local_idx] = P_cal_heads_global[:, global_idx]
            row_sums_cal = P_cal_heads_local.sum(axis=1, keepdims=True)
            row_sums_cal[row_sums_cal == 0] = 1.0
            P_cal_heads_local = P_cal_heads_local / row_sums_cal
            
            y_pred_cal_local = np.argmax(P_cal_heads_local, axis=1)
            f1 = f1_score(y_cal, y_pred_cal_local, average="macro")
            y_pred_cal_global = local_to_global(y_pred_cal_local)
            n_emotions = (y_pred_cal_global > 0).sum()
            print(f"  - scale={scale:.1f}: F1={f1:.4f}, emotions={n_emotions}/{len(y_pred_cal_global)}")
            if f1 > best_f1:
                best_f1, best_scale = f1, scale
        
        print(f"[BEST] emotion_scale={best_scale:.1f} with F1={best_f1:.4f}\n")
        
        P_test_heads = heads_predict_proba(X_test_g, gate, emo, device, emotion_scale=best_scale)
        
        # Debug: Print probability statistics
        print(f"[DEBUG] Heads probability stats (scale={best_scale:.1f}):")
        print(f"  - Gate prob (P emotion present): mean={gate_probs.mean():.3f}, max={gate_probs.max():.3f}, min={gate_probs.min():.3f}")
        print(f"  - P(class 0): mean={P_test_heads[:, 0].mean():.3f}, max={P_test_heads[:, 0].max():.3f}")
        print(f"  - P(emotions 1-9): mean={P_test_heads[:, 1:].mean():.3f}, max={P_test_heads[:, 1:].max():.3f}")
        print(f"  - Raw predictions: class 0={(P_test_heads.argmax(axis=1) == 0).sum()}/{len(P_test_heads)}, emotions={(P_test_heads.argmax(axis=1) > 0).sum()}/{len(P_test_heads)}")
        
        # Standalone heads variants commented out - redundant with LogReg+Features and Ensemble
        # y_heads_raw = np.argmax(P_test_heads, axis=1)  # global 0..9
        # all_timecourse.append(evaluate_and_log(f"Raw_s{best_scale:.1f}", "heads", y_test_g, y_heads_raw))

        # # τ-threshold decision (if available)
        # # Note: Tau rule needs original probabilities (scale=1.0) to correctly extract gate probabilities
        # if isinstance(best_tau, (int, float)) and 0.0 <= best_tau <= 1.0:
        #     P_test_heads_original = heads_predict_proba(X_test_g, gate, emo, device, emotion_scale=1.0)
        #     y_heads_tau = apply_tau_rule(P_test_heads_original, float(best_tau))
        #     n_zero_tau = (y_heads_tau == 0).sum()
        #     n_emo_tau = (y_heads_tau > 0).sum()
        #     print(f"[Tau] tau={best_tau:.2f}: no_emotion={n_zero_tau}/{len(y_heads_tau)}, emotions={n_emo_tau}/{len(y_heads_tau)}")
        #     all_timecourse.append(evaluate_and_log(f"Tau_{best_tau:.2f}", "heads", y_test_g, y_heads_tau))

        # EMA commented out - HMM performs better (learns optimal state transitions vs simple smoothing)
        # for alpha in [0.3, 0.5, 0.7]:
        #     Q10 = ema_probs(P_test_heads, alpha=alpha)
        #     y_ema = np.argmax(Q10, axis=1)
        #     all_timecourse.append(evaluate_and_log(f"EMA_a{alpha}_s{best_scale:.1f}", "heads", y_test_g, y_ema))

        # HMM commented out - keeping only Raw and Tau as sanity checks
        # # HMM in GLOBAL space (10 classes, no_emo=0)
        # A10 = estimate_transition_matrix_hub_spoke(
        #     n_classes=10,
        #     local_no_emo_idx=0,
        #     stay_p=0.9, emo_to_none_p=0.1, none_to_emo_p=0.1
        # )
        # log_A10_base = log_clip(A10)
        # pi10 = np.ones(10, dtype=np.float64) / 10.0
        # log_pi10 = log_clip(pi10)
        # for beta in [0.2, 0.5, 0.8, 0.9, 1.0]:
        #     log_A = log_A10_base.copy()
        #     diag = np.eye(10, dtype=bool)
        #     log_A[diag] += beta
        #     A_boost = np.exp(log_A - log_A.max(axis=1, keepdims=True))
        #     A_boost /= A_boost.sum(axis=1, keepdims=True)
        #     log_A_boost = log_clip(A_boost)
        #     log_emiss = log_clip(P_test_heads)
        #     y_hmm = viterbi_decode_logprobs(log_emiss, log_A_boost, log_pi10)
        #     all_timecourse.append(evaluate_and_log(f"HMM_b{beta}_s{best_scale:.1f}", "heads", y_test_g, y_hmm))
        
        # =========================================================
        # LogReg + Heads Features: Use gate & heads as additional features
        # =========================================================
        print("\n[LogReg+Features] Training LogReg with embeddings + gate + heads probabilities...")
        
        # Get gate probabilities and heads probabilities for train/test/cal sets
        # Using scaled probabilities (best_scale) since they work better
        with torch.no_grad():
            # Train set
            X_train_torch = torch.from_numpy(X_train_g).to(device=device, dtype=torch.float32)
            gate_train = torch.sigmoid(gate(X_train_torch)).cpu().detach().numpy().reshape(-1, 1)
            P_train_heads = heads_predict_proba(X_train_g, gate, emo, device, emotion_scale=best_scale)
            
            # Test set (gate already computed earlier)
            gate_test = gate_probs.reshape(-1, 1)  # Reuse from earlier
            P_test_heads_feat = P_test_heads  # Reuse from earlier (already scaled)
        
        # Create augmented feature sets: [embeddings | gate_prob | heads_probs]
        X_train_augmented = np.concatenate([X_train_g, gate_train, P_train_heads], axis=1)
        X_test_augmented = np.concatenate([X_test_g, gate_test, P_test_heads_feat], axis=1)
        
        print(f"  Augmented features: {X_train_g.shape[1]} (embeddings) + 1 (gate) + {P_train_heads.shape[1]} (heads) = {X_train_augmented.shape[1]} total")
        
        # Split for calibration
        tr_idx_aug, cal_idx_aug = next(sss.split(X_train_augmented, y_train))
        X_tr_aug, y_tr_aug = X_train_augmented[tr_idx_aug], y_train[tr_idx_aug]
        X_cal_aug, y_cal_aug = X_train_augmented[cal_idx_aug], y_train[cal_idx_aug]
        
        # Grid search over C
        best_lr_aug, best_C_aug, best_val_aug = None, None, -1.0
        for C in [0.1, 1.0, 3.0, 10.0]:
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
            if f1_cal_aug > best_val_aug:
                best_val_aug, best_C_aug, best_lr_aug = f1_cal_aug, C, lr_aug
        
        # Calibrate and predict
        cal_aug = CalibratedClassifierCV(best_lr_aug, method="isotonic", cv="prefit")
        cal_aug.fit(X_cal_aug, y_cal_aug)
        P_test_aug_local = cal_aug.predict_proba(X_test_augmented)
        
        y_aug_raw_local = np.argmax(P_test_aug_local, axis=1)
        y_aug_raw_global = local_to_global(y_aug_raw_local)
        all_timecourse.append(evaluate_and_log(f"Raw_C{best_C_aug}", "logreg+features", y_test_g, y_aug_raw_global))
        
        # HMM on LogReg+Features (fine-grained beta grid)
        for beta in beta_test_values:
            log_A = log_A_base.copy()
            diag = np.eye(n_local, dtype=bool)
            log_A[diag] += beta
            A_boost = np.exp(log_A - log_A.max(axis=1, keepdims=True))
            A_boost /= A_boost.sum(axis=1, keepdims=True)
            log_A_boost = log_clip(A_boost)

            log_emiss_aug = log_clip(P_test_aug_local)
            y_aug_hmm_local = viterbi_decode_logprobs(log_emiss_aug, log_A_boost, log_pi)
            y_aug_hmm_global = local_to_global(y_aug_hmm_local)
            
            marker = "*" if beta == best_beta_hmm else ""
            all_timecourse.append(evaluate_and_log(f"HMM_b{beta}{marker}_C{best_C_aug}", "logreg+features", y_test_g, y_aug_hmm_global))
        
        # =========================================================
        # Ensemble: Heads + Logistic Regression (Probability Averaging)
        # Combine probability distributions via weighted average
        # =========================================================
        print("\n[Ensemble] Combining LogReg and Heads via probability averaging...")
        
        print(f"  Using emotion_scale={best_scale:.1f} for heads probabilities (same as heads Raw)")
        
        # Map heads probabilities from global 10-class to local space
        # Use best_scale since scaled probabilities work better for predictions
        P_test_heads_for_ensemble = heads_predict_proba(X_test_g, gate, emo, device, emotion_scale=best_scale)
        
        # Create mapping from global 10 classes to local classes
        P_test_heads_local = np.zeros((len(P_test_heads_for_ensemble), n_local))
        for local_idx, global_idx in l2g.items():
            if global_idx < P_test_heads_for_ensemble.shape[1]:
                P_test_heads_local[:, local_idx] = P_test_heads_for_ensemble[:, global_idx]
        
        # Normalize to ensure valid probabilities
        row_sums = P_test_heads_local.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        P_test_heads_local = P_test_heads_local / row_sums
        
        # Grid search over ensemble weights on calibration set
        print("  Grid searching ensemble weights...")
        best_weight, best_ensemble_f1 = 0.5, -1.0
        
        # Get logreg predictions on calibration set
        P_cal_lr = cal.predict_proba(X_cal)
        n_classes_lr = P_cal_lr.shape[1]  # Classes LogReg actually knows
        
        # Get heads predictions on calibration set
        P_cal_heads_global = heads_predict_proba(X_cal, gate, emo, device, emotion_scale=best_scale)
        
        # Map heads to same class space as LogReg (might be smaller than n_local)
        # Get the classes LogReg knows about
        lr_classes = cal.classes_  # Classes LogReg can predict
        P_cal_heads_local = np.zeros((len(P_cal_heads_global), n_classes_lr))
        
        for lr_idx, local_class in enumerate(lr_classes):
            global_class = l2g[local_class]  # Convert local to global
            if global_class < P_cal_heads_global.shape[1]:
                P_cal_heads_local[:, lr_idx] = P_cal_heads_global[:, global_class]
        
        row_sums_cal = P_cal_heads_local.sum(axis=1, keepdims=True)
        row_sums_cal[row_sums_cal == 0] = 1.0
        P_cal_heads_local = P_cal_heads_local / row_sums_cal
        
        for w_lr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            w_heads = 1.0 - w_lr
            P_cal_ensemble = w_lr * P_cal_lr + w_heads * P_cal_heads_local
            y_cal_pred = np.argmax(P_cal_ensemble, axis=1)
            f1_ens = f1_score(y_cal, y_cal_pred, average="macro")
            print(f"    w_lr={w_lr:.1f}, w_heads={w_heads:.1f}: F1={f1_ens:.4f}")
            if f1_ens > best_ensemble_f1:
                best_ensemble_f1, best_weight = f1_ens, w_lr
        
        print(f"  [BEST] w_lr={best_weight:.1f}, w_heads={1-best_weight:.1f}, F1={best_ensemble_f1:.4f}\n")
        
        # Apply best weights to test set
        # Need to align P_test_heads_local to same classes as P_test_lr_local
        P_test_heads_aligned = np.zeros((len(P_test_heads_for_ensemble), n_classes_lr))
        for lr_idx, local_class in enumerate(lr_classes):
            global_class = l2g[local_class]
            if global_class < P_test_heads_for_ensemble.shape[1]:
                P_test_heads_aligned[:, lr_idx] = P_test_heads_for_ensemble[:, global_class]
        
        row_sums_test = P_test_heads_aligned.sum(axis=1, keepdims=True)
        row_sums_test[row_sums_test == 0] = 1.0
        P_test_heads_aligned = P_test_heads_aligned / row_sums_test
        
        w_lr_best = best_weight
        w_heads_best = 1.0 - best_weight
        P_test_ensemble_local = w_lr_best * P_test_lr_local + w_heads_best * P_test_heads_aligned
        
        y_ensemble_raw_local = np.argmax(P_test_ensemble_local, axis=1)
        y_ensemble_raw_global = local_to_global(y_ensemble_raw_local)
        all_timecourse.append(evaluate_and_log(f"Raw_w{w_lr_best:.1f}", "ensemble", y_test_g, y_ensemble_raw_global))
        
        # EMA commented out - HMM performs better (learns optimal state transitions vs simple smoothing)
        # for alpha in [0.3, 0.5, 0.7]:
        #     Q_local = ema_probs(P_test_ensemble_local, alpha=alpha)
        #     y_ensemble_ema_local = np.argmax(Q_local, axis=1)
        #     y_ensemble_ema_global = local_to_global(y_ensemble_ema_local)
        #     all_timecourse.append(evaluate_and_log(f"EMA_a{alpha}_w{w_lr_best:.1f}", "ensemble", y_test_g, y_ensemble_ema_global))
        
        # HMM on ensemble probabilities (fine-grained beta grid)
        for beta in beta_test_values:
            log_A = log_A_base.copy()
            diag = np.eye(n_local, dtype=bool)
            log_A[diag] += beta
            A_boost = np.exp(log_A - log_A.max(axis=1, keepdims=True))
            A_boost /= A_boost.sum(axis=1, keepdims=True)
            log_A_boost = log_clip(A_boost)

            log_emiss = log_clip(P_test_ensemble_local)
            y_ensemble_hmm_local = viterbi_decode_logprobs(log_emiss, log_A_boost, log_pi)
            y_ensemble_hmm_global = local_to_global(y_ensemble_hmm_local)
            
            marker = "*" if beta == best_beta_hmm else ""
            all_timecourse.append(evaluate_and_log(f"HMM_b{beta}{marker}_w{w_lr_best:.1f}", "ensemble", y_test_g, y_ensemble_hmm_global))
    else:
        print("[INFO] Heads not found; skipping heads-based decoding.")

    # ------------------------------
    # Save decoding summary and plots
    # ------------------------------
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
            n_cols=4,
        )

    print(f"[FINISHED] Full decoding complete for patient {patient_id}.")

if __name__ == "__main__":
    main()
