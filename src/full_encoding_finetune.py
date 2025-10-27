# src/finetune_from_pretrained.py
# ------------------------------------------------------------
# Finetune ONLY (Phase B). No Phase A training.
# Loads the pretrained encoder from MODEL_DIR/<pid>/xcebra_supervised/,
# adds two-stage heads, finetunes with CE+TC+L2SP, and saves
# to <patient_id>/models_finetune/.
#
# NEW:
# - tqdm batch bars
# - per-epoch summaries (loss breakdown + best-τ macro-F1)
# - training curves plot (loss components + val macro-F1)
# ------------------------------------------------------------

import os
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import mat73, scipy.io

from cebra.models import init as init_model

from src.config import (
    MODEL_DIR, NEURAL_PATH, EMOTION_PATH,
    FULL_NEURAL_PATH, FULL_EMOTION_PATH, PATIENT_CONFIG
)

# NEW: for progress bars & plots
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# ============ A: fixed global class set (1..9) ============
ALL_ACTIVE_GLOBALS: List[int] = list(range(1, 10))  # emotions 1..9; 0 is "No emotion"

# -----------------------
# utils
# -----------------------
def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)

def l2_normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    nrm = x.norm(dim=-1, keepdim=True)
    return x / (nrm + eps)

# -----------------------
# Two-stage heads
# -----------------------
class GateHead(nn.Module):
    """Binary gate: NoEmotion (0) vs AnyEmotion (1)."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)
    def forward(self, z):
        return self.fc(z).squeeze(-1)  # (...,)

class EmotionHead(nn.Module):
    """Classifier over ACTIVE classes (excludes NoEmotion)."""
    def __init__(self, in_dim: int, n_active: int):
        super().__init__()
        self.n_active = int(n_active)
        self.fc = nn.Linear(in_dim, self.n_active) if self.n_active > 0 else None
    def forward(self, z):
        if self.n_active == 0:
            return z.new_zeros(z.shape[:-1] + (0,))
        return self.fc(z)

# -----------------------
# Sequence dataset
# -----------------------
class SeqDataset(Dataset):
    """
    Creates short sequences so temporal consistency sees neighbors.
    Returns: (x_seq, y_global, y_gate, y_emo, meta)
    """
    def __init__(self, neural: torch.Tensor, labels_global: torch.Tensor,
                 seq_len: int, stride: int, no_emotion_global: int = 0):
        self.X = neural
        self.yg = labels_global.long()
        self.seq_len, self.stride = int(seq_len), int(stride)
        self.no_global = int(no_emotion_global)

        # fixed global class set (1..9) for the emotion head everywhere
        self.active = np.array(ALL_ACTIVE_GLOBALS, dtype=int)
        self.g2a = {g: i for i, g in enumerate(self.active)}  # 1..9 -> 0..8

        # valid start indices (inclusive)
        self.idxs = list(range(0, len(self.X) - self.seq_len + 1, self.stride))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        s = self.idxs[idx]; e = s + self.seq_len
        x = self.X[s:e]               # (Tctx, F)
        yg = self.yg[s:e]             # (Tctx,)
        y_gate = (yg != self.no_global).long()  # (Tctx,)
        y_emo = torch.full_like(yg, -100)       # ignore index for CE

        mask = (y_gate == 1)
        if mask.any() and len(self.active) > 0:
            g_list = yg[mask].cpu().tolist()
            # map each global id to active-local id (1..9 -> 0..8)
            y_emo[mask] = torch.tensor([self.g2a.get(g, 0) for g in g_list], dtype=torch.long)
        meta = {"active": self.active}  # numpy array of global ids (1..9)
        return x, yg, y_gate, y_emo, meta

# -----------------------
# Losses
# -----------------------
def kl_divergence(p_log_softmax, q_log_softmax):
    p = p_log_softmax.exp()
    return (p * (p_log_softmax - q_log_softmax)).sum(dim=-1)

def tc_gate(logits: torch.Tensor) -> torch.Tensor:
    """Temporal consistency for gate (Bernoulli) via prob MSE."""
    if logits.size(1) < 2:
        return logits.new_zeros(())
    p = torch.sigmoid(logits)
    return ((p[:, 1:] - p[:, :-1])**2).mean()

def tc_emo(logits: torch.Tensor) -> torch.Tensor:
    """Temporal consistency for emotion (categorical) via KL."""
    if logits.ndim != 3 or logits.size(2) == 0 or logits.size(1) < 2:
        return logits.new_zeros(())
    logp = F.log_softmax(logits, dim=-1)
    return kl_divergence(logp[:, 1:], logp[:, :-1]).mean()

def l2sp_loss(model: nn.Module, anchor: Dict[str, torch.Tensor], mu: float) -> torch.Tensor:
    loss = 0.0
    for n, p in model.named_parameters():
        if p.requires_grad and n in anchor:
            loss = loss + (p - anchor[n]).pow(2).sum()
    return loss * mu

# -----------------------
# Train loop + logging/plots
# -----------------------
def finetune_two_stage(
    encoder: nn.Module,
    neural_train: torch.Tensor, labels_train_global: torch.Tensor,
    neural_val: torch.Tensor,   labels_val_global: torch.Tensor,
    emb_dim: int, device: torch.device,
    no_emotion_global: int = 0,
    seq_len: int = 64, stride: int = 8,
    lr_head: float = 3e-4, lr_enc: float = 2e-5, weight_decay: float = 1e-4,
    lambda_tc: float = 0.1, mu_l2sp: float = 1e-5,
    batch_size: int = 16, max_epochs: int = 20, patience: int = 5,
):
    # ---- heads ----
    n_active = len(ALL_ACTIVE_GLOBALS)
    gate_head = GateHead(emb_dim).to(device)
    emo_head  = EmotionHead(emb_dim, n_active).to(device)

    # ---- anchor (L2-SP) ----
    anchor = {n: p.detach().clone() for n, p in encoder.named_parameters() if p.requires_grad}

    # ---- class weights (gate) ----
    y_gate_train = (labels_train_global != no_emotion_global).long()
    cnt0 = int((y_gate_train == 0).sum().item()) + 1
    cnt1 = int((y_gate_train == 1).sum().item()) + 1
    denom = float(cnt0 + cnt1)
    gate_w = torch.tensor([cnt1/denom, cnt0/denom], device=device, dtype=torch.float32)

    bce_gate = nn.CrossEntropyLoss(weight=gate_w, label_smoothing=0.05)

    # ---- class weights (emo) over fixed 1..9 ----
    emo_counts = [int((labels_train_global == g).sum().item()) + 1 for g in ALL_ACTIVE_GLOBALS]
    emo_counts = np.array(emo_counts, dtype=float)
    emo_w = (emo_counts.sum() - emo_counts) / emo_counts.sum()
    emo_w = torch.tensor(emo_w, device=device, dtype=torch.float32)
    ce_emo = nn.CrossEntropyLoss(weight=emo_w)

    # ---- data loaders ----
    train_ds = SeqDataset(neural_train, labels_train_global, seq_len, stride, no_emotion_global)
    val_ds   = SeqDataset(neural_val,   labels_val_global,   seq_len, stride, no_emotion_global)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    # ---- optimizer ----
    opt = torch.optim.AdamW([
        {"params": encoder.parameters(),   "lr": lr_enc},
        {"params": gate_head.parameters(), "lr": lr_head},
        {"params": emo_head.parameters(),  "lr": lr_head},
    ], weight_decay=weight_decay)

    first_conv = next(m for m in encoder.modules() if isinstance(m, nn.Conv1d))

    # ---- histories for plotting ----
    hist = {
        "epoch": [],
        "loss_total": [],
        "loss_gate": [],
        "loss_emo": [],
        "loss_tc": [],
        "loss_sp": [],
        "val_f1": [],
        "best_tau": [],
    }

    best_f1, best_state, best_tau, bad = -1.0, None, 0.5, 0

    for epoch in range(1, max_epochs+1):
        encoder.train(); gate_head.train(); emo_head.train()

        sum_total = sum_gate = sum_emo = sum_tc = sum_sp = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}", leave=False)
        for x, yg, y_gate, y_emo, meta in pbar:
            x = x.to(device)            # (B, Tctx, F)
            yg = yg.to(device)          # (B, Tctx)
            y_gate = y_gate.to(device)  # (B, Tctx)
            y_emo  = y_emo.to(device)   # (B, Tctx)

            x_bct = x.permute(0, 2, 1).contiguous()  # (B, F, Tctx)
            assert x_bct.shape[1] == first_conv.in_channels, \
                f"Encoder expects {first_conv.in_channels} channels, got {x_bct.shape[1]}"

            z_bdt = encoder(x_bct)                      # (B, D, T_enc)
            z = z_bdt.permute(0, 2, 1).contiguous()     # (B, T_enc, D)
            z = l2_normalize_rows(z.reshape(-1, z.size(-1))).reshape(z.size())

            # align labels to T_enc (center-crop)
            T_enc = z.size(1)
            if T_enc != y_gate.size(1):
                shift = (y_gate.size(1) - T_enc) // 2
                y_gate = y_gate[:, shift:shift+T_enc]
                y_emo  = y_emo[:,  shift:shift+T_enc]
                yg     = yg[:,     shift:shift+T_enc]

            # heads
            gate_logits = gate_head(z)                                  # (B, T_enc)
            gate_two    = torch.stack([-gate_logits, gate_logits], -1)  # (B, T_enc, 2)
            emo_logits  = emo_head(z)                                   # (B, T_enc, 9)

            # losses
            loss_gate = bce_gate(gate_two.reshape(-1, 2), y_gate.reshape(-1))

            y_emo_flat = y_emo.reshape(-1)
            mask = (y_emo_flat != -100)
            if mask.any():
                loss_emo = ce_emo(emo_logits.reshape(-1, emo_logits.size(-1))[mask],
                                  y_emo_flat[mask])
            else:
                loss_emo = torch.zeros((), device=z.device)

            loss_tc = lambda_tc * (tc_gate(gate_logits) + tc_emo(emo_logits))
            loss_sp = l2sp_loss(encoder, anchor, mu=mu_l2sp)
            loss = loss_gate + loss_emo + loss_tc + loss_sp

            opt.zero_grad()
            loss.backward()
            opt.step()

            # update running means for the bar
            n_batches += 1
            sum_total += loss.item()
            sum_gate  += loss_gate.item()
            sum_emo   += loss_emo.item()
            sum_tc    += float(loss_tc.item()) if isinstance(loss_tc, torch.Tensor) else float(loss_tc)
            sum_sp    += float(loss_sp.item()) if isinstance(loss_sp, torch.Tensor) else float(loss_sp)

            pbar.set_postfix({
                "loss": f"{sum_total/n_batches:.4f}",
                "gate": f"{sum_gate/n_batches:.4f}",
                "emo":  f"{sum_emo/n_batches:.4f}",
                "tc":   f"{sum_tc/n_batches:.4f}",
                "l2sp": f"{sum_sp/n_batches:.4f}",
            })

        # ---- Validation with gate-threshold tuning (τ grid) ----
        encoder.eval(); gate_head.eval(); emo_head.eval()
        with torch.no_grad():
            all_yg, all_pa, all_pe = [], [], []
            for x, yg, *_ in val_loader:
                x = x.to(device)
                x_bct = x.permute(0, 2, 1).contiguous()  # (B, F, Tctx)
                z_bdt = encoder(x_bct)                   # (B, D, T_enc)
                z = z_bdt.permute(0, 2, 1).contiguous()  # (B, T_enc, D)
                z = l2_normalize_rows(z.reshape(-1, z.size(-1))).reshape(z.size())

                # align labels
                T_enc = z.size(1)
                if T_enc != yg.size(1):
                    shift = (yg.size(1) - T_enc) // 2
                    yg = yg[:, shift:shift+T_enc]         # (B, T_enc)

                gl = gate_head(z)                         # (B, T_enc)
                pa = torch.sigmoid(gl)                    # (B, T_enc)
                el = emo_head(z)                          # (B, T_enc, 9)
                pe = F.softmax(el, dim=-1)                # (B, T_enc, 9)

                all_yg.append(yg.cpu())
                all_pa.append(pa.cpu())
                all_pe.append(pe.cpu())

            YG = torch.cat(all_yg, dim=0)  # (N,T)
            PA = torch.cat(all_pa, dim=0)  # (N,T)
            PE = torch.cat(all_pe, dim=0)  # (N,T,9)

            from sklearn.metrics import f1_score
            taus = torch.tensor([0.1, 0.3, 0.4, 0.45, 0.5, 0.55])
            best_f1_tau, best_tau_epoch = -1.0, 0.5
            for tau in taus:
                gate_on = (PA >= tau.item()).float().unsqueeze(-1)  # (N,T,1)
                p_no    = (1.0 - gate_on)                           # (N,T,1)
                p_act   = gate_on * PE                              # (N,T,9)
                P_all   = torch.cat([p_no, p_act], dim=-1)          # (N,T,10)

                arg   = P_all.argmax(dim=-1)                        # (N,T)
                map_ids = torch.tensor([0] + ALL_ACTIVE_GLOBALS, dtype=torch.long)
                Ypred = map_ids[arg]                                # (N,T)

                f1m = f1_score(YG.reshape(-1).numpy(), Ypred.reshape(-1).numpy(), average="macro")
                if f1m > best_f1_tau:
                    best_f1_tau, best_tau_epoch = f1m, float(tau.item())

            print(f"[Epoch {epoch} - Tau Selection] Best tau: {best_tau_epoch:.2f} with F1: {best_f1_tau:.4f}")

        # ---- Logging & early stop bookkeeping ----
        avg_total = sum_total / max(n_batches, 1)
        avg_gate  = sum_gate  / max(n_batches, 1)
        avg_emo   = sum_emo   / max(n_batches, 1)
        avg_tc    = sum_tc    / max(n_batches, 1)
        avg_sp    = sum_sp    / max(n_batches, 1)

        hist["epoch"].append(epoch)
        hist["loss_total"].append(avg_total)
        hist["loss_gate"].append(avg_gate)
        hist["loss_emo"].append(avg_emo)
        hist["loss_tc"].append(avg_tc)
        hist["loss_sp"].append(avg_sp)
        hist["val_f1"].append(best_f1_tau)
        hist["best_tau"].append(best_tau_epoch)

        print(
            f"[Epoch {epoch:02d}] "
            f"loss={avg_total:.4f} | gate={avg_gate:.4f} emo={avg_emo:.4f} tc={avg_tc:.4f} l2sp={avg_sp:.4f} "
            f"| valF1={best_f1_tau:.3f} (τ={best_tau_epoch:.2f})"
        )

        improved = best_f1_tau > best_f1
        if improved:
            best_f1 = best_f1_tau
            best_tau = best_tau_epoch
            print(f"[New Best] Epoch {epoch} achieved best F1: {best_f1_tau:.4f} with tau: {best_tau_epoch:.2f}")
            bad = 0
            best_state = {
                "enc":  {k: v.detach().cpu().clone() for k,v in encoder.state_dict().items()},
                "gate": {k: v.detach().cpu().clone() for k,v in gate_head.state_dict().items()},
                "emo":  {k: v.detach().cpu().clone() for k,v in emo_head.state_dict().items()},
            }
        else:
            bad += 1
            if bad >= patience:
                print(f"[EarlyStop] no val F1 improvement for {patience} epochs.")
                break

    if best_state is not None:
        encoder.load_state_dict(best_state["enc"])
        gate_head.load_state_dict(best_state["gate"])
        emo_head.load_state_dict(best_state["emo"])

    return encoder, gate_head, emo_head, {
        "val_best_f1": best_f1,
        "best_tau": best_tau,
        "history": hist,         # <- NEW: return full history for plotting
    }

# -----------------------
# Main: LOAD from disk, finetune, SAVE, and PLOT curves
# -----------------------
def main():
    # ---- setup ----
    pid = int(float(os.environ["PATIENT_ID"]))
    _, patient_id = PATIENT_CONFIG[pid]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- load data ----
    neural_data    = mat73.loadmat(NEURAL_PATH)['stim'].T   # (T,F)
    emotion_labels = scipy.io.loadmat(EMOTION_PATH)['resp'].flatten()  # (T,)
    neural_tensor = to_tensor(neural_data)
    label_tensor  = to_tensor(emotion_labels).reshape(-1)
    print(f"[Data] Loaded neural data {tuple(neural_tensor.shape)} and labels {tuple(label_tensor.shape)}")

    # optional artifacts
    torch.save(neural_tensor, FULL_NEURAL_PATH)
    torch.save(label_tensor.view(-1,1), FULL_EMOTION_PATH)

    # ---- contiguous split (80/20) ----
    T = neural_tensor.shape[0]
    split = int(0.8 * T)
    train_idx = torch.arange(split)
    val_idx   = torch.arange(split, T)
    Xtr, Xval = neural_tensor[train_idx], neural_tensor[val_idx]
    Ytr, Yval = label_tensor[train_idx].long(), label_tensor[val_idx].long()

    # ---- init encoder & LOAD Phase-A supervised weights ----
    latent_dim = 16
    encoder = init_model(
        name="offset10-model",
        num_neurons=neural_tensor.shape[1],
        num_units=256,
        num_output=latent_dim
    ).to(device)

    first_conv = next(m for m in encoder.modules() if isinstance(m, nn.Conv1d))
    print("[Sanity] Encoder expects in_channels =", first_conv.in_channels)

    sup_dir = MODEL_DIR / "xcebra_supervised"
    ckpt_paths = [
        sup_dir / "model_weights.pt",
        sup_dir / "model.pt",
        sup_dir / "checkpoint.pt",
    ]
    state_loaded = False
    for pth in ckpt_paths:
        if pth.exists():
            ckpt = torch.load(pth, map_location=device)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                encoder.load_state_dict(ckpt["state_dict"], strict=False)
                state_loaded = True
                break
            elif isinstance(ckpt, dict):
                try:
                    encoder.load_state_dict(ckpt, strict=False)
                    state_loaded = True
                    break
                except Exception:
                    pass
    if not state_loaded:
        raise FileNotFoundError(
            f"Could not find a supervised checkpoint under {sup_dir}. "
            f"Expected one of: {[str(p) for p in ckpt_paths]}"
        )

    # ---- finetune (Phase B only) ----
    encoder.train()
    enc_ft, gate_head, emo_head, logs = finetune_two_stage(
        encoder=encoder,
        neural_train=Xtr.to(device), labels_train_global=Ytr.to(device),
        neural_val=Xval.to(device),   labels_val_global=Yval.to(device),
        emb_dim=latent_dim, device=device,
        no_emotion_global=0,
        seq_len=64, stride=8,
        lr_head=3e-4, lr_enc=1e-5, weight_decay=1e-4,
        lambda_tc=0.1, mu_l2sp=1e-5,
        batch_size=16, max_epochs=20, patience=5,
    )
    print(f"[Phase B] Best val macro-F1 (proxy): {logs['val_best_f1']:.3f}")
    print(f"[Phase B] Selected best_tau: {logs['best_tau']:.2f}")

    # ---- SAVE to <patient_id>/models_finetune/ ----
    pt_dir = MODEL_DIR / "models_finetune"
    pt_dir.mkdir(parents=True, exist_ok=True)

    torch.save({"state_dict": enc_ft.state_dict(), "latent_dim": latent_dim}, pt_dir / "encoder_finetuned.pt")
    torch.save({"state_dict": gate_head.state_dict()}, pt_dir / "gate_head.pt")
    torch.save({"state_dict": emo_head.state_dict()},  pt_dir / "emo_head.pt")
    torch.save({"hyperparams": {
        "seq_len":64, "stride":8, "lr_head":3e-4, "lr_enc":1e-5,
        "weight_decay":1e-4, "lambda_tc":0.1, "mu_l2sp":1e-5,
        "batch_size":16, "max_epochs":20, "patience":5,
        "best_tau": logs['best_tau'],
    }, "val_best_f1": logs['val_best_f1']}, pt_dir / "finetune_meta.pt")

    # ---- Export embedding from the FINETUNED encoder ----
    encoder.eval()
    with torch.no_grad():
        X_full = neural_tensor.to(device)                              # (T, F)
        X_full_bct = X_full.transpose(0, 1).unsqueeze(0).contiguous()  # (1, F, T)
        assert X_full_bct.shape[1] == first_conv.in_channels, \
            f"Export channels {X_full_bct.shape[1]} != encoder expects {first_conv.in_channels}"

        Z_full_bdt = encoder(X_full_bct)                               # (1, D, T')
        torch.save(Z_full_bdt.cpu(), pt_dir / "embedding_finetuned.pt")
        print(f"[DONE] Saved finetuned embedding → {pt_dir / 'embedding_finetuned.pt'}")

    # ---- Plot training curves (losses + val macro-F1) ----
    h = logs["history"]
    epochs = h["epoch"]

    plt.figure(figsize=(10, 4.5))
    # Loss panel
    plt.subplot(1, 2, 1)
    plt.plot(epochs, h["loss_total"], label="total")
    plt.plot(epochs, h["loss_gate"],  label="gate")
    plt.plot(epochs, h["loss_emo"],   label="emo")
    plt.plot(epochs, h["loss_tc"],    label="tc")
    plt.plot(epochs, h["loss_sp"],    label="l2sp")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Finetune Loss (avg per epoch)")
    plt.legend(loc="best")

    # Val F1 panel
    plt.subplot(1, 2, 2)
    plt.plot(epochs, h["val_f1"], marker="o")
    # annotate best τ for each epoch (small text)
    for e, tau in zip(epochs, h["best_tau"]):
        plt.text(e, h["val_f1"][e-1] + 1e-3, f"τ={tau:.2f}", fontsize=7, ha="center")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1 (val)")
    plt.title("Validation Macro-F1 (best τ per epoch)")
    plt.tight_layout()
    out_png = pt_dir / "finetune_curves.png"
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[PLOT] Saved finetune curves → {out_png}")

if __name__ == "__main__":
    main()
