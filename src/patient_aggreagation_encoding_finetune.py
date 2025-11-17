"""
Train encoder on aggregated multi-patient data (from scratch, no pretrained).

- Input: aggregated .npz from src/patient_aggregation.py
         keys: neural (T, F), emotion (T,), patient_ids (T,)
- Start: from scratch only. L2SP is disabled.
- Output: saves encoder and heads, finetuned embedding, and training curves.
"""
#TODO: check on grid search result to choose the parameters to use for the finetuning

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import sys
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from cebra.models import init as init_model
from cebra.data import DatasetxCEBRA, ContrastiveMultiObjectiveLoader
from cebra.solver import MultiObjectiveConfig
from cebra.solver.schedulers import LinearRampUp
from cebra.models.jacobian_regularizer import JacobianReg
import cebra as _cebra

# Ensure project root is on sys.path so `import src.*` works when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local minimal implementations to avoid importing src.config via full_encoding_finetune
class GateHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z).squeeze(-1)

class EmotionHead(nn.Module):
    def __init__(self, in_dim: int, n_active: int):
        super().__init__()
        self.n_active = int(n_active)
        self.fc = nn.Linear(in_dim, self.n_active) if self.n_active > 0 else None
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.n_active == 0:
            return z.new_zeros(z.shape[:-1] + (0,))
        return self.fc(z)

class SeqDataset(Dataset):
    """
    Sequence dataset for temporal context windows.
    Returns (x_seq, y_global, y_gate, y_emo, meta)
    """
    def __init__(self, neural: torch.Tensor, labels_global: torch.Tensor,
                 seq_len: int, stride: int, no_emotion_global: int = 0):
        self.X = neural
        self.yg = labels_global.long()
        self.seq_len, self.stride = int(seq_len), int(stride)
        self.no_global = int(no_emotion_global)
        # fixed active set = 1..9
        self.active = np.array(list(range(1, 10)), dtype=int)
        self.g2a = {g: i for i, g in enumerate(self.active)}  # 1..9 -> 0..8
        self.idxs = list(range(0, len(self.X) - self.seq_len + 1, self.stride))
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, idx):
        s = self.idxs[idx]; e = s + self.seq_len
        x = self.X[s:e]               # (Tctx, F)
        yg = self.yg[s:e]             # (Tctx,)
        y_gate = (yg != self.no_global).long()
        y_emo = torch.full_like(yg, -100)
        mask = (y_gate == 1)
        if mask.any():
            g_list = yg[mask].cpu().tolist()
            y_emo[mask] = torch.tensor([self.g2a.get(g, 0) for g in g_list], dtype=torch.long)
        meta = {"active": self.active}
        return x, yg, y_gate, y_emo, meta

def l2sp_loss(model: nn.Module, anchor: Dict[str, torch.Tensor], mu: float) -> torch.Tensor:
    if mu <= 0 or not anchor:
        return next(model.parameters()).new_zeros(())
    loss = 0.0
    for n, p in model.named_parameters():
        if p.requires_grad and n in anchor:
            loss = loss + (p - anchor[n]).pow(2).sum()
    return loss * mu

def kl_divergence(p_log_softmax, q_log_softmax):
    """KL divergence between two log-softmax distributions."""
    p = p_log_softmax.exp()
    return (p * (p_log_softmax - q_log_softmax)).sum(dim=-1)

def tc_gate(logits: torch.Tensor) -> torch.Tensor:
    """Temporal consistency for gate (Bernoulli) via prob MSE.
    Args:
        logits: (B, T) or (B, T, 1) gate logits
    Returns:
        Scalar loss: MSE between consecutive probabilities
    """
    if logits.ndim == 3:
        logits = logits.squeeze(-1)  # (B, T, 1) -> (B, T)
    if logits.size(1) < 2:
        return logits.new_zeros(())
    p = torch.sigmoid(logits)  # (B, T)
    return ((p[:, 1:] - p[:, :-1])**2).mean()

def tc_emo(logits: torch.Tensor) -> torch.Tensor:
    """Temporal consistency for emotion (categorical) via KL divergence.
    Args:
        logits: (B, T, n_classes) emotion logits
    Returns:
        Scalar loss: mean KL divergence between consecutive timesteps
    """
    if logits.ndim != 3 or logits.size(2) == 0 or logits.size(1) < 2:
        return logits.new_zeros(())
    logp = F.log_softmax(logits, dim=-1)  # (B, T, n_classes)
    return kl_divergence(logp[:, 1:], logp[:, :-1]).mean()


def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)

def l2_normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2 normalize each row (last dimension)."""
    nrm = x.norm(dim=-1, keepdim=True)
    return x / (nrm + eps)


def build_loaders(
    X: torch.Tensor, y: torch.Tensor,
    seq_len: int, stride: int,
    batch_size: int,
    split_ratio: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    T = X.shape[0]
    split = int(split_ratio * T)
    train_idx = torch.arange(split)
    val_idx = torch.arange(split, T)
    Xtr, Xval = X[train_idx], X[val_idx]
    Ytr, Yval = y[train_idx].long(), y[val_idx].long()

    ds_tr = SeqDataset(Xtr, Ytr, seq_len=seq_len, stride=stride, no_emotion_global=0)
    ds_va = SeqDataset(Xval, Yval, seq_len=seq_len, stride=stride, no_emotion_global=0)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, drop_last=False)
    return dl_tr, dl_va


def finetune_two_stage(
    encoder: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    emb_dim: int,
    device: torch.device,
    no_emotion_global: int = 0,
    lr_head: float = 3e-4,
    lr_enc: float = 1e-5,
    weight_decay: float = 1e-4,
    lambda_tc: float = 0.1,
    mu_l2sp: float = 0.0,
    max_epochs: int = 20,
    patience: int = 5,
) -> Tuple[nn.Module, nn.Module, nn.Module, Dict]:
    """Two-stage finetuning matching full_encoding_finetune.py approach."""
    from tqdm.auto import tqdm
    from sklearn.metrics import f1_score
    
    ALL_ACTIVE_GLOBALS = list(range(1, 10))  # emotions 1..9
    
    # ---- heads ----
    n_active = len(ALL_ACTIVE_GLOBALS)
    gate_head = GateHead(emb_dim).to(device)
    emo_head = EmotionHead(emb_dim, n_active).to(device)

    # ---- anchor (L2-SP) ----
    anchor = {n: p.detach().clone() for n, p in encoder.named_parameters() if p.requires_grad} if mu_l2sp > 0 else {}

    # ---- class weights (gate) ----
    # Compute from training data
    all_y_gate = []
    for _, _, y_gate, _, _ in train_loader:
        all_y_gate.append(y_gate)
    y_gate_train = torch.cat(all_y_gate, dim=0)
    y_gate_flat = (y_gate_train != no_emotion_global).long()
    cnt0 = int((y_gate_flat == 0).sum().item()) + 1
    cnt1 = int((y_gate_flat == 1).sum().item()) + 1
    denom = float(cnt0 + cnt1)
    gate_w = torch.tensor([cnt1/denom, cnt0/denom], device=device, dtype=torch.float32)
    bce_gate = nn.CrossEntropyLoss(weight=gate_w, label_smoothing=0.05)

    # ---- class weights (emo) over fixed 1..9 ----
    all_yg = []
    for _, yg, _, _, _ in train_loader:
        all_yg.append(yg)
    yg_train = torch.cat(all_yg, dim=0)
    emo_counts = [int((yg_train == g).sum().item()) + 1 for g in ALL_ACTIVE_GLOBALS]
    emo_counts = np.array(emo_counts, dtype=float)
    emo_w = (emo_counts.sum() - emo_counts) / emo_counts.sum()
    emo_w = torch.tensor(emo_w, device=device, dtype=torch.float32)
    ce_emo = nn.CrossEntropyLoss(weight=emo_w)

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
            loss_sp = l2sp_loss(encoder, anchor, mu=mu_l2sp) if mu_l2sp > 0 else torch.tensor(0.0, device=device)
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

        # ---- Validation with gate-threshold tuning (τ grid search) ----
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
        "history": hist,
    }

def _build_cebra_config_unsupervised(loader, behavior_indices=None, temperature: float = 1.0):
    cfg = MultiObjectiveConfig(loader)
    if behavior_indices is not None:
        try:
            cfg.set_slice(*behavior_indices)
        except Exception:
            pass
    cfg.set_loss("FixedCosineInfoNCE", temperature=temperature)
    cfg.set_distribution("time", time_offset=1)
    cfg.push(); cfg.finalize()
    return cfg

def _build_cebra_config_supervised(loader, behavior_indices=None, temperature: float = 1.0):
    cfg = MultiObjectiveConfig(loader)
    if behavior_indices is not None:
        try:
            cfg.set_slice(*behavior_indices)
        except Exception:
            pass
    cfg.set_loss("FixedCosineInfoNCE", temperature=temperature)
    cfg.set_distribution("time_delta", time_delta=1, label_name="position")
    cfg.push(); cfg.finalize()
    return cfg

def _cebra_train_and_export(model, loader, config, out_dir: Path, full_neural_tensor: torch.Tensor,
                            device: torch.device, num_steps: int):
    opt = torch.optim.Adam(
        list(model.parameters()) + list(config.criterion.parameters()),
        lr=3e-4, weight_decay=0.0
    )
    solver = _cebra.solver.init(
        name="multiobjective-solver",
        model=model,
        feature_ranges=config.feature_ranges,
        regularizer=JacobianReg(),
        renormalize=True,
        use_sam=False,
        criterion=config.criterion,
        optimizer=opt,
        tqdm_on=True
    ).to(device)
    scheduler = LinearRampUp(
        n_splits=1,
        step_to_switch_on_reg=max(1, num_steps // 4),
        step_to_switch_off_reg=max(2, num_steps // 2),
        start_weight=0.0,
        end_weight=0.1
    )
    solver.fit(loader=loader, valid_loader=None, scheduler_regularizer=scheduler)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model_weights.pt")
    # export embedding on full sequence
    model.eval()
    with torch.no_grad():
        X_full = full_neural_tensor.to(device)              # (T, F)
        X_full_bct = X_full.transpose(0, 1).unsqueeze(0)    # (1, F, T)
        emb = model(X_full_bct).detach().cpu()              # (1, D, T')
        torch.save(emb, out_dir / "embedding.pt")
    return solver


def main():
    parser = argparse.ArgumentParser(description="Train encoder on aggregated multi-patient data (from scratch).")
    parser.add_argument(
        "--aggregated-npz",
        type=Path,
        default=PROJECT_ROOT / "output_patient_aggregation" / "aggregated_patient_data_238_239_272_301.npz",
        help="Path to aggregated .npz (default: output_patient_aggregation/aggregated_patient_data_238_239_272_301.npz)."
    )
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output dir. Default: output_patient_aggregation/<patient_codes_from_npz>")

    # Training hyperparameters
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr-enc", type=float, default=1e-5)
    parser.add_argument("--lr-head", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-tc", type=float, default=0.1)
    # L2SP disabled in from-scratch training
    # CEBRA phase steps
    parser.add_argument("--unsup-steps", type=int, default=3500, help="Unsupervised (CEBRA-Time) steps.")
    parser.add_argument("--sup-steps", type=int, default=2500, help="Supervised (CEBRA-TimeDelta) steps.")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Derive default output directory based on the aggregated npz file name
    if args.output_dir is None:
        # Expect names like aggregated_patient_data_238_239_272_301.npz
        npz_name = args.aggregated_npz.name
        m = re.match(r"aggregated_patient_data_(.+)\.npz$", npz_name)
        code_suffix = m.group(1) if m else "agg_run"
        out_dir = (args.aggregated_npz.parent / code_suffix).resolve()
    else:
        out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load aggregated data
    data = np.load(args.aggregated_npz)
    neural = to_tensor(data["neural"])     # (T, F)
    labels = torch.tensor(data["emotion"], dtype=torch.long)  # (T,)
    print(f"[Data] neural {tuple(neural.shape)} labels {tuple(labels.shape)}")

    # Build loaders
    dl_tr, dl_va = build_loaders(
        X=neural, y=labels,
        seq_len=args.seq_len, stride=args.stride,
        batch_size=args.batch_size
    )

    # Build encoder
    encoder = init_model(
        name="offset10-model",
        num_neurons=neural.shape[1],
        num_units=256,
        num_output=args.latent_dim
    ).to(device)

    # ========= Phase A: Unsupervised pretraining (CEBRA-Time) on aggregated data =========
    ds_full = DatasetxCEBRA(neural=neural, position=labels.view(-1, 1).float())
    ds_full.configure_for(encoder)
    unsup_loader = ContrastiveMultiObjectiveLoader(dataset=ds_full, batch_size=512, num_steps=args.unsup_steps)
    unsup_cfg = _build_cebra_config_unsupervised(unsup_loader, behavior_indices=None)
    unsup_dir = out_dir / "xcebra_unsupervised"
    _ = _cebra_train_and_export(encoder, unsup_loader, unsup_cfg, unsup_dir, neural, device, args.unsup_steps)

    # ========= Phase B: Supervised fine-tuning (CEBRA-TimeDelta) =========
    sup_loader = ContrastiveMultiObjectiveLoader(dataset=ds_full, batch_size=512, num_steps=args.sup_steps)
    sup_cfg = _build_cebra_config_supervised(sup_loader, behavior_indices=None)
    sup_dir = out_dir / "xcebra_supervised"
    _ = _cebra_train_and_export(encoder, sup_loader, sup_cfg, sup_dir, neural, device, args.sup_steps)

    # From-scratch only: disable L2SP entirely
    mu_l2sp = 0.0

    # Train / Finetune
    enc, gate, emo, logs = finetune_two_stage(
        encoder=encoder,
        train_loader=dl_tr, val_loader=dl_va,
        emb_dim=args.latent_dim, device=device,
        no_emotion_global=0,
        lr_head=args.lr_head, lr_enc=args.lr_enc, weight_decay=args.weight_decay,
        lambda_tc=args.lambda_tc, mu_l2sp=mu_l2sp,
        max_epochs=args.epochs, patience=args.patience,
    )

    # Save models
    torch.save({"state_dict": enc.state_dict(), "latent_dim": args.latent_dim}, out_dir / "encoder_finetuned.pt")
    torch.save({"state_dict": gate.state_dict()}, out_dir / "gate_head.pt")
    torch.save({"state_dict": emo.state_dict()},  out_dir / "emo_head.pt")
    torch.save({
        "history": logs["history"],
        "val_best_f1": logs["val_best_f1"],
        "best_tau": logs["best_tau"]
    }, out_dir / "finetune_logs.pt")
    
    # Save comprehensive metadata with all hyperparameters and best results
    torch.save({
        "hyperparams": {
            "latent_dim": args.latent_dim,
            "seq_len": args.seq_len,
            "stride": args.stride,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "lr_enc": args.lr_enc,
            "lr_head": args.lr_head,
            "weight_decay": args.weight_decay,
            "lambda_tc": args.lambda_tc,
            "unsup_steps": args.unsup_steps,
            "sup_steps": args.sup_steps,
        },
        "best_results": {
            "val_best_f1": logs["val_best_f1"],
            "best_tau": logs["best_tau"],
        },
        "data_info": {
            "aggregated_npz": str(args.aggregated_npz),
            "neural_shape": tuple(neural.shape),
            "labels_shape": tuple(labels.shape),
        }
    }, out_dir / "finetune_meta.pt")
    print(f"[META] Saved hyperparameters and best results → {out_dir / 'finetune_meta.pt'}")

    # Export embedding for full sequence
    enc.eval()
    with torch.no_grad():
        X_full = neural.to(device)                             # (T, F)
        X_full_bct = X_full.transpose(0, 1).unsqueeze(0)       # (1, F, T)
        Z_full_bdt = enc(X_full_bct)                           # (1, D, T')
        torch.save(Z_full_bdt.cpu(), out_dir / "embedding_finetuned.pt")
        print(f"[DONE] Saved finetuned embedding → {out_dir / 'embedding_finetuned.pt'}")

    # Plot curves
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
    out_png = out_dir / "finetune_curves.png"
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[PLOT] Saved finetune curves → {out_png}")
    print(f"[Phase C] Best val macro-F1: {logs['val_best_f1']:.3f}")
    print(f"[Phase C] Selected best_tau: {logs['best_tau']:.2f}")


if __name__ == "__main__":
    main()


