"""
CEBRA encoding finetune adapted for calc/pred data.
Load supervised encoder, finetune two-stage heads on calc, save embeddings for calc and pred.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import mat73
import scipy.io
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from cebra.models import init as init_model
from sklearn.metrics import f1_score

from src.CEBRA.config_loader import load_cebra_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALL_ACTIVE_GLOBALS: List[int] = list(range(1, 10))


def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)


def l2_normalize_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    nrm = x.norm(dim=-1, keepdim=True)
    return x / (nrm + eps)


class GateHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, z):
        return self.fc(z).squeeze(-1)


class EmotionHead(nn.Module):
    def __init__(self, in_dim: int, n_active: int):
        super().__init__()
        self.n_active = int(n_active)
        self.fc = nn.Linear(in_dim, self.n_active) if self.n_active > 0 else None

    def forward(self, z):
        if self.n_active == 0:
            return z.new_zeros(z.shape[:-1] + (0,))
        return self.fc(z)


class SeqDataset(Dataset):
    def __init__(self, neural: torch.Tensor, labels_global: torch.Tensor,
                 seq_len: int, stride: int, no_emotion_global: int = 0):
        self.X = neural
        self.yg = labels_global.long()
        self.seq_len, self.stride = int(seq_len), int(stride)
        self.no_global = int(no_emotion_global)
        self.active = np.array(ALL_ACTIVE_GLOBALS, dtype=int)
        self.g2a = {g: i for i, g in enumerate(self.active)}
        self.idxs = list(range(0, len(self.X) - self.seq_len + 1, self.stride))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        s = self.idxs[idx]
        e = s + self.seq_len
        x = self.X[s:e]
        yg = self.yg[s:e]
        y_gate = (yg != self.no_global).long()
        y_emo = torch.full_like(yg, -100)
        mask = (y_gate == 1)
        if mask.any() and len(self.active) > 0:
            g_list = yg[mask].cpu().tolist()
            y_emo[mask] = torch.tensor([self.g2a.get(g, 0) for g in g_list], dtype=torch.long)
        return x, yg, y_gate, y_emo, {"active": self.active}


def tc_gate(logits: torch.Tensor) -> torch.Tensor:
    if logits.size(1) < 2:
        return logits.new_zeros(())
    p = torch.sigmoid(logits)
    return ((p[:, 1:] - p[:, :-1]) ** 2).mean()


def kl_divergence(p_log_softmax, q_log_softmax):
    p = p_log_softmax.exp()
    return (p * (p_log_softmax - q_log_softmax)).sum(dim=-1)


def tc_emo(logits: torch.Tensor) -> torch.Tensor:
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


def finetune_two_stage(
    encoder: nn.Module,
    neural_train: torch.Tensor, labels_train_global: torch.Tensor,
    neural_val: torch.Tensor, labels_val_global: torch.Tensor,
    emb_dim: int, device: torch.device,
    no_emotion_global: int = 0,
    seq_len: int = 64, stride: int = 8,
    lr_head: float = 3e-4, lr_enc: float = 2e-5, weight_decay: float = 1e-4,
    lambda_tc: float = 0.1, mu_l2sp: float = 1e-5,
    batch_size: int = 16, max_epochs: int = 20, patience: int = 5,
):
    n_active = len(ALL_ACTIVE_GLOBALS)
    gate_head = GateHead(emb_dim).to(device)
    emo_head = EmotionHead(emb_dim, n_active).to(device)
    anchor = {n: p.detach().clone() for n, p in encoder.named_parameters() if p.requires_grad}

    y_gate_train = (labels_train_global != no_emotion_global).long()
    cnt0 = int((y_gate_train == 0).sum().item()) + 1
    cnt1 = int((y_gate_train == 1).sum().item()) + 1
    gate_w = torch.tensor([cnt1 / (cnt0 + cnt1), cnt0 / (cnt0 + cnt1)], device=device, dtype=torch.float32)
    bce_gate = nn.CrossEntropyLoss(weight=gate_w, label_smoothing=0.05)

    emo_counts = [int((labels_train_global == g).sum().item()) + 1 for g in ALL_ACTIVE_GLOBALS]
    emo_counts = np.array(emo_counts, dtype=float)
    emo_w = (emo_counts.sum() - emo_counts) / emo_counts.sum()
    emo_w = torch.tensor(emo_w, device=device, dtype=torch.float32)
    ce_emo = nn.CrossEntropyLoss(weight=emo_w)

    train_ds = SeqDataset(neural_train, labels_train_global, seq_len, stride, no_emotion_global)
    val_ds = SeqDataset(neural_val, labels_val_global, seq_len, stride, no_emotion_global)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    opt = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": lr_enc},
        {"params": gate_head.parameters(), "lr": lr_head},
        {"params": emo_head.parameters(), "lr": lr_head},
    ], weight_decay=weight_decay)

    first_conv = next(m for m in encoder.modules() if isinstance(m, nn.Conv1d))
    hist = {"epoch": [], "loss_total": [], "val_f1": [], "best_tau": []}
    best_f1, best_state, best_tau, bad = -1.0, None, 0.5, 0

    for epoch in range(1, max_epochs + 1):
        encoder.train()
        gate_head.train()
        emo_head.train()
        sum_total = 0.0
        n_batches = 0

        for x, yg, y_gate, y_emo, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}", leave=False):
            x = x.to(device)
            y_gate = y_gate.to(device)
            y_emo = y_emo.to(device)

            x_bct = x.permute(0, 2, 1).contiguous()
            z_bdt = encoder(x_bct)
            z = z_bdt.permute(0, 2, 1).contiguous()
            z = l2_normalize_rows(z.reshape(-1, z.size(-1))).reshape(z.size())

            T_enc = z.size(1)
            if T_enc != y_gate.size(1):
                shift = (y_gate.size(1) - T_enc) // 2
                y_gate = y_gate[:, shift : shift + T_enc]
                y_emo = y_emo[:, shift : shift + T_enc]

            gate_logits = gate_head(z)
            gate_two = torch.stack([-gate_logits, gate_logits], -1)
            emo_logits = emo_head(z)

            loss_gate = bce_gate(gate_two.reshape(-1, 2), y_gate.reshape(-1))
            y_emo_flat = y_emo.reshape(-1)
            mask = (y_emo_flat != -100)
            loss_emo = ce_emo(emo_logits.reshape(-1, emo_logits.size(-1))[mask], y_emo_flat[mask]) if mask.any() else torch.zeros((), device=z.device)
            loss_tc = lambda_tc * (tc_gate(gate_logits) + tc_emo(emo_logits))
            loss_sp = l2sp_loss(encoder, anchor, mu=mu_l2sp)
            loss = loss_gate + loss_emo + loss_tc + loss_sp

            opt.zero_grad()
            loss.backward()
            opt.step()
            sum_total += loss.item()
            n_batches += 1

        encoder.eval()
        gate_head.eval()
        emo_head.eval()
        all_yg, all_pa, all_pe = [], [], []
        with torch.no_grad():
            for x, yg, *_ in val_loader:
                x = x.to(device)
                x_bct = x.permute(0, 2, 1).contiguous()
                z_bdt = encoder(x_bct)
                z = z_bdt.permute(0, 2, 1).contiguous()
                z = l2_normalize_rows(z.reshape(-1, z.size(-1))).reshape(z.size())
                T_enc = z.size(1)
                if T_enc != yg.size(1):
                    shift = (yg.size(1) - T_enc) // 2
                    yg = yg[:, shift : shift + T_enc]
                gl = gate_head(z)
                pa = torch.sigmoid(gl)
                el = emo_head(z)
                pe = F.softmax(el, dim=-1)
                all_yg.append(yg.cpu())
                all_pa.append(pa.cpu())
                all_pe.append(pe.cpu())

        YG = torch.cat(all_yg, dim=0)
        PA = torch.cat(all_pa, dim=0)
        PE = torch.cat(all_pe, dim=0)
        taus = torch.tensor([0.1, 0.3, 0.4, 0.45, 0.5, 0.55])
        best_f1_tau, best_tau_epoch = -1.0, 0.5
        for tau in taus:
            gate_on = (PA >= tau.item()).float().unsqueeze(-1)
            p_no = (1.0 - gate_on)
            p_act = gate_on * PE
            P_all = torch.cat([p_no, p_act], dim=-1)
            map_ids = torch.tensor([0] + ALL_ACTIVE_GLOBALS, dtype=torch.long)
            Ypred = map_ids[P_all.argmax(dim=-1)]
            f1m = f1_score(YG.reshape(-1).numpy(), Ypred.reshape(-1).numpy(), average="macro")
            if f1m > best_f1_tau:
                best_f1_tau, best_tau_epoch = f1m, float(tau.item())

        hist["epoch"].append(epoch)
        hist["loss_total"].append(sum_total / max(n_batches, 1))
        hist["val_f1"].append(best_f1_tau)
        hist["best_tau"].append(best_tau_epoch)

        if best_f1_tau > best_f1:
            best_f1, best_tau = best_f1_tau, best_tau_epoch
            best_state = {
                "enc": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                "gate": {k: v.detach().cpu().clone() for k, v in gate_head.state_dict().items()},
                "emo": {k: v.detach().cpu().clone() for k, v in emo_head.state_dict().items()},
            }
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        encoder.load_state_dict(best_state["enc"])
        gate_head.load_state_dict(best_state["gate"])
        emo_head.load_state_dict(best_state["emo"])

    return encoder, gate_head, emo_head, {"val_best_f1": best_f1, "best_tau": best_tau, "history": hist}


def load_mat_neural(path):
    try:
        return mat73.loadmat(str(path))["stim"].T
    except Exception:
        return scipy.io.loadmat(str(path))["stim"].T


def load_mat_labels(path):
    return scipy.io.loadmat(str(path))["resp"].flatten()


def main():
    parser = argparse.ArgumentParser(description="CEBRA encoding finetune (calc/pred)")
    parser.add_argument("--patient-id", type=int, default=None)
    parser.add_argument("--target", type=str, default="9emotion",
        choices=["9emotion", "arousal", "valence", "categories"])
    args = parser.parse_args()

    patient_id = args.patient_id or int(float(os.environ["PATIENT_ID"]))
    cfg = load_cebra_config(patient_id, args.target)

    # Finetune two-stage heads (gate + emotion) are designed for 9-emotion (0=no emotion, 1-9=emotions).
    # For arousal/valence/categories, skip finetune and use encoding output directly.

    if args.target != "9emotion":
        print("[INFO] encoding_finetune is for 9emotion only. For arousal/valence/categories, run encoding then decoding_finetune.")
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 16

    neural_calc = load_mat_neural(cfg["neural_train"])
    labels_calc = load_mat_labels(cfg["label_train"])
    T = min(neural_calc.shape[0], labels_calc.shape[0])
    neural_tensor = to_tensor(neural_calc[:T])
    label_tensor = to_tensor(labels_calc[:T]).reshape(-1)

    split = int(0.8 * len(neural_tensor))
    Xtr = neural_tensor[:split]
    Ytr = label_tensor[:split]
    Xval = neural_tensor[split:]
    Yval = label_tensor[split:]

    encoder = init_model(
        name="offset10-model",
        num_neurons=neural_tensor.shape[1],
        num_units=256,
        num_output=latent_dim,
    ).to(device)

    sup_dir = cfg["model_dir"] / "xcebra_supervised"
    for pth in [sup_dir / "model_weights.pt", sup_dir / "model.pt", sup_dir / "checkpoint.pt"]:
        if pth.exists():
            ckpt = torch.load(pth, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                encoder.load_state_dict(ckpt["state_dict"], strict=False)
            elif isinstance(ckpt, dict):
                try:
                    encoder.load_state_dict(ckpt, strict=False)
                except Exception:
                    pass
            break
    else:
        raise FileNotFoundError(f"No supervised checkpoint in {sup_dir}")

    enc_ft, gate_head, emo_head, logs = finetune_two_stage(
        encoder=encoder,
        neural_train=Xtr.to(device), labels_train_global=Ytr.to(device),
        neural_val=Xval.to(device), labels_val_global=Yval.to(device),
        emb_dim=latent_dim, device=device,
        no_emotion_global=0,
        seq_len=64, stride=8,
        lr_head=3e-4, lr_enc=1e-5, weight_decay=1e-4,
        lambda_tc=0.1, mu_l2sp=1e-5,
        batch_size=16, max_epochs=20, patience=5,
    )

    pt_dir = cfg["model_dir"] / "models_finetune"
    pt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": enc_ft.state_dict(), "latent_dim": latent_dim}, pt_dir / "encoder_finetuned.pt")
    torch.save({"state_dict": gate_head.state_dict()}, pt_dir / "gate_head.pt")
    torch.save({"state_dict": emo_head.state_dict()}, pt_dir / "emo_head.pt")
    torch.save({"hyperparams": {"best_tau": logs["best_tau"]}, "val_best_f1": logs["val_best_f1"]}, pt_dir / "finetune_meta.pt")

    first_conv = next(m for m in encoder.modules() if isinstance(m, nn.Conv1d))
    enc_ft.eval()
    with torch.no_grad():
        for name, neural_pt in [("calc", neural_tensor), ("pred", to_tensor(load_mat_neural(cfg["neural_test"])))]:
            X = neural_pt.T.unsqueeze(0).to(device)
            Z = enc_ft(X).cpu()
            torch.save(Z, pt_dir / f"embedding_{name}.pt")

    if logs["history"]["epoch"]:
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(logs["history"]["epoch"], logs["history"]["loss_total"])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.subplot(1, 2, 2)
        plt.plot(logs["history"]["epoch"], logs["history"]["val_f1"])
        plt.xlabel("Epoch")
        plt.ylabel("Val Macro-F1")
        plt.tight_layout()
        plt.savefig(pt_dir / "finetune_curves.png", dpi=160)
        plt.close()

    print(f"[DONE] Finetune complete. Embeddings saved to {pt_dir}")


if __name__ == "__main__":
    main()
