"""
CEBRA encoding adapted for calc/pred data.
Train on calc (unsupervised + supervised), save embeddings for calc and pred.
"""

import argparse
import os
from pathlib import Path

# Set PATIENT_ID before importing src.config (via utils)
def _parse_patient_early():
    p = argparse.ArgumentParser()
    p.add_argument("--patient-id", type=int, default=None)
    p.add_argument("--target", type=str, default="9emotion")
    p.add_argument("--steps", type=int, default=None)
    args, _ = p.parse_known_args()
    pid = args.patient_id if args.patient_id is not None else os.environ.get("PATIENT_ID", "9")
    os.environ["PATIENT_ID"] = str(int(float(pid)))
    return args

_early_args = _parse_patient_early()

import numpy as np
import torch
import mat73
import scipy.io
import cebra

from cebra.data import DatasetxCEBRA, ContrastiveMultiObjectiveLoader
from cebra.models import init as init_model

from src.general.utils import align_embedding_labels
from src.general.utils_training import (
    build_cebra_config_supervised,
    build_cebra_config_unsupervised,
    train_and_save,
    plot_embedding_split,
)
from src.CEBRA.config_loader import load_cebra_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)


def load_mat_neural(path):
    try:
        return mat73.loadmat(str(path))["stim"].T
    except Exception:
        return scipy.io.loadmat(str(path))["stim"].T


def load_mat_labels(path):
    return scipy.io.loadmat(str(path))["resp"].flatten()


def main():
    parser = argparse.ArgumentParser(description="CEBRA encoding (calc=train, pred=test)")
    parser.add_argument("--patient-id", type=int, default=None, help="Patient ID (default: PATIENT_ID env)")
    parser.add_argument("--target", type=str, default="9emotion",
        choices=["9emotion", "arousal", "valence", "categories"],
        help="Target: 9emotion, arousal, valence, categories")
    parser.add_argument("--steps", type=int, default=None,
        help="Override steps for unsup+sup (default: 2000/1500). Use small value for quick test, e.g. --steps 50")
    args = parser.parse_args()

    patient_id = args.patient_id if args.patient_id is not None else int(float(os.environ["PATIENT_ID"]))

    cfg = load_cebra_config(patient_id, args.target)
    if args.target != "9emotion":
        os.environ["PATIENT_ID"] = str(patient_id)
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", PROJECT_ROOT / "src" / "config.py")
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        if patient_id not in getattr(config, "PATIENTS_WITH_VALENCE_AROUSAL", []):
            raise ValueError(f"Patient {patient_id} has no valence/arousal data.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg["model_dir"].mkdir(parents=True, exist_ok=True)

    # Load calc (train) data
    neural_calc = load_mat_neural(cfg["neural_train"])
    labels_calc = load_mat_labels(cfg["label_train"])
    T = min(neural_calc.shape[0], labels_calc.shape[0])
    neural_calc = neural_calc[:T]
    labels_calc = labels_calc[:T]

    neural_tensor = to_tensor(neural_calc)
    label_tensor = to_tensor(labels_calc).unsqueeze(1)

    print(f"[1] Loaded calc: neural {neural_tensor.shape}, labels {label_tensor.shape}")

    # Split train/val by time (80/20) for unsup/sup
    split_point = int(0.8 * neural_tensor.shape[0])
    train_idx = torch.arange(split_point)
    val_idx = torch.arange(split_point, neural_tensor.shape[0])
    neural_train = neural_tensor[train_idx]
    label_train = label_tensor[train_idx]
    neural_val = neural_tensor[val_idx]
    label_val = label_tensor[val_idx]

    # Init model
    latent_dim = 16
    encoder_model = init_model(
        name="offset10-model",
        num_neurons=neural_train.shape[1],
        num_units=256,
        num_output=latent_dim,
    ).to(device)

    train_dataset = DatasetxCEBRA(neural=neural_train, position=label_train)
    val_dataset = DatasetxCEBRA(neural=neural_val, position=label_val)
    train_dataset.configure_for(encoder_model)
    val_dataset.configure_for(encoder_model)

    if args.steps is not None:
        steps_unsup = steps_sup = args.steps
        print(f"[Config] Quick test: steps={args.steps} (use default for full training)")
    else:
        steps_unsup, steps_sup = 2000, 1500
    BEHAVIOR_INDICES = (0, 16)

    unsupervised_loader = ContrastiveMultiObjectiveLoader(
        dataset=train_dataset, batch_size=512, num_steps=steps_unsup
    )
    supervised_loader = ContrastiveMultiObjectiveLoader(
        dataset=train_dataset, batch_size=512, num_steps=steps_sup
    )

    unsupervised_config = build_cebra_config_unsupervised(
        unsupervised_loader, BEHAVIOR_INDICES
    )
    supervised_config = build_cebra_config_supervised(
        supervised_loader, BEHAVIOR_INDICES
    )

    # Unsupervised
    unsup_dir = cfg["model_dir"] / "xcebra_unsupervised"
    unsup_dir.mkdir(parents=True, exist_ok=True)
    _ = train_and_save(
        model=encoder_model,
        loader=unsupervised_loader,
        config=unsupervised_config,
        out_dir=unsup_dir,
        full_neural_tensor=neural_tensor,
        meta={"phase": "unsupervised", "latent_dim": latent_dim},
        device=device,
        num_steps=steps_unsup,
    )
    print(f"[DONE] Unsupervised training")

    # Supervised
    sup_dir = cfg["model_dir"] / "xcebra_supervised"
    sup_dir.mkdir(parents=True, exist_ok=True)
    _ = train_and_save(
        model=encoder_model,
        loader=supervised_loader,
        config=supervised_config,
        out_dir=sup_dir,
        full_neural_tensor=neural_tensor,
        meta={"phase": "supervised", "time_delta": 1, "latent_dim": latent_dim},
        device=device,
        num_steps=steps_sup,
    )
    print(f"[DONE] Supervised training")

    # Encode calc and pred
    encoder_model.eval()
    encoder_model.split_outputs = False

    def encode_neural(neural_pt):
        X = neural_pt.T.unsqueeze(0).to(device)
        with torch.no_grad():
            emb = encoder_model(X).detach().cpu()
        return emb

    emb_calc = encode_neural(neural_tensor)
    torch.save(emb_calc, unsup_dir / "embedding_calc.pt")
    torch.save(emb_calc, sup_dir / "embedding_calc.pt")

    # Load and encode pred
    neural_pred = load_mat_neural(cfg["neural_test"])
    neural_pred_t = to_tensor(neural_pred)
    emb_pred = encode_neural(neural_pred_t)
    torch.save(emb_pred, unsup_dir / "embedding_pred.pt")
    torch.save(emb_pred, sup_dir / "embedding_pred.pt")

    # Save labels for decoding
    torch.save(label_tensor, cfg["model_dir"] / "labels_calc.pt")
    labels_pred = load_mat_labels(cfg["label_test"])
    T_pred = min(neural_pred.shape[0], labels_pred.shape[0])
    torch.save(to_tensor(labels_pred[:T_pred]).unsqueeze(1), cfg["model_dir"] / "labels_pred.pt")

    # Plot embeddings (calc only, train/val split)
    Z_sup = torch.load(sup_dir / "embedding_calc.pt").squeeze(0).T.numpy()
    y_full = label_tensor.squeeze(1).cpu().numpy()
    y_aligned, offset, split = align_embedding_labels(Z_sup, y_full)
    plot_embedding_split(Z_sup, y_aligned, split, sup_dir, "emb_sup", "Supervised (calc)")

    (cfg["model_dir"] / "n_calc.txt").write_text(str(neural_tensor.shape[0]))
    print(f"[DONE] Saved embeddings and plots to {cfg['model_dir']}")


if __name__ == "__main__":
    main()
