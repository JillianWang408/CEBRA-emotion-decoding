# xcebra_pipeline_clean.py
import os
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import mat73, scipy.io
import cebra

from cebra.data import DatasetxCEBRA, ContrastiveMultiObjectiveLoader
from cebra.models import init as init_model

from src.config import (
    MODEL_DIR, NEURAL_PATH, EMOTION_PATH,
    FULL_NEURAL_PATH, FULL_EMOTION_PATH, MODEL_WEIGHTS_PATH, PATIENT_CONFIG
)

from src.utils import (
    align_embedding_labels
)
from src.utils_training import (
    build_cebra_config_supervised, build_cebra_config_unsupervised, train_and_save, plot_embedding_split
)

def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)


def main():
    # ---- setup ----
    pid = int(float(os.environ["PATIENT_ID"]))
    _, patient_id = PATIENT_CONFIG[pid]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ========== Load and prepare data ==========
    neural_data    = mat73.loadmat(NEURAL_PATH)['stim'].T           # (T, F)
    emotion_labels = scipy.io.loadmat(EMOTION_PATH)['resp'].flatten()  # (T,)

    neural_tensor = to_tensor(neural_data)                           # (T, F)
    label_tensor  = to_tensor(emotion_labels).unsqueeze(1)           # (T, 1)

    print(f"Neural data shape: {neural_tensor.shape}, Emotion labels shape: {label_tensor.shape}")
    # Save full tensors for record
    torch.save(neural_tensor, FULL_NEURAL_PATH)
    torch.save(label_tensor,  FULL_EMOTION_PATH)

    # Split train/test by time index (contiguous)
    T = neural_tensor.shape[0]
    split_point = int(0.8 * T)
    train_idx = torch.arange(split_point)
    test_idx  = torch.arange(split_point, T)

    neural_train, neural_test = neural_tensor[train_idx], neural_tensor[test_idx]
    label_train,  label_test  = label_tensor[train_idx],  label_tensor[test_idx]

    # ---------- Initialize model FIRST ----------
    latent_dim = 16  # set the number of offset heads you want (K)
    encoder_model = init_model(
        name="offset10-model",   # keep name consistent with K
        num_neurons=neural_train.shape[1],
        num_units=256,
        num_output=latent_dim
    ).to(device)

    # ---------- Dataset & configure_for(model) BEFORE loaders ----------
    train_dataset = DatasetxCEBRA(neural=neural_train, position=label_train)
    test_dataset  = DatasetxCEBRA(neural=neural_test,  position=label_test)
    train_dataset.configure_for(encoder_model)
    test_dataset.configure_for(encoder_model)

    # ---------- Loaders (use realistic steps; tweak as needed) ----------
    steps_unsup = 2000 
    steps_sup   = 1500

    unsupervised_loader = ContrastiveMultiObjectiveLoader(
        dataset=train_dataset, batch_size=512, num_steps=steps_unsup
    )
    supervised_loader = ContrastiveMultiObjectiveLoader(
        dataset=train_dataset, batch_size=512, num_steps=steps_sup
    )

    # ---------- Configs ----------
    BEHAVIOR_INDICES = (0, 16)
    unsupervised_config = build_cebra_config_unsupervised(
        unsupervised_loader, BEHAVIOR_INDICES
    )
    supervised_config = build_cebra_config_supervised(
        supervised_loader, BEHAVIOR_INDICES
    )

    # ------------------------
    # Unsupervised Pretraining (CEBRA-Time)
    # ------------------------
    unsup_dir = MODEL_DIR / "xcebra_unsupervised"
    unsup_dir.mkdir(parents=True, exist_ok=True)
    _ = train_and_save(
        model=encoder_model,                  # base encoder; weights updated in place
        loader=unsupervised_loader,
        config=unsupervised_config,
        out_dir=unsup_dir,                    # utils saves model_weights.pt + embedding.pt here
        full_neural_tensor=neural_tensor,     # save FULL embedding once; we’ll slice for plots
        meta={"phase": "unsupervised", "latent_dim": latent_dim},
        device=device,
        num_steps=steps_unsup
    )
    print(f"[DONE] Unsupervised training for patient {patient_id}")

    # # Plot and save the unsupervised loss curve  # NEW
    # ax = cebra.plot_loss(_)  # `_` is the unsup solver returned by train_and_save
    # ax.figure.suptitle("Loss – Unsupervised (CEBRA-Time)")
    # ax.figure.savefig(unsup_dir / "loss_unsupervised.png", dpi=200, bbox_inches="tight")
    # plt.close(ax.figure)

    # # Optional: GoF on unsupervised model (no labels)  # NEW
    # gof_unsup = cebra.sklearn.metrics.goodness_of_fit_score(_, train_dataset.neural)
    # print(f"[GoF] Unsupervised (full): {gof_unsup:.3f} bits")

    # ------------------------
    # Supervised Fine-tuning (CEBRA-TimeDelta)
    # ------------------------
    sup_dir = MODEL_DIR / "xcebra_supervised"
    sup_dir.mkdir(parents=True, exist_ok=True)
    _ = train_and_save(
        model=encoder_model,                  # continue from pretrained weights (same instance)
        loader=supervised_loader,
        config=supervised_config,
        out_dir=sup_dir,                      # utils saves model_weights.pt + embedding.pt here
        full_neural_tensor=neural_tensor,     # save FULL embedding once; we’ll slice for plots
        meta={"phase": "supervised", "time_delta": 1, "latent_dim": latent_dim},
        device=device,
        num_steps=steps_sup
    )

    # # Plot and save the supervised loss curve  # NEW
    # ax = cebra.plot_loss(_)
    # ax.figure.suptitle("Loss – Supervised (time_delta)")
    # ax.figure.savefig(sup_dir / "loss_supervised.png", dpi=200, bbox_inches="tight")
    # plt.close(ax.figure)

    # # Optional: GoF on supervised model; pass labels since it's supervised  # NEW
    # gof_sup = cebra.sklearn.metrics.goodness_of_fit_score(
    #     _, train_dataset.neural, train_dataset.position.cpu().numpy()
    # )
    # print(f"[GoF] Supervised (full): {gof_sup:.3f} bits")

    # ------------------------
    # Load saved embeddings and plot (train/test splits)
    # ------------------------
    Z_unsup_full = torch.load(unsup_dir / "embedding.pt")  
    Z_sup_full   = torch.load(sup_dir  / "embedding.pt")

    Z_unsup = Z_unsup_full.squeeze(0).T  # (T, K)
    Z_sup   = Z_sup_full.squeeze(0).T    # (T, K)
    print("Z_unsup shape:", Z_unsup.shape)
    print("Z_sup shape:", Z_sup.shape)
    
    T = Z_unsup.shape[0]
    split_point_Z = int(0.8 * T)
    print(f"Embedding split point at index {split_point_Z} of {T} total time points")

    y_full = label_tensor.squeeze(1).cpu().numpy()

    # ---- Align and Plot UNSUPERVISED ----
    y_aligned_unsup, offset_unsup, split_unsup = align_embedding_labels(Z_unsup, y_full)
    plot_embedding_split(Z_unsup, y_aligned_unsup, split_unsup, unsup_dir, "emb_unsup", "Unsupervised")

    # ---- Align and Plot SUPERVISED ----
    y_aligned_sup, offset_sup, split_sup = align_embedding_labels(Z_sup, y_full)
    plot_embedding_split(Z_sup, y_aligned_sup, split_sup, sup_dir, "emb_sup", "Supervised (time_delta)")

    print(f"[DONE] Saved embeddings and plots in:\n  {unsup_dir}\n  {sup_dir}")


if __name__ == "__main__":
    main()
