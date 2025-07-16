# RUN WITH: python -m src.attribution

import numpy as np
import torch
import seaborn as sns
import cebra
from cebra.models import init as init_model
import os

import matplotlib
matplotlib.use('MacOSX') 
import matplotlib.pyplot as plt

from src.config import (
    EMOTION_TENSOR_PATH, NEURAL_TENSOR_PATH,
    N_LATENTS, N_ELECTRODES, ELECTRODE_NAMES, MODEL_WEIGHTS_PATH
)


def compute_and_plot_attribution(model):
    # === Load Neural Data ===
    emotion_tensor = torch.load(EMOTION_TENSOR_PATH)
    neural_tensor = torch.load(NEURAL_TENSOR_PATH)
    neural_tensor.requires_grad_(True)

    # === Attribution Method ===
    model.split_outputs = False
    method = cebra.attribution.init(
        name="jacobian-based",
        model=model,
        input_data=neural_tensor,
        output_dimension=model.num_output
    )
    result = method.compute_attribution_map()

    # === Create output folder ===
    os.makedirs("attribution_outputs", exist_ok=True)

    # === Plot Per-Latent Heatmap + Electrode Covariance ===
    for key in ["jf", "jf-inv-svd"]:
        jf = result[key].transpose(1, 2, 0)  # [latents, neurons, time]
        jf_mean = np.abs(jf).mean(-1)       # [latents, neurons]

        for i in range(jf_mean.shape[0]):
            plt.figure(figsize=(12, 1.5))
            plt.imshow(jf_mean[i][None, :], cmap='viridis', aspect='auto')
            plt.colorbar(label="Attribution Strength")
            plt.yticks([0], [f"Latent {i}"])
            plt.xticks(np.linspace(0, jf_mean.shape[1] - 1, 10).astype(int))
            plt.xlabel("Neuron Index")
            plt.title(f"Attribution Map for Latent {i} ({key})")
            plt.tight_layout()
            plt.savefig(f"attribution_outputs/latent_{i}_{key}_heatmap.png")
            plt.close()
        
        # === 40x40 Electrode Covariance Plots ===
        if key == "jf":
            for i in range(N_LATENTS):
                jf_cov = jf_mean[i].reshape(N_ELECTRODES, N_ELECTRODES)

                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    jf_cov,
                    cmap='viridis',
                    square=True,
                    cbar=True,
                    xticklabels=ELECTRODE_NAMES,
                    yticklabels=ELECTRODE_NAMES
                )
                plt.xticks(rotation=90)
                plt.title(f"Latent {i} — Electrode Attribution Covariance")
                plt.xlabel("Electrode")
                plt.ylabel("Electrode")
                plt.tight_layout()
                plt.savefig(f"attribution_outputs/latent_{i}_electrode_covariance.png")
                plt.close()

    # === Summary Attribution Maps ===
    plt.matshow(np.abs(result['jf']).mean(0), aspect="auto")
    plt.colorbar()
    plt.title("Attribution map of JF (mean over time)")
    plt.tight_layout()
    plt.savefig("attribution_outputs/jf_summary.png")
    plt.close()

    plt.matshow(np.abs(result['jf-inv-svd']).mean(0), aspect="auto")
    plt.colorbar()
    plt.title("Attribution map of JFinv (mean over time)")
    plt.tight_layout()
    plt.savefig("attribution_outputs/jfinv_summary.png")
    plt.close()

    # === Diagnostics ===
    top_inputs = result['jf'].mean(0).argsort()[:10]
    print("Attribution analysis complete.")
    print("Neural shape:", neural_tensor.shape)
    print("Top input feature indices:", top_inputs)
    print("Saved outputs in: ./attribution_outputs/")

if __name__ == "__main__":
    # === Load trained model
    embedding_model = init_model(
        name="offset10-model",
        num_neurons=torch.load(NEURAL_TENSOR_PATH).shape[1],
        num_units=256,
        num_output=N_LATENTS
    )
    
    # Load saved model weights
    raw_state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location="cpu")

    # Fix nested "module." prefixes
    fixed_state_dict = {}
    fixed_state_dict = {}

    for k, v in raw_state_dict.items():
        # Remove top-level 'module.' if present
        if k.startswith("module."):
            k = k[len("module."):]

        # Remove extra nested 'module.' inside keys
        k = k.replace(".module.", ".")

        # Fix specific nested keys
        if k.startswith("net.2.0"):
            k = k.replace("net.2.0", "net.2.module.0")
        elif k.startswith("net.3.0"):
            k = k.replace("net.3.0", "net.3.module.0")
        elif k.startswith("net.4.0"):
            k = k.replace("net.4.0", "net.4.module.0")

        # Add the (possibly modified) key-value pair
        fixed_state_dict[k] = v

    # Load into model
    embedding_model.load_state_dict(fixed_state_dict)
    embedding_model.eval()

    # === Run attribution
    compute_and_plot_attribution(embedding_model)
