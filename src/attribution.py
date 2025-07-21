# RUN WITH: python -m src.attribution

import numpy as np
import torch
import seaborn as sns
import matplotlib
matplotlib.use('MacOSX') 
import matplotlib.pyplot as plt
import os
import cebra

from src.config import (
    EMOTION_TENSOR_PATH, NEURAL_TENSOR_PATH,
    N_LATENTS, N_ELECTRODES, ELECTRODE_NAMES, MODEL_WEIGHTS_PATH
)
from src.utils import load_fixed_cebra_model

def compute_and_plot_attribution(model):
    # === Load data ===
    emotion_tensor = torch.load(EMOTION_TENSOR_PATH)
    neural_tensor = torch.load(NEURAL_TENSOR_PATH)
    neural_tensor.requires_grad_(True)

    # === Attribution ===
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

    # === Latent heatmaps ===
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

        # === Electrode 40x40 grid heatmap ===
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

    # === Summary attribution maps ===
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
    model = load_fixed_cebra_model(MODEL_WEIGHTS_PATH, num_output=N_LATENTS)
    compute_and_plot_attribution(model)
