# RUN WITH: python -m src.attribution_supervised

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
    N_LATENTS, N_ELECTRODES, ELECTRODE_NAMES,
    MODEL_WEIGHTS_PATH, ATTRIBUTION_OUTPUT_DIR
)
from src.utils import load_fixed_cebra_model

# Ensure attribution output directory exists
ATTRIBUTION_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def compute_and_plot_attribution(model):
    # === Load tensors ===
    emotion_tensor = torch.load(EMOTION_TENSOR_PATH)
    neural_tensor = torch.load(NEURAL_TENSOR_PATH)
    neural_tensor.requires_grad_(True)

    # === Initialize attribution method ===
    model.split_outputs = False
    method = cebra.attribution.init(
        name="jacobian-based",
        model=model,
        input_data=neural_tensor,
        output_dimension=model.num_output
    )
    result = method.compute_attribution_map()

    # === Save latent heatmaps ===
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
            plt.savefig(ATTRIBUTION_OUTPUT_DIR / f"latent_{i}_{key}_heatmap.png")
            plt.close()

        # Save electrode grid heatmaps only for 'jf'
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
                plt.savefig(ATTRIBUTION_OUTPUT_DIR / f"latent_{i}_electrode_covariance.png")
                plt.close()

    # === Summary maps (average over time) ===
    plt.matshow(np.abs(result['jf']).mean(0), aspect="auto")
    plt.colorbar()
    plt.title("Attribution map of JF (mean over time)")
    plt.tight_layout()
    plt.savefig(ATTRIBUTION_OUTPUT_DIR / "jf_summary.png")
    plt.close()

    plt.matshow(np.abs(result['jf-inv-svd']).mean(0), aspect="auto")
    plt.colorbar()
    plt.title("Attribution map of JFinv (mean over time)")
    plt.tight_layout()
    plt.savefig(ATTRIBUTION_OUTPUT_DIR / "jfinv_summary.png")
    plt.close()

    # === Aggregate electrode attribution (average over all latents) ===
    jf = result["jf"].transpose(1, 2, 0)
    jf_mean = np.abs(jf).mean(-1)

    def plot_electrode_cov(mean_vector, title, filename):
        """Helper to plot reshaped electrode-level heatmap."""
        jf_cov = mean_vector.reshape(N_ELECTRODES, N_ELECTRODES)
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
        plt.title(title)
        plt.xlabel("Electrode")
        plt.ylabel("Electrode")
        plt.tight_layout()
        plt.savefig(ATTRIBUTION_OUTPUT_DIR / filename)
        plt.close()

    # Grouped summary plots
    plot_electrode_cov(
        mean_vector=jf_mean.mean(axis=0),
        title="Electrode Attribution Covariance — all10 latents",
        filename="electrode_covariance_summary_all10.png"
    )

    # === Diagnostics ===
    top_inputs = result['jf'].mean(0).argsort()[:10]
    print("Attribution analysis complete.")
    print("Neural shape:", neural_tensor.shape)
    print("Top input feature indices:", top_inputs)

if __name__ == "__main__":
    model = load_fixed_cebra_model(MODEL_WEIGHTS_PATH, num_output=N_LATENTS)
    compute_and_plot_attribution(model)
