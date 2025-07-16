import numpy as np
import torch
import matplotlib.pyplot as plt
import cebra
from config import NEURAL_PATH
import mat73
import seaborn as sns


def compute_and_plot_attribution(model):
    # === Load Neural Data ===
    emotion_tensor = torch.load(EMOTION_TENSOR_PATH)
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

    # === Plot Per-Latent Heatmap ===
    for key in ["jf", "jf-inv-svd"]:
        jf = result[key].transpose(1, 2, 0)  # shape: [num_latents, num_neurons, time]
        jf_mean = np.abs(jf).mean(-1)       # shape: [num_latents, num_neurons]

        for i in range(jf_mean.shape[0]):
            plt.figure(figsize=(12, 1.5))
            plt.imshow(jf_mean[i][None, :], cmap='viridis', aspect='auto')
            plt.colorbar(label="Attribution Strength")
            plt.yticks([0], [f"Latent {i}"])
            plt.xticks(np.linspace(0, jf_mean.shape[1] - 1, 10).astype(int))
            plt.xlabel("Neuron Index")
            plt.title(f"Attribution Map for Latent Dimension {i} ({key})")
            plt.tight_layout()
            plt.show()

    # === Summary Visualization ===
    plt.matshow(np.abs(result['jf']).mean(0), aspect="auto")
    plt.colorbar()
    plt.title("Attribution map of JF")
    plt.show()

    plt.matshow(np.abs(result['jf-inv-svd']).mean(0), aspect="auto")
    plt.colorbar()
    plt.title("Attribution map of JFinv")
    plt.show()

    # === Diagnostics ===
    top_inputs = result['jf'].mean(0).argsort()[:10]
    print("Neural shape:", neural_tensor.shape)
    print("Top input feature indices:", top_inputs)

# === Convert to 40x40 electrode covariance map for each latent ===
num_latents = jf.shape[0]
num_electrodes = 40

for i in range(num_latents):
    jf_cov = jf[i].reshape(num_electrodes, num_electrodes)

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
    plt.title(f"Latent Dimension {i} — Electrode Attribution Covariance")
    plt.xlabel("Electrode")
    plt.ylabel("Electrode")
    plt.tight_layout()
    plt.show()
