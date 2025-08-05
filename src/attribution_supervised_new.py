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
    
    print(neural_tensor.shape)

    assert neural_tensor.shape[1] == 1000, "Expected neural input to have 1000 features (40×5×5)."

    # === Compute electrode mask across all lags and bands
    neural_np = neural_tensor.detach().cpu().numpy()  # [T, 1000]
    electrode_mask = []

    for e in range(N_ELECTRODES):
        indices = []
        for lag in range(5):  # 5 lags
            base = lag * 200
            start = base + e * 5
            end = start + 5  # 5 bands
            indices.extend(range(start, end))
        segment = neural_np[:, indices]  # shape: [T, 25]
        is_active = np.any(np.abs(segment) > 1e-6)
        electrode_mask.append(is_active)

    electrode_mask = np.array(electrode_mask)  # shape: [40]
    np.save(ATTRIBUTION_OUTPUT_DIR / "electrode_mask.npy", electrode_mask)

    # === Initialize attribution method ===
    model.split_outputs = False
    method = cebra.attribution.init(
        name="jacobian-based",
        model=model,
        input_data=neural_tensor,
        output_dimension=model.num_output
    )
    result = method.compute_attribution_map()

    # === Compute average attribution over latents and time
    jf = result["jf"]  # shape: (time, latents, 1000)
    jf_mean = np.abs(jf).mean(axis=(0)) # [10, 1000]

    # === Process each latent → [40, 5, 5] → [200, 5]
    latent_maps = []
    for i in range(N_LATENTS):
        latent_attr = jf_mean[i].reshape(5, N_ELECTRODES, 5).transpose(1, 0, 2)  # [40, 5, 5]
        latent_attr_200x5 = latent_attr.reshape(N_ELECTRODES * 5, 5)  # [200, 5]

        # Mask: expand electrode_mask [40] → [200]
        mask_per_band = np.repeat(electrode_mask, 5)  # [200]
        masked_attr = np.where(mask_per_band[:, None], latent_attr_200x5, np.nan)
        latent_maps.append(masked_attr)

        # === Plot this latent
        plt.figure(figsize=(10, 12))
        sns.heatmap(
            masked_attr,
            cmap="viridis",
            xticklabels=[f"Lag {i+1}" for i in range(5)],
            yticklabels=[ELECTRODE_NAMES[i // 5] if i % 5 == 0 else "" for i in range(200)],
            cbar=True,
            mask=np.isnan(masked_attr)
        )
        plt.xlabel("Time Lags")
        plt.ylabel("Electrode × Band")
        plt.title(f"Latent {i} Attribution Map (200×5)")
        plt.tight_layout()
        plt.savefig(ATTRIBUTION_OUTPUT_DIR / f"latent_{i}_attr_map.png")
        plt.close()

    # === Average across latents
    latent_stack = np.stack(latent_maps)  # shape: [10, 200, 5]
    avg_latent_map = np.nanmean(latent_stack, axis=0)  # shape: [200, 5]

    plt.figure(figsize=(10, 12))
    sns.heatmap(
        avg_latent_map,
        cmap="viridis",
        xticklabels=[f"Lag {i+1}" for i in range(5)],
        yticklabels=[ELECTRODE_NAMES[i // 5] if i % 5 == 0 else "" for i in range(200)],
        cbar=True
    )
    plt.xlabel("Time Lags")
    plt.ylabel("Electrode × Band")
    plt.title("Average Attribution Map Across Latents")
    plt.tight_layout()
    plt.savefig(ATTRIBUTION_OUTPUT_DIR / "avg_latent_attr_map.png")
    plt.close()

    # === Save numpy files
    np.save(ATTRIBUTION_OUTPUT_DIR / "all_latent_attr_maps.npy", latent_stack)  # [10, 200, 5]
    np.save(ATTRIBUTION_OUTPUT_DIR / "avg_latent_attr_map.npy", avg_latent_map)

    print("Attribution analysis complete.")
    print("Latent attribution shape:", latent_stack.shape)

    # === Diagnostics
    print("Attribution analysis complete.")
    print("Neural shape:", neural_tensor.shape)
    print("Active electrodes:", np.count_nonzero(electrode_mask), "/", N_ELECTRODES)

if __name__ == "__main__":
    model = load_fixed_cebra_model(MODEL_WEIGHTS_PATH, num_output=N_LATENTS)
    compute_and_plot_attribution(model)
