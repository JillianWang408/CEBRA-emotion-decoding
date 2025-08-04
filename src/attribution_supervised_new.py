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
    jf = result["jf"]  # shape: [latents, 1000, time]
    jf_mean = np.abs(jf).mean(axis=(0, 1))

    # === Reshape to [40, 5, 5] = [electrode, lag, band]
    attr_tensor = jf_mean.reshape(5, N_ELECTRODES, 5).transpose(1, 0, 2)  # → shape: [electrode, lag, band]

    band_names = ["Theta", "Alpha", "Beta", "Low-γ", "High-γ"]

    for b in range(5):
        band_attr = attr_tensor[:, :, b]  # [electrode, lag]
        masked_band_attr = np.where(electrode_mask[:, None], band_attr, np.nan)
        
        np.save(ATTRIBUTION_OUTPUT_DIR / f"attribution_band_{b}_{band_names[b]}.npy", band_attr)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            masked_band_attr,
            cmap="viridis",
            xticklabels=[f"Lag {i+1}" for i in range(5)],
            yticklabels=ELECTRODE_NAMES,
            cbar=True,
            mask=np.isnan(masked_band_attr)
        )
        plt.xlabel("Neural Lag (seconds before emotion)")
        plt.ylabel("Electrode")
        plt.title(f"Attribution Map: Electrode × Time Lag — {band_names[b]}")
        plt.tight_layout()
        plt.savefig(ATTRIBUTION_OUTPUT_DIR / f"attribution_lag_band_{b}_{band_names[b]}.png")
    plt.close()

    attr_40x5 = attr_tensor.mean(axis=2)  # [electrode, lag]
    masked_attr = np.where(electrode_mask[:, None], attr_40x5, np.nan)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        masked_attr,
        cmap="viridis",
        xticklabels=[f"Lag {i+1}" for i in range(5)],
        yticklabels=ELECTRODE_NAMES,
        cbar=True,
        mask=np.isnan(masked_attr)
    )
    plt.xlabel("Neural Lag (seconds before emotion)")
    plt.ylabel("Electrode")
    plt.title("Attribution Map: Electrode × Time Lag (Avg across bands)")
    plt.tight_layout()
    plt.savefig(ATTRIBUTION_OUTPUT_DIR / "attribution_electrode_lag_summary.png")
    plt.close()

    # === Diagnostics
    print("Attribution analysis complete.")
    print("Neural shape:", neural_tensor.shape)
    print("Active electrodes:", np.count_nonzero(electrode_mask), "/", N_ELECTRODES)
    print("Attribution shape:", attr_40x5.shape)

if __name__ == "__main__":
    model = load_fixed_cebra_model(MODEL_WEIGHTS_PATH, num_output=N_LATENTS)
    compute_and_plot_attribution(model)
