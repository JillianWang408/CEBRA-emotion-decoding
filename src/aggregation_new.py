import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

patient_ids = ["238", "239", "272", "301", "304", "280", "288", "293", "PR06", "325", "326"]
ELECTRODE_NAMES = [
    'LOFC7', 'LOFC8', 'LOFC9', 'LOFC10', 'LOFC1', 'LOFC2', 'LOFC3', 'LOFC4', 
    'ROFC1', 'ROFC2', 'ROFC3', 'ROFC4', 'ROFC7', 'ROFC8', 'ROFC9', 'ROFC10', 
    'LAD1', 'LAD2', 'LAD3', 'LAD4',
    'LINS1', 'LINS2', 'LINS3', 'LINS4',
    'LC1', 'LC2', 'LC3', 'LC4',
    'RC1', 'RC2', 'RC3', 'RC4',
    'RINS1', 'RINS2', 'RINS3', 'RINS4',
    'RAD1', 'RAD2', 'RAD3', 'RAD4'
]
BAND_NAMES = ["Theta", "Alpha", "Beta", "Low-γ", "High-γ"]
N_ELECTRODES = 40
N_BANDS = 5
N_LAGS = 5
N_LATENTS = 10

# === For latent aggregation ===
sum_latents = np.zeros((N_LATENTS, N_ELECTRODES * N_BANDS, N_LAGS))
count_latents = np.zeros_like(sum_latents)

for pid in patient_ids:
    base_path = Path("output") / pid / "attribution_outputs"
    
    # === Load electrode mask
    mask_path = base_path / "electrode_mask.npy"
    mask = np.load(mask_path)  # shape: (40,)

    # Expand mask to 200 rows (40 electrodes × 5 bands)
    mask_200 = np.repeat(mask, N_BANDS)  # shape: (200,)

    # === Load latent attribution maps
    latent_path = base_path / "all_latent_attr_maps.npy"

    latents = np.load(latent_path)  # shape: [10, 200, 5]
    
    for i in range(N_LATENTS):
        latent_map = latents[i]  # shape [200, 5]
        valid_attr = np.where(mask_200[:, None], latent_map, 0)
        valid_count = np.where(mask_200[:, None], 1, 0)

        sum_latents[i] += valid_attr
        count_latents[i] += valid_count

# === Average over patients (handle divide-by-zero)
with np.errstate(divide='ignore', invalid='ignore'):
    avg_latents = np.divide(sum_latents, count_latents, where=(count_latents > 0))
    avg_latents[np.isnan(avg_latents)] = 0

# === Summary across latents
summary_latent = np.mean(avg_latents, axis=0)  # shape: (200, 5)

# === Normalize
total_sum = np.sum(avg_latents)
avg_latents /= total_sum
summary_latent /= np.sum(summary_latent)

# === Save and Plot
output_dir = Path("output/aggregate_outputs")
output_dir.mkdir(exist_ok=True, parents=True)

np.save(output_dir / "avg_latent_attr_maps.npy", avg_latents)         # shape [10, 200, 5]
np.save(output_dir / "summary_avg_latent_attr_map.npy", summary_latent)  # shape [200, 5]

# === Plot each latent
for i in range(N_LATENTS):
    plt.figure(figsize=(10, 12))
    sns.heatmap(
        avg_latents[i],
        cmap="viridis",
        xticklabels=[f"Lag {j+1}" for j in range(N_LAGS)],
        yticklabels=[ELECTRODE_NAMES[i // 5] if i % 5 == 0 else "" for i in range(200)],
        cbar=True
    )
    plt.xlabel("Neural Lag")
    plt.ylabel("Electrode × Band (200)")
    plt.title(f"Latent {i} — Avg Attribution Across Patients")
    plt.tight_layout()
    plt.savefig(output_dir / f"avg_latent_{i}_across_patients.png")
    plt.close()

# === Plot summary
plt.figure(figsize=(10, 12))
sns.heatmap(
    summary_latent,
    cmap="viridis",
    xticklabels=[f"Lag {j+1}" for j in range(N_LAGS)],
    yticklabels=[ELECTRODE_NAMES[i // 5] if i % 5 == 0 else "" for i in range(200)],
    cbar=True
)
plt.xlabel("Neural Lag")
plt.ylabel("Electrode × Band (200)")
plt.title("Average Attribution Across All Latents and Patients")
plt.tight_layout()
plt.savefig(output_dir / "summary_avg_latent_attr_map.png")
plt.close()

print("✅ All latent-level aggregated maps saved and plotted.")