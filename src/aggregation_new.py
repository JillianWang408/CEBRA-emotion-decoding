import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

patient_ids = ["238", "239", "272", "301", "304", "280", "288", "293", "PR06", "325", "326"]
N = 40
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

sum_tensor = np.zeros((5, 40, 5))
count_tensor = np.zeros((5, 40, 5))

for pid in patient_ids:
    base_path = Path("output") / pid / "attribution_outputs"
    mask_path = base_path / "electrode_mask.npy"
    mask = np.load(mask_path)      # shape (40,)
    
    for b in range(5):
        attr_path = base_path / f"attribution_band_{b}_{BAND_NAMES[b]}.npy"
        if not attr_path.exists():
            print(f"⚠️ Missing attribution map for patient {pid}, band {BAND_NAMES[b]}")
            continue
        
        attr = np.load(attr_path)  # shape: (40, 5)
        valid_attr = np.where(mask[:, None], attr, 0)
        valid_counts = np.where(mask[:, None], 1, 0)
        
        sum_tensor[b] += valid_attr
        count_tensor[b] += valid_counts

# === Average over patients (only where count > 0)
with np.errstate(divide='ignore', invalid='ignore'):
    avg_tensor = np.divide(sum_tensor, count_tensor, where=(count_tensor > 0))
    avg_tensor[np.isnan(avg_tensor)] = 0

# === Normalize so total sum is 1
output_dir = Path("output/aggregate_outputs")
output_dir.mkdir(exist_ok=True, parents=True)

avg_tensor /= avg_tensor.sum()

# === Save
output_dir = Path("output/aggregate_outputs")
output_dir.mkdir(exist_ok=True, parents=True)

np.save(output_dir / "aggregated_covariance_normalized.npy", avg_tensor)


for b in range(5):
    out_path = output_dir / f"aggregated_attribution_band_{BAND_NAMES[b]}.npy"
    np.save(out_path, avg_tensor[b])

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        avg_tensor[b],
        cmap="viridis",
        xticklabels=[f"Lag {i+1}" for i in range(5)],
        yticklabels=ELECTRODE_NAMES,
        cbar=True
    )
    plt.xlabel("Neural Lag")
    plt.ylabel("Electrode")
    plt.title(f"Aggregated Attribution — {BAND_NAMES[b]}")
    plt.tight_layout()
    plt.savefig(output_dir / f"aggregated_attribution_band_{BAND_NAMES[b]}.png")
    plt.close()

print("✅ All aggregated attribution maps saved and plotted.")