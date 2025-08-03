import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

patient_ids = ["239", "272", "304"]  # or however many you have
N = 40

sum_matrix = np.zeros((N, N))
count_matrix = np.zeros((N, N))

for pid in patient_ids:
    base_path = Path("output") / pid / "attribution_outputs"
    cov_path = base_path / "electrode_covariance_summary_all10.npy"
    mask_path = base_path / "electrode_mask.npy"
    
    if not cov_path.exists() or not mask_path.exists():
        print(f"Missing data for patient {pid}")
        continue

    cov = np.load(cov_path)        # shape (40, 40)
    mask = np.load(mask_path)      # shape (40,)
    pair_mask = np.outer(mask, mask)  # shape (40, 40)

    sum_matrix += np.where(pair_mask, cov, 0)
    count_matrix += pair_mask

# === Average over patients (only where count > 0)
with np.errstate(divide='ignore', invalid='ignore'):
    summary_cov = np.divide(sum_matrix, count_matrix, where=(count_matrix > 0))
    summary_cov[np.isnan(summary_cov)] = 0

# === Normalize so total sum is 1
summary_cov /= summary_cov.sum()

# === Save
output_dir = Path("output/aggregate_outputs")
output_dir.mkdir(exist_ok=True, parents=True)

np.save(output_dir / "aggregated_covariance_normalized.npy", summary_cov)

# === Plot
plt.figure(figsize=(10, 8))
sns.heatmap(summary_cov, cmap="viridis", square=True, cbar=True)
plt.title("Aggregated Normalized Pairwise Electrode Attribution")
plt.xlabel("Electrode")
plt.ylabel("Electrode")
plt.tight_layout()
plt.savefig(output_dir / "aggregated_covariance_normalized.png")
plt.close()

print("✅ Aggregated summary saved and plotted.")
