# RUN WITH: python -m src.aggregate_covariances

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# === Specify patients to include ===
#id = ["238", "239", "272", "301", "304"]
id = ["239", "272", "304"]
#id = ["238", "239", "272", "280", "288", "293", "301", "304", "325", "326"]
n_electrodes = 40
electorde_names =  ['LOFC7', 'LOFC8', 'LOFC9', 'LOFC10', 'LOFC1', 'LOFC2', 'LOFC3', 'LOFC4', 
    'ROFC1', 'ROFC2', 'ROFC3', 'ROFC4', 'ROFC7', 'ROFC8', 'ROFC9', 'ROFC10', 
    'LAD1', 'LAD2', 'LAD3', 'LAD4',
    'LINS1', 'LINS2', 'LINS3', 'LINS4',
    'LC1', 'LC2', 'LC3', 'LC4',
    'RC1', 'RC2', 'RC3', 'RC4',
    'RINS1', 'RINS2', 'RINS3', 'RINS4',
    'RAD1', 'RAD2', 'RAD3', 'RAD4'
]

# === Aggregate matrices ===
sum_matrix = np.zeros((n_electrodes, n_electrodes))

for pid in id:
    matrix_path = Path("output") / pid / "attribution_outputs" / "electrode_covariance_summary_all10.npy"
    if matrix_path.exists():
        cov = np.load(matrix_path)
        sum_matrix += cov
    else:
        print(f"Missing file for {pid}: {matrix_path}")

# === Normalize aggregated matrix ===
sum_matrix_normalized = sum_matrix / sum_matrix.sum()

# === Save and plot ===
output_dir = Path("output/aggregate_outputs")
output_dir.mkdir(parents=True, exist_ok=True)

np.save(output_dir / "aggregated_covariance_normalized.npy", sum_matrix_normalized)

plt.figure(figsize=(10, 8))
sns.heatmap(
    sum_matrix_normalized,
    cmap='viridis',
    square=True,
    xticklabels=electorde_names,
    yticklabels=electorde_names,
    cbar=True
)
plt.xticks(rotation=90)
plt.title("Aggregated Normalized Electrode Attribution Covariance")
plt.xlabel("Electrode")
plt.ylabel("Electrode")
plt.tight_layout()
plt.savefig(output_dir / "aggregated_covariance_normalized.png")
plt.close()

print("Aggregated covariance matrix saved and plotted.")
