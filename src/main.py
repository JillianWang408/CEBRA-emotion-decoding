import os
import argparse
import subprocess
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


# Default patients
# Default patients
PATIENT_IDS = [1, 2, 9, 27, 28, 15, 22, 24, 29, 30, 31] #2(239) not working well
#PATIENT_IDS = [1, 9, 27, 28]
#PATIENT_IDS = [29, 30, 31]

PIPELINE_SCRIPTS = [
    #"single_node_encoding",
    #"single_node_decoding",
    #"single_node_pairing_encoding",
    #"single_node_pairing_decoding",
    "single_node_pairing_aggregation",
    #"OFC_training",
    #"evaluate_supervised"
]


def run_pipeline_for_patient(pid: int):
    env = os.environ.copy()
    env["PATIENT_ID"] = str(pid)

    print(f"\n====================== Running pipeline for Patient {pid} ======================")
    for script in PIPELINE_SCRIPTS:
        print(f"\n--- Running {script}.py for Patient {pid} ---")
        cmd = ["caffeinate", "-dimsu", "python", "-m", f"src.{script}"]
        print("[CMD]", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to run {script}.py for Patient {pid}: {e}")
            return


def aggregate_group(patients):
    """Aggregate per-patient z-scores into a group-level mean z-score matrix and plot heatmap."""
    zmats = []
    for pid in patients:
        _, patient_id = PATIENT_CONFIG[pid]
        zpath = Path(f"./output_xCEBRA_lags/{patient_id}/pair_nodes_eval_new/zscore_matrix.csv")
        if not zpath.exists():
            print(f"[warn] no zscore_matrix.csv for patient {patient_id}, skipping")
            continue
        zmats.append(np.loadtxt(zpath, delimiter=","))

    if not zmats:
        print("[warn] no patient z-score matrices found, skipping group aggregation")
        return

    # average across patients
    group_z = np.nanmean(zmats, axis=0)
    group_out = Path("./output_xCEBRA_lags/aggregate_outputs_new/aggregate_pair")
    group_out.mkdir(parents=True, exist_ok=True)

    # save CSV
    out_path = group_out / "group_zscores.csv"
    np.savetxt(out_path, group_z, delimiter=",")
    print(f"[ok] saved group z-scores → {out_path}")

    # plot heatmap
    nodes = list(NODE_MAP.keys())
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        group_z, annot=True, fmt=".2f",
        xticklabels=nodes, yticklabels=nodes,
        cmap="RdBu_r", center=0,
        cbar_kws={"label": "Z-scored Accuracy"}
    )
    ax.set_title("Group-level Aggregated Z-scored Accuracy")
    plt.tight_layout()
    heatmap_path = group_out / "group_zscores_heatmap.png"
    plt.savefig(heatmap_path, dpi=220)
    plt.close(fig)
    print(f"[ok] saved group heatmap → {heatmap_path}")



PATIENT_CONFIG = {
    1:    ("EC238", "238"),
    2:    ("EC239", "239"),
    9:    ("EC272", "272"),
    27:    ("EC301", "301"),
    28:    ("EC304", "304"),

    15: ("EC280", "280"), #280 noisiest
    22: ("EC288", "288"),
    24: ("EC293", "293"),
    29: ("PR06", "PR06"),
    30: ("EC325", "325"),
    31: ("EC326", "326"),
}

NODE_MAP = {
    # Orbitofrontal Cortex — Left (LOFC)
    "LOFC_Medial":  ["LOFC1", "LOFC2", "LOFC3", "LOFC4"],
    "LOFC_Lateral": ["LOFC7", "LOFC8", "LOFC9", "LOFC10"],

    # Orbitofrontal Cortex — Right (ROFC)
    "ROFC_Medial":  ["ROFC1", "ROFC2", "ROFC3", "ROFC4"],
    "ROFC_Lateral": ["ROFC7", "ROFC8", "ROFC9", "ROFC10"],

    # Anterior Dorsal Cingulate
    "LAD": ["LAD1", "LAD2", "LAD3", "LAD4"],
    "RAD": ["RAD1", "RAD2", "RAD3", "RAD4"],

    # Insula
    "LINS": ["LINS1", "LINS2", "LINS3", "LINS4"],
    "RINS": ["RINS1", "RINS2", "RINS3", "RINS4"],

    # Cingulate
    "LC": ["LC1", "LC2", "LC3", "LC4"],
    "RC": ["RC1", "RC2", "RC3", "RC4"],
}

def main():
    parser = argparse.ArgumentParser(description="Run xCEBRA training/eval pipeline")
    parser.add_argument("--patients", nargs="*", type=int, default=PATIENT_IDS,
                        help="Patient IDs to process (space-separated)")
    args = parser.parse_args()

    for pid in args.patients:
        run_pipeline_for_patient(pid)

    if ("single_node_pairing_aggregation" in PIPELINE_SCRIPTS) and len(args.patients) > 1:
        aggregate_group(args.patients)

if __name__ == "__main__":
    main()
