import os
import csv
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.config import PATIENT_CONFIG, NODE_MAP
from src.utils_eval import evaluate_node


def run_single_node_eval(pid: int, base_dir: Path):
    _, patient_id = PATIENT_CONFIG[pid]
    out_dir = Path(base_dir) / patient_id / "single_node_eval_new"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, nodes, acc_logreg = [], [], []

    for node in NODE_MAP.keys():
        R2_behavior, acc_knn, acc_logreg_val = evaluate_node(pid, node, base_dir)
        if R2_behavior is None:
            continue

        rows.append({
            "node": node,
            "R2_behavior": f"{R2_behavior:.4f}",
            "acc_knn": f"{acc_knn:.4f}",
            "acc_logreg": f"{acc_logreg_val:.4f}"
        })
        nodes.append(node)
        acc_logreg.append(acc_logreg_val)
        print(f"[ok] {node}: LogReg={acc_logreg_val:.3f}, KNN={acc_knn:.3f}, R²={R2_behavior:.3f}")

    # --- Write summary ---
    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["node", "R2_behavior", "acc_knn", "acc_logreg"])
        writer.writeheader()
        writer.writerows(rows)

    # # --- Simple bar plot (default: LogReg accuracy) ---
    # fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(nodes)), 4))
    # ax.bar(nodes, acc_logreg)
    # ax.set_xticklabels(nodes, rotation=45, ha="right")
    # ax.set_ylabel("Logistic Regression Accuracy")
    # ax.set_title(f"Single-node decoding accuracy for patient {patient_id}")
    # fig.tight_layout()
    # fig.savefig(out_dir / "nodes_accuracy.png", dpi=200)
    # plt.close(fig)

    # print(f"[done] wrote {summary_path} and nodes_accuracy.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", type=int, help="Patient index in PATIENT_CONFIG (optional)")
    ap.add_argument("--base_dir", type=str, default="./output_xCEBRA_lags",
                    help="Base directory for experiment outputs")
    args = ap.parse_args()

    pid = args.patient if args.patient is not None else int(float(os.environ["PATIENT_ID"]))
    run_single_node_eval(pid, Path(args.base_dir))