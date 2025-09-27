import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os

from src.config import PATIENT_CONFIG, NODE_MAP
from src.utils_eval import evaluate_node  # <- still external wrapper, unchanged


def run_single_node_eval(pid: int, base_dir="./output_xCEBRA_lags"):
    _, alt = PATIENT_CONFIG[pid]
    out_dir = Path(base_dir) / str(alt) / "single_node_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, accs, nodes = [], [], []

    for node in NODE_MAP.keys():
        # path where training should have written embedding
        _, alt = PATIENT_CONFIG[pid]
        train_node_dir = Path(f"./output_xCEBRA_lags/{alt}/single_node/{node}")
        emb_path = train_node_dir / "embedding.pt"

        if not emb_path.exists():
            print(f"[SKIP] {node}: no trained embedding found, skipping eval.")
            continue  # don’t create eval folder

        # only make eval folder if node actually trained
        node_eval_dir = out_dir / node
        node_eval_dir.mkdir(parents=True, exist_ok=True)

        acc, r2, n_test = evaluate_node(pid, node, node_eval_dir)
        rows.append({"node": node, "acc": acc, "R2_behavior": r2, "n_test": n_test})
        nodes.append(node)
        accs.append(acc)


    # summary.csv
    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # bar plot
    fig, ax = plt.subplots(figsize=(max(8, 0.5*len(nodes)), 4))
    ax.bar(nodes, accs)
    ax.set_xticklabels(nodes, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Single-node accuracy for patient {alt}")
    fig.tight_layout()
    fig.savefig(out_dir / "nodes_accuracy.png", dpi=200)
    plt.close(fig)

    print(f"[ok] wrote {out_dir/'summary.csv'} and nodes_accuracy.png")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", type=int, required=False,
                    help="Patient ID (optional, otherwise read from PATIENT_ID env)")
    ap.add_argument("--feature-mode", choices=["lags", "cov"], default="lags",
                    help="(ignored here, only for pipeline consistency)")
    args = ap.parse_args()

    # Priority: explicit --patient > env variable
    if args.patient is not None:
        pid = args.patient
    else:
        pid = int(float(os.environ["PATIENT_ID"]))

    run_single_node_eval(pid)

