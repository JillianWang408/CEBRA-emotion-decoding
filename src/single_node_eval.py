# src/eval_single_nodes_all_patients.py
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple, DefaultDict
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import PATIENT_CONFIG  # {pid: (code, alt)}

AGG_OUT_ROOT = Path("./output_xCEBRA_lags/_aggregate_single_node")  # aggregate outputs here
AGG_OUT_ROOT.mkdir(parents=True, exist_ok=True)

def _read_patient_summary(alt: str) -> List[Dict]:
    """
    Read one patient's per-node summary.csv produced by eval_single_nodes_patient.py.
    Returns a list of dict rows. If not present, returns [].
    """
    p = Path(f"./output_xCEBRA_lags/{alt}/single_node_eval/summary.csv")
    if not p.exists():
        print(f"[warn] missing summary.csv for patient {alt} at {p}")
        return []
    rows: List[Dict] = []
    with p.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            # coerce numeric fields
            for k in ["acc", "R2_behavior", "n_test", "embedding_len", "test_idx_len_used"]:
                if k in row and row[k] != "":
                    row[k] = float(row[k]) if k in ("acc", "R2_behavior") else int(float(row[k]))
            rows.append(row)
    return rows

def _mean_std_sem(values: List[float]) -> Tuple[float, float, float]:
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(np.mean(arr))
    s = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    sem = s / math.sqrt(len(arr)) if len(arr) > 0 else float("nan")
    return m, s, sem

def main():
    # --- config knobs ---
    intersection_only = False  # If True: only plot nodes present for all patients

    # --- load all patient summaries ---
    patient_nodes: Dict[str, List[Dict]] = {}  # alt -> rows
    all_nodes_set = set()
    patient_count = 0

    for pid, (code, alt) in PATIENT_CONFIG.items():
        rows = _read_patient_summary(alt)
        if not rows:
            continue
        patient_nodes[alt] = rows
        all_nodes_set.update(r["node"] for r in rows)
        patient_count += 1

    if patient_count == 0:
        print("[error] No patient summaries found. Run eval_single_nodes_patient.py for each patient first.")
        return

    # --- decide node universe ---
    if intersection_only:
        # nodes that appear in every included patient
        node_counts: DefaultDict[str, int] = defaultdict(int)
        for alt, rows in patient_nodes.items():
            seen = set(r["node"] for r in rows)
            for n in seen:
                node_counts[n] += 1
        nodes = sorted([n for n, c in node_counts.items() if c == len(patient_nodes)])
        print(f"[info] Using intersection of nodes across patients: {len(nodes)} nodes")
    else:
        nodes = sorted(all_nodes_set)
        print(f"[info] Using union of nodes across patients: {len(nodes)} nodes (patients may contribute unevenly)")

    # --- aggregate metrics per node ---
    per_node_acc: DefaultDict[str, List[float]] = defaultdict(list)
    per_node_r2: DefaultDict[str, List[float]] = defaultdict(list)
    per_node_contrib: DefaultDict[str, List[str]] = defaultdict(list)  # which patients contributed

    # Also collect overall lists for global averages
    all_acc_values: List[float] = []
    all_r2_values: List[float] = []

    for alt, rows in patient_nodes.items():
        by_node = {r["node"]: r for r in rows}
        for node in nodes:
            if node not in by_node:
                # skip if this patient doesn't have the node and we're doing union
                if not intersection_only:
                    continue
                # if intersection_only, node wouldn't be here
            else:
                acc = float(by_node[node]["acc"])
                r2  = float(by_node[node]["R2_behavior"])
                per_node_acc[node].append(acc)
                per_node_r2[node].append(r2)
                per_node_contrib[node].append(alt)
                all_acc_values.append(acc)
                all_r2_values.append(r2)

    # --- compute stats and write CSV ---
    agg_csv = AGG_OUT_ROOT / ("aggregate_nodes_intersection.csv" if intersection_only else "aggregate_nodes_union.csv")
    fieldnames = [
        "node", "n_patients_node",
        "mean_acc", "std_acc", "sem_acc",
        "mean_R2_behavior", "std_R2_behavior", "sem_R2_behavior",
        "patients_contributed"
    ]
    with agg_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for node in nodes:
            acc_list = per_node_acc.get(node, [])
            r2_list  = per_node_r2.get(node, [])
            mean_acc, std_acc, sem_acc = _mean_std_sem(acc_list)
            mean_r2,  std_r2,  sem_r2  = _mean_std_sem(r2_list)
            w.writerow({
                "node": node,
                "n_patients_node": len(acc_list),
                "mean_acc": f"{mean_acc:.6f}" if not math.isnan(mean_acc) else "",
                "std_acc":  f"{std_acc:.6f}"  if not math.isnan(std_acc)  else "",
                "sem_acc":  f"{sem_acc:.6f}"  if not math.isnan(sem_acc)  else "",
                "mean_R2_behavior": f"{mean_r2:.6f}" if not math.isnan(mean_r2) else "",
                "std_R2_behavior":  f"{std_r2:.6f}"  if not math.isnan(std_r2)  else "",
                "sem_R2_behavior":  f"{sem_r2:.6f}"  if not math.isnan(sem_r2)  else "",
                "patients_contributed": ",".join(sorted(per_node_contrib.get(node, []))),
            })
    print(f"[ok] wrote {agg_csv}")

    # --- plot mean ± SEM accuracy across nodes ---
    # Only plot nodes with at least 1 patient contributing
    plot_nodes = [n for n in nodes if len(per_node_acc.get(n, [])) > 0]
    means = [float(np.mean(per_node_acc[n])) for n in plot_nodes]
    sems  = [float(np.std(per_node_acc[n], ddof=1)) / math.sqrt(len(per_node_acc[n]))
             if len(per_node_acc[n]) > 1 else 0.0 for n in plot_nodes]

    x = np.arange(len(plot_nodes))
    fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(plot_nodes)), 5))
    ax.errorbar(x, means, yerr=sems, fmt="-o", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_nodes, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("KNN accuracy (mean ± SEM across patients)")
    ax.set_title("Single-node performance — patient-averaged")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    out_png = AGG_OUT_ROOT / ("nodes_accuracy_mean_sem_intersection.png" if intersection_only
                              else "nodes_accuracy_mean_sem_union.png")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] saved {out_png}")

    # --- global (across all nodes/patients) quick stats ---
    g_mean_acc, g_std_acc, g_sem_acc = _mean_std_sem(all_acc_values)
    g_mean_r2,  g_std_r2,  g_sem_r2  = _mean_std_sem(all_r2_values)
    print("[overall] across all (node, patient) pairs used:")
    print(f"  acc: mean={g_mean_acc:.4f} std={g_std_acc:.4f} sem={g_sem_acc:.4f} n={len(all_acc_values)}")
    print(f"  R² : mean={g_mean_r2:.4f} std={g_std_r2:.4f} sem={g_sem_r2:.4f} n={len(all_r2_values)}")

if __name__ == "__main__":
    main()
