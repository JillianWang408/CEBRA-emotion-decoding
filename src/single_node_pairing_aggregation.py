# This code is used to aggregate single-node accuracies (both pairwise and single node, so 55 for full-node patient)
# into z-scores, so as to aggregate results across patients
# actual aggregation and plotting is done in main.py

import os
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import PATIENT_CONFIG, NODE_MAP


def safe_get(r, key, fallback=np.nan):
    """Safely fetch numeric value from a row."""
    try:
        val = float(r[key])
        return val if not np.isnan(val) else fallback
    except Exception:
        return fallback


def build_full_matrix(pid: int):
    """Combine single-node + pair-node accuracies into one symmetric matrix."""
    _, patient_id = PATIENT_CONFIG[pid]
    nodes = list(NODE_MAP.keys())
    n = len(nodes)
    mat = np.full((n, n), np.nan, dtype=float)

    # --- load single-node results ---
    single_csv = Path(f"./output_xCEBRA_lags/{patient_id}/single_node_eval_new/summary.csv")
    if single_csv.exists():
        df_single = pd.read_csv(single_csv)
        for _, r in df_single.iterrows():
            node = str(r["node"])
            if node in nodes and pd.notna(r.get("acc_logreg", np.nan)):
                i = nodes.index(node)
                mat[i, i] = safe_get(r, "acc_logreg")

    # --- load pair-node results ---
    pair_csv = Path(f"./output_xCEBRA_lags/{patient_id}/pair_nodes_eval_new/summary.csv")
    if pair_csv.exists():
        df_pairs = pd.read_csv(pair_csv)
        for _, r in df_pairs.iterrows():
            pair = str(r["pair"])
            if "__" not in pair or pd.isna(r.get("acc_logreg", np.nan)):
                continue
            node1, node2 = pair.split("__", 1)
            if node1 in nodes and node2 in nodes:
                i, j = nodes.index(node1), nodes.index(node2)
                acc = safe_get(r, "acc_logreg")
                mat[i, j] = acc
                mat[j, i] = acc

    return nodes, mat


def main():
    pid = int(float(os.environ["PATIENT_ID"]))
    _, patient_id = PATIENT_CONFIG[pid]

    nodes, mat = build_full_matrix(pid)

    if np.all(np.isnan(mat)):
        print(f"[warn] no accuracy results for patient {patient_id}")
        return

    mean, std = np.nanmean(mat), np.nanstd(mat)
    zmat = np.zeros_like(mat) if std == 0 else (mat - mean) / std

    out_dir = Path(f"./output_xCEBRA_lags/{patient_id}/pair_nodes_eval_new")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save both plain and labeled versions
    np.savetxt(out_dir / "zscore_matrix.csv", zmat, delimiter=",")
    pd.DataFrame(zmat, index=nodes, columns=nodes).to_csv(out_dir / "zscore_matrix_labeled.csv")

    print(f"[ok] wrote z-score matrix for patient {patient_id} → {out_dir/'zscore_matrix_labeled.csv'}")


if __name__ == "__main__":
    main()
