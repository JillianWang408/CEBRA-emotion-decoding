# This code is used to aggregate single-node accuracies (both pairwise and single node, so 55 for full-node patient)
# into z-scores, so as to aggregate results across patients

import os
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import PATIENT_CONFIG, NODE_MAP

def build_full_matrix(pid: int):
    """Combine single-node + pair-node accuracies into one symmetric matrix."""
    _, patient_id = PATIENT_CONFIG[pid]
    nodes = list(NODE_MAP.keys())
    n = len(nodes)
    mat = np.full((n, n), np.nan, dtype=float)

    # --- load single-node results ---
    single_csv = Path(f"./output_xCEBRA_lags/{patient_id}/single_node_eval/summary.csv")
    if single_csv.exists():
        df_single = pd.read_csv(single_csv)
        for _, r in df_single.iterrows():
            node = str(r["node"])
            if node in nodes and pd.notna(r["acc"]):
                i = nodes.index(node)
                mat[i, i] = float(r["acc"])

    # --- load pair-node results ---
    pair_csv = Path(f"./output_xCEBRA_lags/{patient_id}/pair_nodes_eval/summary.csv")
    if pair_csv.exists():
        df_pairs = pd.read_csv(pair_csv)
        for _, r in df_pairs.iterrows():
            pair = str(r["pair"])
            if "__" not in pair or pd.isna(r["acc"]):
                continue
            node1, node2 = pair.split("__", 1)
            if node1 in nodes and node2 in nodes:
                i, j = nodes.index(node1), nodes.index(node2)
                acc = float(r["acc"])
                mat[i, j] = acc
                mat[j, i] = acc

    return nodes, mat

def main():
    pid = int(float(os.environ["PATIENT_ID"]))
    _, patient_id = PATIENT_CONFIG[pid]

    # build combined matrix
    nodes, mat = build_full_matrix(pid)

    if np.all(np.isnan(mat)):
        print(f"[warn] no accuracy results for patient {patient_id}")
        return

    # per-patient z-score
    mean, std = np.nanmean(mat), np.nanstd(mat)
    zmat = (mat - mean) / (std if std > 0 else 1)

    out_dir = Path(f"./output_xCEBRA_lags/{patient_id}/pair_nodes_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / "zscore_matrix.csv", zmat, delimiter=",")
    print(f"[ok] wrote z-score matrix for patient {patient_id} → {out_dir/'zscore_matrix.csv'}")

if __name__ == "__main__":
    main()
