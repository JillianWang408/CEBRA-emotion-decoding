import csv
from pathlib import Path
import os
import pandas as pd
import numpy as np
import torch

from src.config import (
    MODEL_DIR, FULL_EMOTION_PATH,
    PATIENT_CONFIG, NODE_MAP
)
from src.utils_eval import evaluate_embedding, load_embedding_TxD, build_acc_matrix, plot_heatmap


def main():
    pid = int(float(os.environ["PATIENT_ID"]))
    _, patient_id = PATIENT_CONFIG[pid]

    out_root = Path(f"./output_xCEBRA_lags/{patient_id}/pair_nodes")
    eval_root = Path(f"./output_xCEBRA_lags/{patient_id}/pair_nodes_eval")
    eval_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Evaluating pairwise embeddings for patient {patient_id}")

    # labels & test split
    emotion_tensor_full = torch.load(FULL_EMOTION_PATH, map_location="cpu")
    if emotion_tensor_full.ndim == 1:
        emotion_tensor_full = emotion_tensor_full.unsqueeze(1)
    emotion_tensor_full = emotion_tensor_full.float().contiguous().numpy()
    test_idx = np.load(MODEL_DIR / "test_idx.npy")

    # summary.csv
    summary_path = eval_root / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair", "R2_behavior", "acc", "confusion_path"])
        writer.writeheader()

        for pair_dir in sorted(out_root.glob("*__*")):
            pair_name = pair_dir.name
            emb_path = pair_dir / "embedding.pt"
            if not pair_dir.exists() or not emb_path.exists():
                print(f"[SKIP] {pair_name}: missing pair embedding (likely skipped in training).")
                continue

            try:
                embedding = load_embedding_TxD(emb_path)
            except Exception as e:
                print(f"[warn] skipping {pair_name}: {e}")
                continue

            try:
                R2_behavior, acc_knn, cm_path = evaluate_embedding(
                    embedding, emotion_tensor_full, test_idx, pair_name, eval_root
                )
                writer.writerow({
                    "pair": pair_name,
                    "R2_behavior": f"{R2_behavior:.4f}",
                    "acc": f"{acc_knn:.4f}",
                    "confusion_path": str(cm_path)
                })
                print(f"[ok] {pair_name}: acc={acc_knn:.3f}, R²={R2_behavior:.3f}")
            except Exception as e:
                print(f"[error] failed {pair_name}: {e}")

    print(f"[done] wrote {summary_path}")

    # NULL baseline
    acc_null = None
    null_dir = Path(f"./output_xCEBRA_lags/{patient_id}/null_model")
    emb_path = null_dir / "embedding.pt"
    if emb_path.exists():
        try:
            embedding = load_embedding_TxD(emb_path)
            R2_behavior, acc_knn, cm_path = evaluate_embedding(
                embedding, emotion_tensor_full, test_idx, "NULL", eval_root
            )
            with summary_path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["pair", "R2_behavior", "acc", "confusion_path"])
                writer.writerow({
                    "pair": "NULL",
                    "R2_behavior": f"{R2_behavior:.4f}",
                    "acc": f"{acc_knn:.4f}",
                    "confusion_path": str(cm_path)
                })
            print(f"[ok] NULL model: acc={acc_knn:.3f}, R²={R2_behavior:.3f}")
            acc_null = acc_knn
        except Exception as e:
            print(f"[error] failed null model eval: {e}")
    else:
        print(f"[warn] no null model embedding at {emb_path}")

    df_pairs = pd.read_csv(summary_path)
    df_pairs["pair"] = df_pairs["pair"].astype(str).str.strip()

    if acc_null is None:
        null_rows = df_pairs[df_pairs["pair"].str.upper() == "NULL"]
        if len(null_rows) > 0:
            try:
                acc_null = float(null_rows.iloc[0]["acc"])
                print(f"[baseline] Using NULL acc from CSV = {acc_null:.3f}")
            except Exception as e:
                print(f"[warn] NULL acc parse failed ({e}), fallback 0.0")
                acc_null = 0.0
        else:
            print("[warn] NULL baseline not found in CSV, using 0.0")
            acc_null = 0.0

    single_csv = Path(f"./output_xCEBRA_lags/{patient_id}/single_node_eval/summary.csv")
    df_single = pd.read_csv(single_csv) if single_csv.exists() else None

    nodes = list(NODE_MAP.keys())

    mat_raw = build_acc_matrix(nodes, df_pairs, df_single=df_single, baseline=None)
    plot_heatmap(mat_raw, nodes,
        title="Pairwise Node Accuracy Heatmap",
        out_path=eval_root.parent / "summary_accuracy_heatmap.png",
        cmap="viridis", cbar_label="KNN Accuracy"
    )

    mat_delta = build_acc_matrix(nodes, df_pairs, df_single=df_single, baseline=acc_null)
    plot_heatmap(mat_delta, nodes,
        title=f"Pairwise Node Accuracy Δ vs NULL (NULL={acc_null:.2f})",
        out_path=eval_root.parent / "summary_accuracy_heatmap_NULL_Baseline.png",
        cmap="RdBu", center=0, cbar_label="Δ KNN Accuracy vs NULL"
    )


if __name__ == "__main__":
    main()
