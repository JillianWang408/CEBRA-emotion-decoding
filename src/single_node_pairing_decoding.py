import csv
from pathlib import Path
import os
import pandas as pd
import numpy as np
import torch
import mat73

from src.config import (
    MODEL_DIR, FULL_EMOTION_PATH,
    PATIENT_CONFIG, NODE_MAP, NEURAL_PATH
)
from src.utils_eval import evaluate_embedding, load_embedding_TxD, build_acc_matrix, plot_heatmap


def main():
    pid = int(float(os.environ["PATIENT_ID"]))
    _, patient_id = PATIENT_CONFIG[pid]

    out_root = Path(f"./output_xCEBRA_lags/{patient_id}/pair_nodes")
    eval_root = Path(f"./output_xCEBRA_lags/{patient_id}/pair_nodes_eval_new")
    eval_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Evaluating pairwise embeddings for patient {patient_id}")

    # labels & test split
    emotion_tensor_full = torch.load(FULL_EMOTION_PATH, map_location="cpu")
    if emotion_tensor_full.ndim == 1:
        emotion_tensor_full = emotion_tensor_full.unsqueeze(1)
    emotion_tensor_full = emotion_tensor_full.float().contiguous().numpy()
    test_idx = np.load(MODEL_DIR / "test_idx.npy")

    # neural = mat73.loadmat(NEURAL_PATH)["stim"].T
    # emotion = torch.load(FULL_EMOTION_PATH)

    # summary.csv
    summary_path = eval_root / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair", "R2_behavior", "acc_knn", "acc_logreg"])
        writer.writeheader()

        for pair_dir in sorted(out_root.glob("*__*")):
            pair_name = pair_dir.name
            emb_path = pair_dir / "embedding.pt"

            #embedding = torch.load(emb_path).squeeze(0).T
            # print("Neural:", neural.shape[0])
            # print("Embedding:", embedding.shape[0])
            # print("Labels:", len(emotion))

            if not pair_dir.exists() or not emb_path.exists():
                print(f"[SKIP] {pair_name}: missing pair embedding (likely skipped in training).")
                continue

            try:
                embedding = load_embedding_TxD(emb_path)
            except Exception as e:
                print(f"[warn] skipping {pair_name}: {e}")
                continue

            try:
                R2_behavior, acc_knn, acc_logreg = evaluate_embedding(
                    embedding=embedding, y=emotion_tensor_full, pair_name=pair_name,out_dir=eval_root
                )
                writer.writerow({
                    "pair": pair_name,
                    "R2_behavior": f"{R2_behavior:.4f}",
                    "acc_knn": f"{acc_knn:.4f}",
                    "acc_logreg": f"{acc_logreg:.4f}"
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
            R2_behavior, acc_knn, acc_logreg = evaluate_embedding(
                embedding=embedding, y=emotion_tensor_full, pair_name="NULL", out_dir=eval_root
            )
            with summary_path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["pair", "R2_behavior", "acc_knn", "acc_logreg"])
                writer.writerow({
                    "pair": "NULL",
                    "R2_behavior": f"{R2_behavior:.4f}",
                    "acc_knn": f"{acc_knn:.4f}",
                    "acc_logreg": f"{acc_logreg:.4f}"
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

    single_csv = Path(f"./output_xCEBRA_lags/{patient_id}/single_node_eval_new/summary.csv")
    df_single = pd.read_csv(single_csv) if single_csv.exists() else None

    nodes = list(NODE_MAP.keys())

    mat_raw = build_acc_matrix(nodes, df_pairs, df_single=df_single, baseline=None, metric="acc_logreg")
    plot_heatmap(mat_raw, nodes,
        title="Pairwise Node Accuracy Heatmap",
        out_path=eval_root.parent / "summary_accuracy_heatmap.png",
        cmap="viridis", cbar_label="LogReg Accuracy"
    )

    mat_delta = build_acc_matrix(nodes, df_pairs, df_single=df_single, baseline=acc_null, metric="acc_logreg")
    plot_heatmap(mat_delta, nodes,
        title=f"Pairwise Node Accuracy Δ vs NULL (NULL={acc_null:.2f})",
        out_path=eval_root.parent / "summary_accuracy_heatmap_NULL_Baseline.png",
        cmap="RdBu", center=0, cbar_label="Δ LogReg Accuracy vs NULL"
    )


if __name__ == "__main__":
    main()
