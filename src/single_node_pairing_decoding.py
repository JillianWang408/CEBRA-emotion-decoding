import csv
from pathlib import Path
import os
import pandas as pd
import numpy as np
import torch

from src.config import (
    MODEL_DIR, FULL_EMOTION_PATH,
    PATIENT_CONFIG, NODE_MAP, EMOTION_MAP
)
from src.utils_eval import (
    evaluate_embedding, load_embedding_TxD,
    build_acc_matrix, plot_heatmap,
    collect_decoding_timecourse, save_decoding_timecourse,
    plot_decoding_timecourses
)


def main():
    pid = int(float(os.environ["PATIENT_ID"]))
    _, patient_id = PATIENT_CONFIG[pid]

    out_root = Path(f"./output_xCEBRA_lags/{patient_id}/pair_nodes")
    eval_root = Path(f"./output_xCEBRA_lags/{patient_id}/pair_nodes_eval_new")
    eval_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Evaluating pairwise embeddings for patient {patient_id}")

    emotion_tensor_full = torch.load(FULL_EMOTION_PATH, map_location="cpu")
    if emotion_tensor_full.ndim == 1:
        emotion_tensor_full = emotion_tensor_full.unsqueeze(1)
    emotion_tensor_full = emotion_tensor_full.float().contiguous().numpy()

    test_idx = np.load(MODEL_DIR / "test_idx.npy")
    print(f"[INFO] Loaded labels shape={emotion_tensor_full.shape}, test_idx={len(test_idx)} samples")

    # containers
    all_timecourse = []
    summary_path = eval_root / "summary.csv"

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair", "R2_behavior", "acc_knn", "acc_logreg"])
        writer.writeheader()

        # MAIN LOOP — Evaluate all pair embeddings
        for pair_dir in sorted(out_root.glob("*__*")):
            pair_name = pair_dir.name
            emb_path = pair_dir / "embedding.pt"

            if not emb_path.exists():
                print(f"[SKIP] {pair_name}: missing embedding.")
                continue

            try:
                embedding = load_embedding_TxD(emb_path)
                #print(f"[INFO] Loaded {pair_name} embedding shape={embedding.shape}")
            except Exception as e:
                print(f"[warn] skipping {pair_name}: {e}")
                continue

            try:
                R2_behavior, acc_knn, acc_logreg, y_pred, y_true, test_idx_local = evaluate_embedding(
                    embedding=embedding,
                    y=emotion_tensor_full,
                    pair_name=pair_name,
                    out_dir=eval_root,
                    return_predictions=True,
                    test_idx=test_idx  # ensure only test subset is evaluated
                )

                # Collect timecourse
                df_pair = collect_decoding_timecourse(pair_name, y_true, y_pred, test_idx_local)
                all_timecourse.append(df_pair)

                # Write summary row
                writer.writerow({
                    "pair": pair_name,
                    "R2_behavior": f"{R2_behavior:.4f}",
                    "acc_knn": f"{acc_knn:.4f}",
                    "acc_logreg": f"{acc_logreg:.4f}"
                })
                print(f"[ok] {pair_name}: acc={acc_knn:.3f}, R²={R2_behavior:.3f}")

            except Exception as e:
                print(f"[error] failed {pair_name}: {e}")

    # NULL model evaluation
    null_dir = Path(f"./output_xCEBRA_lags/{patient_id}/null_model")
    null_emb_path = null_dir / "embedding.pt"

    if null_emb_path.exists():
        try:
            embedding = load_embedding_TxD(null_emb_path)
            R2_behavior, acc_knn, acc_logreg, y_pred, y_true, test_idx_local = evaluate_embedding(
                embedding=embedding,
                y=emotion_tensor_full,
                pair_name="NULL",
                out_dir=eval_root,
                return_predictions=True,
                test_idx=test_idx  # same test split
            )

            with summary_path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["pair", "R2_behavior", "acc_knn", "acc_logreg"])
                writer.writerow({
                    "pair": "NULL",
                    "R2_behavior": f"{R2_behavior:.4f}",
                    "acc_knn": f"{acc_knn:.4f}",
                    "acc_logreg": f"{acc_logreg:.4f}"
                })

            df_null = collect_decoding_timecourse("NULL", y_true, y_pred, test_idx_local)
            all_timecourse.append(df_null)
            print(f"[ok] NULL model: acc={acc_knn:.3f}, R²={R2_behavior:.3f}")

        except Exception as e:
            print(f"[error] failed null model eval: {e}")
    else:
        print(f"[warn] no null model embedding found at {null_emb_path}")

    # Save combined decoding timecourse (all pairs + NULL)
    df_all = save_decoding_timecourse(all_timecourse, eval_root / "decoding_timecourse.csv")

    # Build summary heatmaps
    df_pairs = pd.read_csv(summary_path, keep_default_na=False) #else NULL read as NaN
    df_pairs["pair"] = df_pairs["pair"].astype(str).str.strip()

    nodes = list(NODE_MAP.keys())
    single_csv = Path(f"./output_xCEBRA_lags/{patient_id}/single_node_eval_new/summary.csv")
    df_single = pd.read_csv(single_csv) if single_csv.exists() else None

    mat_raw = build_acc_matrix(nodes, df_pairs, df_single=df_single, baseline=None, metric="acc_logreg")
    plot_heatmap(mat_raw, nodes,
        title="Pairwise Node Accuracy Heatmap",
        out_path=eval_root.parent / "summary_accuracy_heatmap.png",
        cmap="Blues", cbar_label="LogReg Accuracy"
    )

    # compute summary heatmap wth NULL baseline
    null_acc = df_pairs.loc[df_pairs["pair"].str.upper() == "NULL", "acc_logreg"]
    null_acc = float(null_acc.iloc[0]) if len(null_acc) > 0 else 0.0

    mat_delta = build_acc_matrix(nodes, df_pairs, df_single=df_single, baseline=null_acc, metric="acc_logreg")
    plot_heatmap(mat_delta, nodes,
        title=f"Pairwise Node Accuracy Δ vs NULL (NULL={null_acc:.2f})",
        out_path=eval_root.parent / "summary_accuracy_heatmap_NULL_Baseline.png",
        cmap="RdBu", center=0, cbar_label="Δ LogReg Accuracy vs NULL"
    )

    if df_all is not None:
        plot_decoding_timecourses(
            csv_path=eval_root / "decoding_timecourse.csv",
            out_path=eval_root / "decoding_timecourse_grid.png",
            emotion_map=EMOTION_MAP,
            n_cols=4 
        )

if __name__ == "__main__":
    main()
