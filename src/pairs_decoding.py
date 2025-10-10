import os
import csv
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, r2_score

from src.config import FULL_EMOTION_PATH, NODE_MAP, PATIENT_CONFIG
from src.utils_decoding import load_embedding_TxD, build_acc_matrix, plot_heatmap


DECODERS = ["knn", "logreg"]

def decode_embedding(embedding, y, decoder):
    """Run KNN or LogReg decoding on the embedding."""
    min_len = min(len(embedding), len(y))
    X = embedding[:min_len]
    y = y[:min_len].squeeze().astype(int)

    # 80/20 split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # linear regression R²
    coef, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
    y_pred_lin = X_test @ coef
    R2_behavior = r2_score(y_test, y_pred_lin)

    if decoder == "logreg":
        model = LogisticRegression(max_iter=2000, solver="lbfgs")
    elif decoder == "knn":
        model = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError(f"Unknown decoder: {decoder}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return R2_behavior, acc

def main():
    pid = int(float(os.environ["PATIENT_ID"]))
    _, patient_id = PATIENT_CONFIG[pid]

    # --- paths ---
    enc_root = Path(f"./output_xCEBRA_lags/{patient_id}/pairs_encoding")
    null_root = Path(f"./output_xCEBRA_lags/{patient_id}/null_model")
    dec_root = Path(f"./output_xCEBRA_lags/{patient_id}/pairs_decoding")
    eval_dir = dec_root / "eval_summary"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # --- load emotion labels ---
    emotion_tensor_full = torch.load(FULL_EMOTION_PATH, map_location="cpu")
    if emotion_tensor_full.ndim == 1:
        emotion_tensor_full = emotion_tensor_full.unsqueeze(1)
    emotion_tensor_full = emotion_tensor_full.float().contiguous().numpy()

    rows = []
    print(f"[INFO] Starting decoding for patient: {patient_id}")

    # -----------------------------------------------------
    #   Evaluate all embeddings (single + pair)
    # -----------------------------------------------------
    for subdir in sorted(enc_root.glob("*")):
        if not subdir.is_dir():
            continue

        emb_path = subdir / "embedding.pt"
        if not emb_path.exists():
            continue

        name = subdir.name
        node_type = "pair" if "__" in name else "single"

        try:
            embedding = load_embedding_TxD(emb_path)
        except Exception as e:
            print(f"[WARN] Skipping {name}: failed to load ({e})")
            continue

        # run all decoders
        result = {"name": name, "type": node_type}
        for decoder in DECODERS:
            R2_behavior, acc = decode_embedding(embedding, emotion_tensor_full, decoder)
            result[f"R2_behavior"] = f"{R2_behavior:.4f}"
            result[f"acc_{decoder}"] = f"{acc:.4f}"
            print(f"[ok] {name:<15} [{decoder}] acc={acc:.3f}, R²={R2_behavior:.3f}")

        rows.append(result)

    # -----------------------------------------------------
    #   NULL BASELINE
    # -----------------------------------------------------
    acc_null = 0.0
    if (null_root / "embedding.pt").exists():
        try:
            embedding = load_embedding_TxD(null_root / "embedding.pt")
            R2_behavior, acc_knn = decode_embedding(embedding, emotion_tensor_full, "knn")
            _, acc_logreg = decode_embedding(embedding, emotion_tensor_full, "logreg")
            acc_null = np.mean([acc_knn, acc_logreg])
            rows.append({
                "name": "NULL", "type": "null",
                "R2_behavior": f"{R2_behavior:.4f}",
                "acc_knn": f"{acc_knn:.4f}",
                "acc_logreg": f"{acc_logreg:.4f}"
            })
            print(f"[NULL] acc_knn={acc_knn:.3f}, acc_logreg={acc_logreg:.3f}")
        except Exception as e:
            print(f"[WARN] Failed NULL model eval: {e}")

    # -----------------------------------------------------
    #   SAVE SUMMARY CSV
    # -----------------------------------------------------
    summary_csv = eval_dir / "summary_all.csv"
    cols = ["name", "type", "R2_behavior"] + [f"acc_{d}" for d in DECODERS]
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[DONE] Saved summary_all.csv to {summary_csv}")

    # -----------------------------------------------------
    #   GENERATE HEATMAPS (one per decoder)
    # -----------------------------------------------------
    df = pd.read_csv(summary_csv)
    nodes = list(NODE_MAP.keys())

    for decoder in DECODERS:
        mat_raw = build_acc_matrix(nodes, df, df_single=None,
                                   baseline=None, metric=f"acc_{decoder}")
        plot_heatmap(mat_raw, nodes,
            title=f"{decoder.upper()} Accuracy Heatmap ({patient_id})",
            out_path=eval_dir / f"summary_accuracy_heatmap_{decoder}.png",
            cmap="viridis", cbar_label=f"{decoder.upper()} Accuracy"
        )

        mat_delta = build_acc_matrix(nodes, df, df_single=None,
                                     baseline=acc_null, metric=f"acc_{decoder}")
        plot_heatmap(mat_delta, nodes,
            title=f"{decoder.upper()} Δ Accuracy vs NULL ({patient_id})",
            out_path=eval_dir / f"summary_accuracy_heatmap_NULL_Baseline_{decoder}.png",
            cmap="RdBu", center=0, cbar_label=f"Δ {decoder.upper()} Accuracy vs NULL"
        )

    print(f"[DONE] Generated heatmaps for decoders: {', '.join(DECODERS)}")
    print(f"[FINISHED] Full decoding complete for patient {patient_id}.")


if __name__ == "__main__":
    main()
