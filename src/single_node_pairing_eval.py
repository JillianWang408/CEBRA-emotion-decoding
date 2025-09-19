# src/eval_pair_nodes_patient.py

import csv
import json
from pathlib import Path
import os
import pandas as pd

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from src.config import (
    MODEL_DIR, FULL_EMOTION_PATH, BEHAVIOR_INDICES,
    EMOTION_MAP, PATIENT_CONFIG, NODE_MAP
)

def evaluate_embedding(embedding, y, test_idx, pair_name, out_dir):
    """Evaluate R², KNN accuracy, and save confusion matrix."""
    # --- slice test split ---
    valid_idx = test_idx[test_idx < len(embedding)]
    embedding = embedding[valid_idx]
    y = y[valid_idx]
    y = y.squeeze().astype(int)

    # --- behavior slice ---
    X_behavior = embedding[:, slice(*BEHAVIOR_INDICES)]

    # --- R² ---
    linear_model = LinearRegression()
    R2_behavior = linear_model.fit(X_behavior, y).score(X_behavior, y)

    # --- KNN ---
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_behavior, y)
    y_pred = knn_model.predict(X_behavior)
    acc_knn = accuracy_score(y, y_pred)

    # --- Confusion matrix ---
    unique_labels = np.unique(y)
    filtered_emotion_labels = [EMOTION_MAP[i] for i in unique_labels]

    cm = confusion_matrix(y, y_pred, labels=unique_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=filtered_emotion_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.title(f"{pair_name} — KNN acc={acc_knn:.2f}, R²={R2_behavior:.2f}")
    fig.tight_layout()
    cm_path = out_dir / f"{pair_name}_confusion.png"
    plt.savefig(cm_path, dpi=220)
    plt.close(fig)

    return R2_behavior, acc_knn, cm_path


def _load_embedding_TxD(emb_path: Path):
    """Load .pt and return [T, D] numpy array with robust shape fixes."""
    embedding = torch.load(emb_path, map_location="cpu")
    if embedding.ndim == 3 and embedding.shape[0] == 1:
        # common saved shape: [1, D, T] → [T, D]
        embedding = embedding.squeeze(0).T
    elif embedding.ndim == 2:
        # already [T, D]
        pass
    else:
        raise ValueError(f"bad embedding shape {tuple(embedding.shape)}")
    return embedding.numpy()


def _build_acc_matrix(nodes, df_pairs, df_single=None, baseline=None):
    """Build (len(nodes) x len(nodes)) matrix of accuracies (or Δ vs baseline)."""
    mat = np.full((len(nodes), len(nodes)), np.nan, dtype=float)

    # diagonal from single-node eval
    if df_single is not None and len(df_single) > 0:
        for _, r in df_single.iterrows():
            node = str(r["node"])
            if node in nodes:
                i = nodes.index(node)
                acc = float(r["acc"])
                if baseline is not None:
                    acc -= baseline
                mat[i, i] = acc

    # off-diagonals from pair eval
    for _, r in df_pairs.iterrows():
        pair = str(r["pair"])
        if pair.upper() == "NULL" or "__" not in pair:
            continue
        node1, node2 = pair.split("__", 1)
        if node1 in nodes and node2 in nodes:
            acc = float(r["acc"])
            if baseline is not None:
                acc -= baseline
            i, j = nodes.index(node1), nodes.index(node2)
            mat[i, j] = acc
            mat[j, i] = acc
    return mat


def _plot_heatmap(mat, nodes, title, out_path, cmap="viridis", center=None, cbar_label="KNN Accuracy"):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        mat, annot=True, fmt=".2f",
        xticklabels=nodes, yticklabels=nodes,
        cmap=cmap, center=center,
        cbar_kws={"label": cbar_label}
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[ok] saved heatmap → {out_path}")


def main():
    pid = int(float(os.environ["PATIENT_ID"]))
    _, patient_id = PATIENT_CONFIG[pid]

    out_root = Path(f"./output_xCEBRA_lags/{patient_id}/pair_nodes")
    eval_root = Path(f"./output_xCEBRA_lags/{patient_id}/pair_nodes_eval")
    eval_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Evaluating pairwise embeddings for patient {patient_id}")

    # --- load labels & test split ---
    emotion_tensor_full = torch.load(FULL_EMOTION_PATH, map_location="cpu")
    if emotion_tensor_full.ndim == 1:
        emotion_tensor_full = emotion_tensor_full.unsqueeze(1)
    emotion_tensor_full = emotion_tensor_full.float().contiguous().numpy()

    test_idx = np.load(MODEL_DIR / "test_idx.npy")

    # --- prepare summary file ---
    summary_path = eval_root / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair", "R2_behavior", "acc", "confusion_path"])
        writer.writeheader()

        # --- iterate over all pairs ---
        for pair_dir in sorted(out_root.glob("*__*")):
            pair_name = pair_dir.name
            emb_path = pair_dir / "embedding.pt"
            if not emb_path.exists():
                print(f"[warn] missing embedding for {pair_name}")
                continue

            try:
                embedding = _load_embedding_TxD(emb_path)
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

    # --- evaluate null model (also record acc_null for plotting) ---
    acc_null = None
    null_dir = Path(f"./output_xCEBRA_lags/{patient_id}/null_model")
    emb_path = null_dir / "embedding.pt"
    if emb_path.exists():
        try:
            embedding = _load_embedding_TxD(emb_path)
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

    # --- load once for plotting ---
    df_pairs = pd.read_csv(summary_path)
    # normalize 'pair' column just in case
    df_pairs["pair"] = df_pairs["pair"].astype(str).str.strip()

    # If null wasn't evaluated above, try to read it here
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

    # single-node summary (for diagonal)
    single_csv = Path(f"./output_xCEBRA_lags/{patient_id}/single_node_eval/summary.csv")
    df_single = pd.read_csv(single_csv) if single_csv.exists() else None

    # --- build & plot heatmaps (concise) ---
    nodes = list(NODE_MAP.keys())

    # Raw accuracy
    mat_raw = _build_acc_matrix(nodes, df_pairs, df_single=df_single, baseline=None)
    _plot_heatmap(
        mat_raw, nodes,
        title="Pairwise Node Accuracy Heatmap",
        out_path=eval_root.parent / "summary_accuracy_heatmap.png",
        cmap="viridis",
        cbar_label="KNN Accuracy"
    )

    # Δ vs NULL
    mat_delta = _build_acc_matrix(nodes, df_pairs, df_single=df_single, baseline=acc_null)
    _plot_heatmap(
        mat_delta, nodes,
        title=f"Pairwise Node Accuracy Δ vs NULL (NULL={acc_null:.2f})",
        out_path=eval_root.parent / "summary_accuracy_heatmap_NULL_Baseline.png",
        cmap="RdBu", center=0,
        cbar_label="Δ KNN Accuracy vs NULL"
    )


if __name__ == "__main__":
    main()