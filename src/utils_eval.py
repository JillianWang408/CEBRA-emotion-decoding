import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from src.config import (
    PATIENT_CONFIG, NODE_MAP,
    MODEL_DIR, FULL_EMOTION_PATH,
    BEHAVIOR_INDICES, EMOTION_MAP
)

# ------------------------------
# Single-node evaluation
# ------------------------------

def evaluate_node(pid: int, node: str, node_eval_dir: Path):
    """
    Evaluate a single-node embedding:
    - Skip if node missing (embedding not produced in training)
    - Load embedding from training folder (single_node/{node}/embedding.pt)
    - Slice test split
    - Compute R² and KNN accuracy
    - Save confusion matrix into eval folder (single_node_eval/{node}/)
    Returns: (acc_knn, R2_behavior, n_test) OR (None, None, 0) if skipped
    """
    _, alt = PATIENT_CONFIG[pid]
    train_node_dir = Path(f"./output_xCEBRA_lags/{alt}/single_node/{node}")
    emb_path = train_node_dir / "embedding.pt"

    if not emb_path.exists():
        print(f"[SKIP] {node}: no embedding found (likely skipped in training).")
        return None, None, 0

    # --- load embedding ---
    embedding = torch.load(emb_path, map_location="cpu")
    if embedding.ndim == 3 and embedding.shape[0] == 1:
        embedding = embedding.squeeze(0).T
    elif embedding.ndim == 2:
        pass
    else:
        raise ValueError(f"Bad embedding shape {tuple(embedding.shape)}")
    embedding = embedding.numpy()

    # --- labels & test split ---
    emotion_tensor_full = torch.load(FULL_EMOTION_PATH, map_location="cpu")
    if emotion_tensor_full.ndim == 1:
        emotion_tensor_full = emotion_tensor_full.unsqueeze(1)
    y = emotion_tensor_full.float().contiguous().numpy().squeeze()

    test_idx = np.load(MODEL_DIR / "test_idx.npy")
    valid_idx = test_idx[test_idx < len(embedding)]
    emb_test = embedding[valid_idx]
    y_test = y[valid_idx].astype(int)

    # --- behavior slice ---
    X_behavior = emb_test[:, slice(*BEHAVIOR_INDICES)]

    # --- R² ---
    linear_model = LinearRegression()
    R2_behavior = linear_model.fit(X_behavior, y_test).score(X_behavior, y_test)

    # --- KNN ---
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_behavior, y_test)
    y_pred = knn_model.predict(X_behavior)
    acc_knn = accuracy_score(y_test, y_pred)

    # --- Confusion matrix ---
    unique_labels = np.unique(y_test)
    filtered_emotion_labels = [EMOTION_MAP[i] for i in unique_labels]
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=filtered_emotion_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.title(f"{node} — KNN acc={acc_knn:.2f}, R²={R2_behavior:.2f}")
    fig.tight_layout()
    cm_path = node_eval_dir / f"{node}_confusion.png"
    plt.savefig(cm_path, dpi=220)
    plt.close(fig)

    return acc_knn, R2_behavior, len(valid_idx)


def run_single_node_eval(pid: int, base_dir="./output_xCEBRA_lags"):
    _, alt = PATIENT_CONFIG[pid]
    out_dir = Path(base_dir) / str(alt) / "single_node_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, accs, nodes = [], [], []

    for node in NODE_MAP.keys():
        node_dir = out_dir / node
        node_dir.mkdir(exist_ok=True)

        acc, r2, n_test = evaluate_node(pid, node, node_dir)
        rows.append({"node": node, "acc": acc, "R2_behavior": r2, "n_test": n_test})
        nodes.append(node)
        accs.append(acc)

    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(max(8, 0.5*len(nodes)), 4))
    ax.bar(nodes, accs)
    ax.set_xticklabels(nodes, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Single-node accuracy for patient {alt}")
    fig.tight_layout()
    fig.savefig(out_dir / "nodes_accuracy.png", dpi=200)
    plt.close(fig)

    print(f"[ok] wrote {out_dir/'summary.csv'} and nodes_accuracy.png")


# ------------------------------
# Pair-node evaluation helpers
# ------------------------------

def evaluate_embedding(embedding, y, test_idx, pair_name, out_dir: Path):
    """Evaluate R², KNN accuracy, and save confusion matrix."""
    valid_idx = test_idx[test_idx < len(embedding)]
    embedding = embedding[valid_idx]
    y = y[valid_idx]
    y = y.squeeze().astype(int)

    X_behavior = embedding[:, slice(*BEHAVIOR_INDICES)]

    linear_model = LinearRegression()
    R2_behavior = linear_model.fit(X_behavior, y).score(X_behavior, y)

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_behavior, y)
    y_pred = knn_model.predict(X_behavior)
    acc_knn = accuracy_score(y, y_pred)

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


def load_embedding_TxD(emb_path: Path):
    """Load .pt embedding and return [T, D] numpy array with robust shape fixes."""
    embedding = torch.load(emb_path, map_location="cpu")
    if embedding.ndim == 3 and embedding.shape[0] == 1:
        embedding = embedding.squeeze(0).T
    elif embedding.ndim == 2:
        pass
    else:
        raise ValueError(f"Bad embedding shape {tuple(embedding.shape)}")
    return embedding.numpy()


def build_acc_matrix(nodes, df_pairs, df_single=None, baseline=None):
    """Build accuracy (or Δ vs baseline) matrix."""
    mat = np.full((len(nodes), len(nodes)), np.nan, dtype=float)

    if df_single is not None and len(df_single) > 0:
        for _, r in df_single.iterrows():
            node = str(r["node"])
            if node in nodes:
                i = nodes.index(node)
                acc = float(r["acc"])
                if baseline is not None:
                    acc -= baseline
                mat[i, i] = acc

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


def plot_heatmap(mat, nodes, title, out_path: Path, cmap="viridis", center=None, cbar_label="KNN Accuracy"):
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