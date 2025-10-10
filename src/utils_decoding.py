import csv
from pathlib import Path
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, r2_score

from src.config import FULL_EMOTION_PATH, NODE_MAP


# EMBEDDING LOADING
def load_embedding_TxD(emb_path: Path):
    """
    Load .pt embedding and return [T, D] numpy array with robust shape fixes.
    Compatible with embeddings saved from train_and_save().
    """
    embedding = torch.load(emb_path, map_location="cpu")

    if embedding.ndim == 3 and embedding.shape[0] == 1:
        embedding = embedding.squeeze(0).T  # [T, D]
    elif embedding.ndim == 2:
        pass  # already [T, D]
    else:
        raise ValueError(f"Unexpected embedding shape {tuple(embedding.shape)}")

    return embedding.numpy()


# DECODING EVALUATION
def evaluate_embedding(embedding, y, n_neighbors=5):
    """
    Evaluate a neural embedding with three metrics:
    - R²: variance explained (diagnostic, continuous structure)
    - KNN accuracy: nonlinear neighborhood consistency
    - Logistic Regression accuracy: linear separability
    """

    # --- Align lengths ---
    min_len = min(len(embedding), len(y))
    X = embedding[:min_len]
    y = y[:min_len].squeeze().astype(int)

    # --- 80/20 temporal split ---
    T = len(X)
    split = int(0.8 * T)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # --- Linear regression (diagnostic R²) ---
    coef, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
    y_pred_lin = X_test @ coef
    R2_behavior = r2_score(y_test, y_pred_lin)

    # --- Logistic Regression decoder ---
    log_reg = LogisticRegression(max_iter=2000, solver="lbfgs")
    log_reg.fit(X_train, y_train)
    acc_logreg = accuracy_score(y_test, log_reg.predict(X_test))

    # --- KNN decoder ---
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    acc_knn = accuracy_score(y_test, knn.predict(X_test))

    return R2_behavior, acc_knn, acc_logreg


# MATRIX BUILDING (for heatmaps)
def build_acc_matrix(nodes, df, baseline=None, metric="acc_logreg"):
    """
    Build a full node × node accuracy matrix from summary DataFrame.
    Works for both single-node (diagonal) and pairwise (off-diagonal) entries.
    """

    mat = np.full((len(nodes), len(nodes)), np.nan, dtype=float)

    for _, r in df.iterrows():
        name = str(r["name"]).strip()

        if name.upper() == "NULL":
            continue

        try:
            acc = float(r[metric])
        except KeyError:
            acc = float(r.get("acc_logreg", r.get("acc_knn", np.nan)))

        if baseline is not None:
            acc -= baseline

        # pair (A__B)
        if "__" in name:
            node1, node2 = name.split("__", 1)
            if node1 in nodes and node2 in nodes:
                i, j = nodes.index(node1), nodes.index(node2)
                mat[i, j] = mat[j, i] = acc

        # single node (A)
        else:
            if name in nodes:
                i = nodes.index(name)
                mat[i, i] = acc

    return mat


# HEATMAP PLOTTING
def plot_heatmap(mat, nodes, title, out_path: Path,
                 cmap="viridis", center=None, cbar_label="Accuracy"):
    """
    Plot and save a symmetric accuracy or delta-accuracy heatmap.
    """
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