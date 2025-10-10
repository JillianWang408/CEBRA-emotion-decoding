import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,  r2_score, confusion_matrix, ConfusionMatrixDisplay


from src.config import (
    PATIENT_CONFIG, NODE_MAP,
    MODEL_DIR, FULL_EMOTION_PATH,
    BEHAVIOR_INDICES, EMOTION_MAP
)

# ------------------------------
# Single-node evaluation
# ------------------------------

import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, r2_score

from src.config import FULL_EMOTION_PATH, PATIENT_CONFIG


def evaluate_node(pid: int, node: str, base_dir: Path, n_neighbors=5):
    """
    Evaluate a single-node embedding:
    - Uses 80/20 split on embedding length
    - Computes Logistic Regression (default) + KNN + R²
    Returns (R2_behavior, acc_knn, acc_logreg)
    """
    _, patient_id = PATIENT_CONFIG[pid]
    node_dir = Path(base_dir) / patient_id / "single_node" / node
    emb_path = node_dir / "embedding.pt"

    if not emb_path.exists():
        print(f"[SKIP] {node}: no embedding found.")
        return None, None, None

    # --- Load embedding ---
    embedding = torch.load(emb_path, map_location="cpu")
    if embedding.ndim == 3 and embedding.shape[0] == 1:
        embedding = embedding.squeeze(0).T
    elif embedding.ndim == 2:
        pass
    else:
        raise ValueError(f"Bad embedding shape {tuple(embedding.shape)}")
    embedding = embedding.numpy()

    # --- Load emotion labels ---
    emotion_tensor_full = torch.load(FULL_EMOTION_PATH, map_location="cpu")
    if emotion_tensor_full.ndim == 1:
        emotion_tensor_full = emotion_tensor_full.unsqueeze(1)
    y = emotion_tensor_full.float().contiguous().numpy().squeeze().astype(int)

    # --- Align lengths (embedding shorter than label array) ---
    min_len = min(len(embedding), len(y))
    embedding = embedding[:min_len]
    y = y[:min_len]

    # --- 80/20 split ---
    T = len(embedding)
    split = int(0.8 * T)
    train_idx = np.arange(split)
    test_idx = np.arange(split, T)

    X_train, y_train = embedding[train_idx], y[train_idx]
    X_test, y_test = embedding[test_idx], y[test_idx]

    # --- Linear regression (R²) ---
    regr_coef, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
    y_pred_lin = X_test @ regr_coef
    R2_behavior = r2_score(y_test, y_pred_lin)

    # --- Logistic Regression decoder (default) ---
    log_reg = LogisticRegression(max_iter=2000, solver="lbfgs")
    log_reg.fit(X_train, y_train)
    acc_logreg = accuracy_score(y_test, log_reg.predict(X_test))

    # --- KNN decoder ---
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    acc_knn = accuracy_score(y_test, knn.predict(X_test))

    return R2_behavior, acc_knn, acc_logreg


# ------------------------------
# Pair-node evaluation helpers
# ------------------------------

def evaluate_embedding(embedding, y, pair_name, out_dir: Path, n_neighbors=5):
    """Evaluate R², KNN accuracy, and Logistic Regression accuracy using embedding-based train/test split."""

    # --- Align lengths (embedding shorter than raw) ---
    min_len = min(len(embedding), len(y))
    embedding = embedding[:min_len]
    y = y[:min_len].squeeze().astype(int)

    # --- Recompute train/test split based on embedding length ---
    T = len(embedding)
    split = int(0.8 * T)
    train_idx = np.arange(split)
    test_idx = np.arange(split, T)

    # --- Split data ---
    X_train, y_train = embedding[train_idx], y[train_idx]
    X_test, y_test = embedding[test_idx], y[test_idx]

    # --- Linear Regression for diagnostic R² ---
    regr_coef, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
    y_pred_lin = X_test @ regr_coef
    R2_behavior = r2_score(y_test, y_pred_lin)

    # --- Logistic Regression decoder ---
    log_reg = LogisticRegression(max_iter=2000, solver="lbfgs")
    log_reg.fit(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)
    acc_logreg = accuracy_score(y_test, y_pred_log)

    # --- KNN decoder ---
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    acc_knn = accuracy_score(y_test, y_pred_knn)

    # --- Return metrics ---
    return R2_behavior, acc_knn, acc_logreg

#     # --- Confusion matrix for KNN ---

#     unique_labels = np.unique(y)
#     filtered_emotion_labels = [EMOTION_MAP[i] for i in unique_labels]
#     cm = confusion_matrix(y, y_pred, labels=unique_labels)

#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=filtered_emotion_labels)
#     fig, ax = plt.subplots(figsize=(8, 6))
#     disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
#     plt.title(f"{pair_name} — KNN acc={acc_knn:.2f}, R²={R2_behavior:.2f}")
#     fig.tight_layout()
#     cm_path = out_dir / f"{pair_name}_confusion.png"
#     plt.savefig(cm_path, dpi=220)
#     plt.close(fig)

#     return R2_behavior, acc_knn, cm_path


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


def build_acc_matrix(nodes, df_pairs, df_single=None, baseline=None, metric="acc_logreg"):
    """
    Build accuracy (or Δ vs baseline) matrix.
    metric: one of {"acc_knn", "acc_logreg"}
    """
    mat = np.full((len(nodes), len(nodes)), np.nan, dtype=float)

    # --- Fill single-node diagonals ---
    if df_single is not None and len(df_single) > 0:
        for _, r in df_single.iterrows():
            node = str(r["node"])
            if node in nodes:
                i = nodes.index(node)
                try:
                    acc = float(r[metric])
                except KeyError:
                    # fallback if missing column
                    acc = float(r.get("acc_logreg", r.get("acc_knn", np.nan)))
                if baseline is not None:
                    acc -= baseline
                mat[i, i] = acc

    # --- Fill pairwise off-diagonals ---
    for _, r in df_pairs.iterrows():
        pair = str(r["pair"])
        if pair.upper() == "NULL" or "__" not in pair:
            continue

        node1, node2 = pair.split("__", 1)
        if node1 in nodes and node2 in nodes:
            try:
                acc = float(r[metric])
            except KeyError:
                acc = float(r.get("acc_logreg", r.get("acc_knn", np.nan)))
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