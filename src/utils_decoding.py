import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, r2_score
from src.config import (
    PATIENT_CONFIG, FULL_EMOTION_PATH
)
from src.utils import align_embedding_labels

# ------------------------------
# Embedding loading
# ------------------------------
def load_embedding_TxD(emb_path: Path):
    """Load .pt embedding and return [T, D] numpy array."""
    embedding = torch.load(emb_path, map_location="cpu")
    if embedding.ndim == 3 and embedding.shape[0] == 1:
        embedding = embedding.squeeze(0).T
    elif embedding.ndim != 2:
        raise ValueError(f"Unexpected embedding shape {tuple(embedding.shape)}")
    return embedding.numpy()


# ===============================================================
# Shared helpers
# ===============================================================
def _split_train_test(embedding, y_full):
    """Align embedding with labels, then return aligned train/test splits."""
    y_aligned, offset, split = align_embedding_labels(embedding, y_full)
    X = embedding[:len(y_aligned)]
    y = np.squeeze(y_aligned).astype(int)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, X_test, y_train, y_test, offset, split


def _compute_decoding_metrics(X_train, X_test, y_train, y_test, n_neighbors=5):
    """Compute R², KNN, and Logistic Regression accuracies."""
    coef, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
    y_pred_lin = X_test @ coef
    R2_behavior = r2_score(y_test, y_pred_lin)

    log_reg = LogisticRegression(max_iter=2000, solver="lbfgs")
    log_reg.fit(X_train, y_train)
    acc_logreg = accuracy_score(y_test, log_reg.predict(X_test))

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    acc_knn = accuracy_score(y_test, knn.predict(X_test))

    return R2_behavior, acc_knn, acc_logreg

# ------------------------------
# Single-node evaluation
# ------------------------------

def evaluate_embedding(embedding, y, n_neighbors=5):
    """Evaluate embedding with R², KNN, and Logistic Regression metrics."""
    X_train, X_test, y_train, y_test, offset, split = _split_train_test(embedding, y)
    return _compute_decoding_metrics(X_train, X_test, y_train, y_test, n_neighbors)

def evaluate_node(pid: int, node: str, base_dir: Path, n_neighbors=5):
    """Evaluate a single-node embedding with consistent alignment."""
    _, patient_id = PATIENT_CONFIG[pid]
    node_dir = Path(base_dir) / patient_id / "single_node" / node
    emb_path = node_dir / "embedding.pt"
    if not emb_path.exists():
        print(f"[SKIP] {node}: no embedding found.")
        return None, None, None

    embedding = load_embedding_TxD(emb_path)
    y_full = torch.load(FULL_EMOTION_PATH, map_location="cpu").float().numpy()
    X_train, X_test, y_train, y_test, offset, split = _split_train_test(embedding, y_full)
    return _compute_decoding_metrics(X_train, X_test, y_train, y_test, n_neighbors)

# ------------------------------
# Pair-node evaluation helpers
# ------------------------------
def evaluate_pair_embedding(
    embedding, y, pair_name, out_dir: Path, n_neighbors=5, return_predictions=False
):
    """Evaluate R², KNN, and LogReg accuracy; optionally return predictions."""
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    X_train, X_test, y_train, y_test, offset, split = _split_train_test(embedding, y)
    R2_behavior, acc_knn, acc_logreg = _compute_decoding_metrics(
        X_train, X_test, y_train, y_test, n_neighbors
    )

    if return_predictions:
        log_reg = LogisticRegression(max_iter=2000, solver="lbfgs")
        log_reg.fit(X_train, y_train)
        y_pred_log = log_reg.predict(X_test)
        test_idx = np.arange(split, split + len(y_test))
        return R2_behavior, acc_knn, acc_logreg, y_pred_log, y_test, test_idx
    else:
        return R2_behavior, acc_knn, acc_logreg


# ===============================================================
# Accuracy matrix
# ===============================================================
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


