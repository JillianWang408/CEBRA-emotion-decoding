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
def evaluate_embedding(embedding, y, pair_name, out_dir: Path, n_neighbors=5,return_predictions=False, test_idx=None):
    """Evaluate R², KNN accuracy, and Logistic Regression accuracy using embedding-based train/test split."""

    if isinstance(embedding, torch.Tensor):
        embedding = embedding.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    # --- Align lengths (embedding shorter than label array) ---
    min_len = min(len(embedding), len(y))
    embedding = embedding[:min_len]
    y = y[:min_len].squeeze().astype(int)

    
    # Default 80/20 temporal split
    T = len(embedding)
    split = int(0.8 * T)
    train_idx = np.arange(split)
    test_idx_local = np.arange(split, T)

    X_train, y_train = embedding[train_idx], y[train_idx]
    X_test, y_test = embedding[test_idx_local], y[test_idx_local]

    # --- Linear Regression (diagnostic R²) ---
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

    if return_predictions:
        return R2_behavior, acc_knn, acc_logreg, y_pred_log, y_test, test_idx_local
    else:
        return R2_behavior, acc_knn, acc_logreg



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


def plot_heatmap(mat, nodes, title, out_path: Path, cmap="Blues", center=None, cbar_label="KNN Accuracy"):
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



# ===============================================================
# Decoding timecourse utilities
# ===============================================================
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def collect_decoding_timecourse(pair_name, y_true, y_pred, test_idx):
    df = pd.DataFrame({
        "pair": pair_name,
        "timestep": test_idx,
        "actual": y_true,
        "prediction": y_pred
    })
    return df


def save_decoding_timecourse(all_results, save_path):
    """
    Concatenate and save decoding results from all pairs into one CSV.

    Args:
        all_results (list[pd.DataFrame]): List of per-pair DataFrames
        save_path (Path): Where to save combined CSV
    """
    if not all_results:
        print("[warn] No decoding timecourse data to save.")
        return None

    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv(save_path, index=False)
    print(f"[done] wrote decoding timecourse: {save_path}")
    return df_all


def plot_decoding_timecourses(
    csv_path,
    out_path=None,
    emotion_map=None,
    n_cols=4
):
    """
    Plot prediction vs actual over time for each pair as separate subplots.
    Includes NULL model and removes blank panels.
    """
    df = pd.read_csv(csv_path, keep_default_na=False).dropna(subset=["pair"])
    pairs = [p for p in df["pair"].unique()]
    if "NULL" in pairs:
        pairs = [p for p in pairs if p != "NULL"] + ["NULL"]  # put NULL last

    n_pairs = len(pairs)
    n_rows = int(np.ceil(n_pairs / n_cols))
    fig_height = 2.8 * n_rows
    fig_width = 4.8 * n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    axes = np.array(axes).reshape(-1)

    for i, pair in enumerate(pairs):
        ax = axes[i]
        dfp = df[df["pair"] == pair].copy()
        if dfp.empty:
            ax.set_visible(False)
            continue

        if emotion_map is not None:
            ticks = sorted(emotion_map.keys())
            ax.set_yticks(ticks)
            ax.set_yticklabels([emotion_map[k] for k in ticks], fontsize=7)
            ax.set_ylim(min(ticks) - 0.5, max(ticks) + 0.5)
            ax.grid(axis="y", linestyle="--", alpha=0.3)

        ax.plot(dfp["timestep"], dfp["actual"], color="tab:blue", label="Actual", alpha=0.9, linewidth=1)
        ax.plot(dfp["timestep"], dfp["prediction"], color="tab:orange", label="Prediction", alpha=0.7, linewidth=1)
        ax.set_title(pair, fontsize=9)
        ax.set_xlabel("Timestep", fontsize=8)
        ax.set_ylabel("Emotion", fontsize=8)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    # hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Decoding Timecourse per Pair (including NULL)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if out_path:
        plt.savefig(out_path, dpi=300)
        print(f"[plot] saved multi-panel timecourse → {out_path}")
    plt.close(fig)
