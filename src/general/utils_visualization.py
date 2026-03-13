import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ===============================================================
# Confusion Matrix Heatmap
# ===============================================================
def plot_confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_map: dict,
    out_path: Path,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    show_all_classes: bool = True,
):
    """
    Plot confusion matrix heatmap for classification results.

    Args:
        y_true: True labels (original label space, e.g. 0-6 for arousal/valence)
        y_pred: Predicted labels (same space)
        label_map: Dict mapping label index -> display string (e.g. AROUSAL_MAP)
        out_path: Where to save the figure
        title: Plot title
        cmap: Matplotlib colormap name
        show_all_classes: If True, use all keys from label_map so the matrix shows the full
            scale (e.g. 7x7 for arousal) even when some classes are missing in the data.
    """
    if show_all_classes and label_map:
        labels = np.array(sorted(label_map.keys()), dtype=int)
    else:
        labels = np.unique(np.concatenate([y_true, y_pred])).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    label_names = [label_map.get(int(l), str(l)) for l in labels]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(ax=ax, xticks_rotation=45, cmap=cmap, values_format="d")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] saved confusion matrix heatmap → {out_path}")


# ===============================================================
# Heatmap Visualization
# ===============================================================
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
# Decoding Timecourse Visualization
# ===============================================================

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
    fig_height = 3.2 * n_rows
    fig_width = 20 * n_cols
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

    fig.suptitle("Decoding Timecourse", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if out_path:
        plt.savefig(out_path, dpi=300)
        print(f"[plot] saved multi-panel timecourse → {out_path}")
    plt.close(fig)
