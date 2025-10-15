import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path
import csv


from src.config import (
    MODEL_DIR,
    EMBEDDING_PATH, FULL_EMOTION_PATH,
    BEHAVIOR_INDICES, EMOTION_MAP,
    EVALUATION_OUTPUT_DIR, EVALUATION_EMBEDDING_PLOT, EVALUATION_CONFUSION_PLOT, PATIENT_CONFIG
)


def evaluate_embedding(
    model_dir=MODEL_DIR,
    embedding_path=EMBEDDING_PATH,
    out_dir=EVALUATION_OUTPUT_DIR,
    embedding_plot=EVALUATION_EMBEDDING_PLOT,
    confusion_plot=EVALUATION_CONFUSION_PLOT
):
    # === Ensure output directories exist ===
    out_dir.mkdir(exist_ok=True, parents=True)

    # === Load embedding and label tensors ===
    embedding_full = torch.load(embedding_path).squeeze(0).T.numpy()
    emotion_tensor_full = torch.load(FULL_EMOTION_PATH).squeeze().cpu().numpy().astype(int)

    # === Load test indices ===
    test_split_path = out_dir.parent / "test_idx.npy"
    if test_split_path.exists():
        test_idx = np.load(test_split_path)
    else:
        test_idx = np.load(MODEL_DIR / "test_idx.npy")

    max_valid = embedding_full.shape[0]
    test_idx = test_idx[test_idx < max_valid]

    print("[debug] embedding_full shape:", embedding_full.shape)
    print("[debug] test_idx max:", test_idx.max(), "len:", len(test_idx))

    # === Slice to test split only ===
    embedding = embedding_full[test_idx]
    y = emotion_tensor_full[test_idx]

    X_behavior = embedding[:, slice(*BEHAVIOR_INDICES)]

    # === Linear Regression (R² score) ===
    R2_behavior = LinearRegression().fit(X_behavior, y).score(X_behavior, y)
    print(f"[R²] Behavior contrastive: {R2_behavior:.2f}")

    # === KNN Classification (accuracy) ===
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_behavior, y)
    y_pred = knn_model.predict(X_behavior)
    avg_knn_behavior = accuracy_score(y, y_pred)
    print(f"[KNN] Behavior contrastive: {avg_knn_behavior:.2f}")

    # === 3D Embedding Plot ===
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        embedding[:, 0], embedding[:, 1], embedding[:, 2],
        c=y, s=1, cmap="tab10"
    )
    ax.set_title(f'Embedding (behavior), KNN/R²: {avg_knn_behavior:.2f}/{R2_behavior:.2f}',
                 y=1.0, pad=-10)
    ax.set_axis_off()
    plt.savefig(embedding_plot, dpi=300)
    plt.close()

    # === Confusion Matrix ===
    unique_labels = np.unique(y)
    filtered_labels = [EMOTION_MAP[i] for i in unique_labels]
    c_m = confusion_matrix(y, y_pred, labels=unique_labels)
    ConfusionMatrixDisplay(confusion_matrix=c_m,
                           display_labels=filtered_labels).plot(
        xticks_rotation=45, cmap="Blues")
    plt.title("Confusion Matrix - Emotion Classification (Test Set)")
    plt.savefig(confusion_plot, dpi=300)
    plt.close()

    return avg_knn_behavior  # return only KNN since you care about that


def save_knn_summary(out_dir, pid, knn_acc):
    summary_path = out_dir / "summary.csv"
    file_exists = summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["PatientID", "KNN_behavior"])
        writer.writerow([pid, knn_acc])
    print(f"Saved KNN summary to {summary_path}")


if __name__ == "__main__":
    pid = int(float(os.environ["PATIENT_ID"]))
    _, patient_id = PATIENT_CONFIG[pid]

    base_dir = Path(f"output_xCEBRA_cov/{patient_id}/OFC_only_model")
    out_dir = base_dir / "evaluation_outputs"

    knn_acc = evaluate_embedding(
        embedding_path=base_dir / "embedding.pt",
        out_dir=out_dir,
        embedding_plot=out_dir / "embedding.png",
        confusion_plot=out_dir / "confusion_matrix.png",
    )

    save_knn_summary(Path(f"output_xCEBRA_cov/aggregate_outputs/OFC_only_model"), patient_id, knn_acc)