# RUN WITH: python -m src.evaluate_supervised

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier

from src.config import (
    EMBEDDING_PATH, EMOTION_TENSOR_PATH,
    BEHAVIOR_INDICES, EMOTION_MAP,
    EVALUATION_OUTPUT_DIR, EVALUATION_EMBEDDING_PLOT, EVALUATION_CONFUSION_PLOT
)


def evaluate_embedding():
    # === Ensure output directories exist ===
    os.makedirs("evaluation_outputs", exist_ok=True)
    EVALUATION_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # === Load embedding and label tensors ===
    embedding = torch.load(EMBEDDING_PATH).numpy()
    emotion_tensor = torch.load(EMOTION_TENSOR_PATH)
    print("Embedding shape:", embedding.shape)

    # Convert emotion tensor to label array
    full_labels = emotion_tensor.squeeze().cpu().numpy().astype(int)
    y = full_labels

    # === Slice the embedding to only behavior-relevant dimensions ===
    X_behavior = embedding[:, slice(*BEHAVIOR_INDICES)]

    # === Linear Regression (R² score) ===
    linear_model_behavior = LinearRegression()
    R2_behavior = linear_model_behavior.fit(X_behavior, y).score(X_behavior, y)
    print(f"[R²] Behavior contrastive   : {R2_behavior:.2f}")

    # === KNN Classification with TimeSeriesSplit ===
    tscv = TimeSeriesSplit(n_splits=5)
    knn_model_behavior = KNeighborsClassifier(n_neighbors=5)

    def evaluate_knn(X, y, model):
        """Run KNN across TSCV splits and return average accuracy."""
        scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            scores.append(accuracy_score(y_test, y_pred))
        return sum(scores) / len(scores)

    avg_knn_behavior = evaluate_knn(X_behavior, y, knn_model_behavior)
    print(f"[KNN] Behavior contrastive   : {avg_knn_behavior:.2f}")

    # === 3D Embedding Plot ===
    fig = plt.figure(figsize=(20, 15))
    idx0, idx1, idx2 = 0, 1, 2
    min_, max_ = 0, 10000

    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(
        embedding[min_:max_, idx0],
        embedding[min_:max_, idx1],
        embedding[min_:max_, idx2],
        c=emotion_tensor[min_:max_], s=1, cmap="tab10"
    )
    ax.set_title(f'Embedding (behavior contrastive), KNN/R²: {avg_knn_behavior:.2f} / {R2_behavior:.2f}',
                 y=1.0, pad=-10)
    ax.set_axis_off()

    plt.savefig(EVALUATION_EMBEDDING_PLOT, dpi=300)
    plt.show()
    print(f"Saved 3D embedding visualization to {EVALUATION_EMBEDDING_PLOT}")

    # === Confusion Matrix from last fold ===
    for train_idx, test_idx in tscv.split(X_behavior):
        pass  # get the last fold

    X_train, X_test = X_behavior[train_idx], X_behavior[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    unique_labels = np.unique(y_test)
    filtered_emotion_labels = [EMOTION_MAP[i] for i in unique_labels]

    c_m = confusion_matrix(y_test, y_pred, labels=unique_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=c_m, display_labels=filtered_emotion_labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=45, cmap='Blues')
    plt.title("Confusion Matrix - Emotion Classification")

    plt.savefig(EVALUATION_CONFUSION_PLOT, dpi=300)
    plt.show()
    print(f"Saved confusion matrix to {EVALUATION_CONFUSION_PLOT}")

    print("Unique labels in test set:", unique_labels)
    print("Filtered emotion labels used in confusion matrix:", filtered_emotion_labels)

    return {
        "embedding": embedding,
        "R2": R2_behavior,
        "KNN": avg_knn_behavior,
        "confusion_matrix": c_m
    }


if __name__ == "__main__":
    evaluate_embedding()