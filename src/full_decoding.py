import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    r2_score
)
import matplotlib.pyplot as plt
from src.config import FULL_EMOTION_PATH, MODEL_DIR, PATIENT_CONFIG, EMOTION_MAP
from src.utils_decoding import evaluate_embedding, load_embedding_TxD, _split_train_test
from src.utils_visualization import (
    collect_decoding_timecourse,
    save_decoding_timecourse,
    plot_decoding_timecourses,
)

DECODERS = ["knn", "logreg"]

# Helper: dwell-time (segment length) computation
def compute_dwell_times(labels: np.ndarray):
    """Return list of dwell times (consecutive same-label segment lengths)."""
    if len(labels) == 0:
        return np.array([])
    dwell = []
    current_len = 1
    for i in range(1, len(labels)):
        if labels[i] == labels[i - 1]:
            current_len += 1
        else:
            dwell.append(current_len)
            current_len = 1
    dwell.append(current_len)
    return np.array(dwell)


def main():
    pid = int(float(os.environ["PATIENT_ID"]))
    _, patient_id = PATIENT_CONFIG[pid]

    enc_dir = MODEL_DIR / "xcebra_supervised"       # supervised embedding path
    dec_dir = MODEL_DIR.parent / "full_decoding"           # output directory
    dec_dir.mkdir(parents=True, exist_ok=True)
    all_timecourse = []

    print(f"[INFO] Starting decoding for patient {patient_id}")

    # Load embedding and labels
    emb_path = enc_dir / "embedding.pt"
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing embedding file: {emb_path}")

    embedding = load_embedding_TxD(emb_path)

    emotion_tensor_full = torch.load(FULL_EMOTION_PATH, map_location="cpu")
    if emotion_tensor_full.ndim == 1:
        emotion_tensor_full = emotion_tensor_full.unsqueeze(1)
    y_full = emotion_tensor_full.float().contiguous().numpy()

    rows = []

    for decoder in DECODERS:
        X_train, X_test, y_train, y_test, offset, split = _split_train_test(
            embedding, y_full
        )

        coef, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
        y_pred_lin = X_test @ coef
        R2_behavior = r2_score(y_test, y_pred_lin)

        log_reg = LogisticRegression(max_iter=2000, solver="lbfgs")
        log_reg.fit(X_train, y_train)
        y_pred_logreg = log_reg.predict(X_test)
        acc_logreg = accuracy_score(y_test, y_pred_logreg)

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        acc_knn = accuracy_score(y_test, y_pred_knn)
        

        if decoder == "knn":
            acc = acc_knn
            y_pred = y_pred_knn
        elif decoder == "logreg":
            acc = acc_logreg
            y_pred = y_pred_logreg
        else:
            continue

        macro_f1 = f1_score(y_test, y_pred, average="macro")
        dwell_pred = compute_dwell_times(y_pred).mean()

        # Save metrics
        result = {
            "patient": patient_id,
            "decoder": decoder,
            "R2_behavior": f"{R2_behavior:.4f}",
            "accuracy": f"{acc:.4f}",
            "macroF1": f"{macro_f1:.4f}",
            "mean_dwell_pred": f"{dwell_pred:.2f}",
        }
        rows.append(result)
        print(f"[ok] [{decoder}] acc={acc:.3f}, R²={R2_behavior:.3f}, F1={macro_f1:.3f}, dwell={dwell_pred:.1f}")

        # Confusion matrix
        unique_labels = np.unique(np.concatenate([y_test, y_pred]))
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        label_names = [EMOTION_MAP[int(l)] for l in unique_labels]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
        disp.plot(xticks_rotation=90, cmap="Blues")
        plt.title(f"{decoder} confusion matrix (patient {patient_id})")
        plt.tight_layout()
        plt.savefig(dec_dir / f"cm_{decoder}_{patient_id}.png", dpi=150)
        plt.close()

        test_idx_local = np.arange(split, split + len(y_test))

        df_pair = collect_decoding_timecourse(
            pair_name=f"{decoder}_supervised",
            y_true=y_test,
            y_pred=y_pred,
            test_idx=test_idx_local,
        )
        all_timecourse.append(df_pair)

    # ------------------------------
    # Save decoding summary and plots
    # ------------------------------
    df_summary = pd.DataFrame(rows)
    summary_path = dec_dir / "decoding_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"[done] wrote decoding summary → {summary_path}")

    # Timecourse visualization
    df_all = save_decoding_timecourse(all_timecourse, dec_dir / "decoding_timecourse.csv")
    if df_all is not None:
        plot_decoding_timecourses(
            csv_path=dec_dir / "decoding_timecourse.csv",
            out_path=dec_dir / "decoding_timecourse_grid.png",
            emotion_map=EMOTION_MAP,
            n_cols=4,
        )

    print(f"[FINISHED] Full decoding complete for patient {patient_id}.")


if __name__ == "__main__":
    main()
