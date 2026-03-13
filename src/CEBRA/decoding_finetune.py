"""
CEBRA decoding finetune adapted for calc/pred data.
Train decoder on calc embeddings, evaluate on pred embeddings.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.CEBRA.config_loader import load_cebra_config
from src.general.utils_visualization import (
    collect_decoding_timecourse,
    save_decoding_timecourse,
    plot_decoding_timecourses,
    plot_confusion_matrix_heatmap,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_embedding_TxD(emb_path: Path) -> np.ndarray:
    embedding = torch.load(emb_path, map_location="cpu", weights_only=False)
    if embedding.ndim == 3 and embedding.shape[0] == 1:
        embedding = embedding.squeeze(0).T
    return embedding.numpy()


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (nrm + eps)


def main():
    parser = argparse.ArgumentParser(description="CEBRA decoding (calc=train, pred=test)")
    parser.add_argument("--patient-id", type=int, default=None)
    parser.add_argument("--target", type=str, default="9emotion",
        choices=["9emotion", "arousal", "valence", "categories"])
    args = parser.parse_args()

    patient_id = args.patient_id or int(float(os.environ["PATIENT_ID"]))
    cfg = load_cebra_config(patient_id, args.target)

    # Prefer models_finetune (if encoding_finetune was run), else xcebra_supervised
    finetune_dir = cfg["model_dir"] / "models_finetune"
    sup_dir = cfg["model_dir"] / "xcebra_supervised"
    if (finetune_dir / "embedding_calc.pt").exists():
        pt_dir = finetune_dir
    elif (sup_dir / "embedding_calc.pt").exists():
        pt_dir = sup_dir
    else:
        raise FileNotFoundError(
            f"Run encoding (and optionally encoding_finetune) first. "
            f"Missing embedding_calc.pt in {finetune_dir} or {sup_dir}"
        )

    out_dir = cfg["output_base"] / "decoding"
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_calc_path = pt_dir / "embedding_calc.pt"
    emb_pred_path = pt_dir / "embedding_pred.pt"
    labels_calc_path = cfg["model_dir"] / "labels_calc.pt"
    labels_pred_path = cfg["model_dir"] / "labels_pred.pt"

    if not emb_calc_path.exists() or not emb_pred_path.exists():
        raise FileNotFoundError(f"Missing: {emb_calc_path} or {emb_pred_path}")

    X_train = load_embedding_TxD(emb_calc_path)
    X_test = load_embedding_TxD(emb_pred_path)

    y_train = torch.load(labels_calc_path, map_location="cpu", weights_only=False)
    y_test = torch.load(labels_pred_path, map_location="cpu", weights_only=False)
    if y_train.ndim > 1:
        y_train = y_train.squeeze(-1)
    if y_test.ndim > 1:
        y_test = y_test.squeeze(-1)
    y_train = y_train.numpy().ravel().astype(int)
    y_test = y_test.numpy().ravel().astype(int)

    # Align lengths (embedding may be shorter due to conv)
    T_train = min(X_train.shape[0], len(y_train))
    T_test = min(X_test.shape[0], len(y_test))
    X_train = X_train[:T_train]
    X_test = X_test[:T_test]
    y_train = y_train[:T_train]
    y_test = y_test[:T_test]

    # Filter test samples with labels not seen in train
    train_classes = np.unique(y_train)
    mask = np.isin(y_test, train_classes)
    if mask.sum() < len(y_test):
        print(f"[WARN] Excluding {(~mask).sum()} test samples with unseen labels")
    X_test = X_test[mask]
    y_test = y_test[mask]

    X_train = l2_normalize_rows(X_train)
    X_test = l2_normalize_rows(X_test)

    # Scale for numerical stability (helps when using --steps 50 quick test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train/cal split for LogReg
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tr_idx, cal_idx = next(sss.split(X_train, y_train))
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_cal, y_cal = X_train[cal_idx], y_train[cal_idx]

    C_grid = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    best_lr, best_C, best_f1 = None, None, -1.0
    for C in C_grid:
        lr = LogisticRegression(solver="lbfgs", class_weight="balanced", C=C, max_iter=5000, n_jobs=1)
        lr.fit(X_tr, y_tr)
        f1 = f1_score(y_cal, lr.predict(X_cal), average="macro")
        if f1 > best_f1:
            best_f1, best_C, best_lr = f1, C, lr

    cal = CalibratedClassifierCV(best_lr, method="isotonic", cv="prefit")
    cal.fit(X_cal, y_cal)
    y_pred = cal.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"[Results] Accuracy: {acc:.4f}, Macro-F1: {macro_f1:.4f}")

    # Save
    np.save(out_dir / "y_test.npy", y_test)
    np.save(out_dir / "y_pred.npy", y_pred)

    df_summary = pd.DataFrame([{
        "patient": cfg["patient_code"],
        "target": args.target,
        "accuracy": acc,
        "macroF1": macro_f1,
    }])
    df_summary.to_csv(out_dir / "decoding_summary.csv", index=False)

    # Timecourse
    df_tc = collect_decoding_timecourse(
        pair_name=f"CEBRA_{args.target}",
        y_true=y_test,
        y_pred=y_pred,
        test_idx=np.arange(len(y_test)),
    )
    save_decoding_timecourse([df_tc], out_dir / "decoding_timecourse.csv")
    plot_decoding_timecourses(
        csv_path=out_dir / "decoding_timecourse.csv",
        out_path=out_dir / "decoding_timecourse_grid.png",
        emotion_map=cfg["label_map"],
        n_cols=1,
    )

    # Confusion matrix
    plot_confusion_matrix_heatmap(
        y_true=y_test,
        y_pred=y_pred,
        label_map=cfg["label_map"],
        out_path=out_dir / "confusion_matrix.png",
        title=f"CEBRA {args.target.capitalize()} Decoding – Confusion Matrix",
    )

    print(f"[DONE] Decoding saved to {out_dir}")


if __name__ == "__main__":
    main()
