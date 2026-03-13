"""
Single-emotion trial-level training and prediction pipeline.

Trains a classifier on trial-level neural data (each trial = one movie) and predicts
emotion on held-out test trials. Supports logreg, svm, knn, or DPAD. Uses calc files
for training and pred files for testing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

# TF/keras setup before DPAD (same as DPAD_main)
try:
    import tensorflow as tf
    import keras
    if not hasattr(tf, "keras"):
        tf.keras = keras
    if hasattr(tf.keras.optimizers, "legacy") and hasattr(tf.keras.optimizers.legacy, "Adam"):
        tf.keras.optimizers.Adam = tf.keras.optimizers.legacy.Adam
except ImportError:
    tf = None

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.single_emotion.data import (
    load_trial_data,
    concatenate_trials_for_dpad,
    aggregate_dpad_predictions_per_trial,
    reduce_trials_to_features,
)


def _import_dpad_functions():
    from src.DPAD.DPAD_main import (
        train_dpad,
        predict_dpad,
        run_flexible_dpad,
        evaluate_decoding,
    )
    return train_dpad, predict_dpad, run_flexible_dpad, evaluate_decoding


def _load_config(patient_id: int):
    """Load config after setting PATIENT_ID."""
    os.environ["PATIENT_ID"] = str(patient_id)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "config", PROJECT_ROOT / "src" / "config.py"
    )
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Single-emotion trial-level training and prediction"
    )
    parser.add_argument(
        "--patient-id",
        type=int,
        required=True,
        help="Patient ID (1, 2, 9, 27, 28, etc.)",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        choices=["logreg", "svm", "knn", "dpad"],
        default="logreg",
        help="Classifier type (default: logreg). Use dpad for DPAD model (trial concatenation)",
    )
    parser.add_argument(
        "--reduction",
        type=str,
        choices=["mean", "concat", "max"],
        default="mean",
        help="Trial reduction for logreg/svm/knn only (default: mean). Ignored for dpad.",
    )
    parser.add_argument(
        "--skip-flexible",
        action="store_true",
        help="[DPAD only] Skip flexible nonlinearity search",
    )
    parser.add_argument(
        "--nx",
        type=int,
        default=16,
        help="[DPAD only] Total latent dimension",
    )
    parser.add_argument(
        "--n1",
        type=int,
        default=16,
        help="[DPAD only] Behavior-relevant latent dimension",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2500,
        help="[DPAD only] Max training epochs",
    )
    parser.add_argument(
        "--neural-train",
        type=str,
        default=None,
        help="Override path to training neural .mat",
    )
    parser.add_argument(
        "--emotion-train",
        type=str,
        default=None,
        help="Override path to training emotion .mat",
    )
    parser.add_argument(
        "--neural-test",
        type=str,
        default=None,
        help="Override path to test neural .mat",
    )
    parser.add_argument(
        "--emotion-test",
        type=str,
        default=None,
        help="Override path to test emotion .mat",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: output_single_emotion/<patient>)",
    )
    args = parser.parse_args()

    patient_id = args.patient_id
    config = _load_config(patient_id)
    ec_code, output_dir = config.PATIENT_CONFIG[patient_id]
    out_dir = Path(args.output_dir) if args.output_dir else config.SINGLE_EMOTION_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve paths
    neural_train = Path(args.neural_train) if args.neural_train else config.SINGLE_EMOTION_CALC_NEURAL
    emotion_train = Path(args.emotion_train) if args.emotion_train else config.SINGLE_EMOTION_CALC_RESP
    neural_test = Path(args.neural_test) if args.neural_test else config.SINGLE_EMOTION_PRED_NEURAL
    emotion_test = Path(args.emotion_test) if args.emotion_test else config.SINGLE_EMOTION_PRED_RESP

    for p, name in [
        (neural_train, "training neural"),
        (emotion_train, "training emotion"),
        (neural_test, "test neural"),
        (emotion_test, "test emotion"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {name}: {p}")

    print(f"[INFO] Patient {output_dir} ({ec_code})")
    print(f"[INFO] Loading training data...")
    trials_train, y_train, label_to_idx, idx_to_label = load_trial_data(
        neural_train, emotion_train
    )

    print(f"[INFO] Loading test data...")
    trials_test, y_test, _, _ = load_trial_data(neural_test, emotion_test)

    if args.classifier == "dpad":
        # ----- DPAD path: concatenate trials, train DPAD, aggregate predictions per trial -----
        np.random.seed(42)
        if tf is not None:
            tf.random.set_seed(42)
        if tf is None:
            raise ImportError("TensorFlow required for DPAD. Install: pip install -r requirements-dpad.txt")
        (
            train_dpad_fn,
            predict_dpad_fn,
            run_flexible_dpad_fn,
            evaluate_decoding_fn,
        ) = _import_dpad_functions()

        y_train_seq, z_train_seq = concatenate_trials_for_dpad(trials_train, y_train)
        trial_lengths_test = [t.shape[0] for t in trials_test]
        y_test_seq = np.concatenate(trials_test, axis=0).astype(np.float64)

        nx, n1 = args.nx, args.n1
        flex_dir = out_dir / "flexible_search"

        if args.skip_flexible:
            method_code = "DPAD_RTR2_uAKCzCy1HL64U_ErSV16"  # readouts (Cz, Cy) nonlinear; RNN (A, K) linear
            print(f"[DPAD] Using fixed nonlinearity: {method_code}")
        else:
            print("[DPAD] Running flexible nonlinearity search...")
            method_code = run_flexible_dpad_fn(
                y_train_seq, z_train_seq, nx, n1, flex_dir
            )
            print(f"    Selected: {method_code}")

        print("[DPAD] Training...")
        model = train_dpad_fn(
            y_train_seq, z_train_seq,
            nx=nx, n1=n1, method_code=method_code, epochs=args.epochs
        )

        print("[DPAD] Inference on test trials...")
        z_pred_raw, _, x_pred = predict_dpad_fn(model, y_test_seq)
        y_pred = aggregate_dpad_predictions_per_trial(
            z_pred_raw, trial_lengths_test
        )

        nb_classes = len(idx_to_label)
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = evaluate_decoding_fn(y_test, y_pred, n_classes=nb_classes)
        (out_dir / "method_code.txt").write_text(method_code)

        # Save DPAD outputs (same format as DPAD_main)
        z_test_original = np.array([idx_to_label.get(int(i), i) for i in y_test])
        z_pred_original = np.array([idx_to_label.get(int(i), i) for i in y_pred])
        np.save(out_dir / "z_pred.npy", np.asarray(z_pred_raw))
        np.save(out_dir / "x_pred.npy", np.asarray(x_pred))
        np.save(out_dir / "z_test.npy", z_test_original)
        np.save(out_dir / "z_pred_class.npy", z_pred_original)
        with open(out_dir / "label_mapping.json", "w") as f:
            json.dump({
                "original_to_model": {str(k): v for k, v in label_to_idx.items()},
                "model_to_original": {str(k): v for k, v in idx_to_label.items()},
            }, f, indent=2)
        print(f"    Saved z_pred, x_pred, z_test, z_pred_class, label_mapping.json to {out_dir}")
    else:
        # ----- sklearn path: reduce trials to features, train classifier -----
        X_train = reduce_trials_to_features(trials_train, args.reduction)
        X_test = reduce_trials_to_features(trials_test, args.reduction)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if args.classifier == "logreg":
            clf = LogisticRegression(max_iter=2000, solver="lbfgs", multi_class="multinomial")
        elif args.classifier == "svm":
            clf = SVC(kernel="rbf", C=1.0, gamma="scale")
        else:
            clf = KNeighborsClassifier(n_neighbors=5)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"[RESULT] Accuracy: {acc:.4f}, Macro F1: {macro_f1:.4f}")

    # Save metrics
    metrics_path = out_dir / "metrics.txt"
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    train_summary = ", ".join(
        f"{config.EMOTION_MAP.get(idx_to_label.get(int(l), l), str(l))}: {c}"
        for l, c in zip(train_unique, train_counts)
    )
    with open(metrics_path, "w") as f:
        f.write(f"patient: {output_dir}\n")
        f.write(f"classifier: {args.classifier}\n")
        if args.classifier != "dpad":
            f.write(f"reduction: {args.reduction}\n")
        f.write(f"n_train_trials: {len(y_train)}\n")
        f.write(f"train_emotion_summary: {train_summary}\n")
        f.write(f"n_test_trials: {len(y_test)}\n")
        f.write(f"accuracy: {acc:.4f}\n")
        f.write(f"macro_f1: {macro_f1:.4f}\n")
    print(f"[SAVE] Metrics -> {metrics_path}")

    # Confusion matrix
    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    label_names = [config.EMOTION_MAP.get(idx_to_label.get(int(l), l), str(l)) for l in unique_labels]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(xticks_rotation=90, cmap="Blues")
    plt.title(f"Single-emotion trial decoding ({args.classifier}, patient {output_dir})")
    plt.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[SAVE] Confusion matrix -> {cm_path}")

    # Save predictions
    np.savez(
        out_dir / "predictions.npz",
        y_true=y_test,
        y_pred=y_pred,
        idx_to_label=np.array(list(idx_to_label.items())),
    )
    print(f"[SAVE] Predictions -> {out_dir / 'predictions.npz'}")

    print("[DONE] Single-emotion pipeline finished.")


if __name__ == "__main__":
    main()
