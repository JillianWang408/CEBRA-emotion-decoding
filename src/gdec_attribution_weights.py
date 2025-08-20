import os
from pathlib import Path
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from src.config import (
    MODEL_DIR,              # e.g. .../output_gdec/<pid>/models
    ATTRIBUTION_OUTPUT_DIR, # e.g. .../output_gdec/<pid>/attribution
    N_ELECTRODES,           # 40
    ELECTRODE_NAMES,        # list of 40 names (optional)
    EMOTION_MAP             # dict or list (optional)
)

# ---- constants matching your feature layout ----
N_LAGS   = 5
N_BANDS  = 5
F_PER_LAG = N_ELECTRODES * N_BANDS     # 40 * 5 = 200
N_FEATS   = N_LAGS * F_PER_LAG         # 1000
BAND_NAMES = ["δ", "θ", "α", "β", "γ"]  # only for doc; rows are (elec,band)

def ids_to_names(id_list):
    if EMOTION_MAP is None:
        return [str(int(i)) for i in id_list]
    if isinstance(EMOTION_MAP, dict):
        return [EMOTION_MAP.get(int(i), str(int(i))) for i in id_list]
    return [EMOTION_MAP[int(i)] if int(i) < len(EMOTION_MAP) else str(int(i)) for i in id_list]

def load_weights_and_classes_gdec(model_root: Path):
    """
    Returns (W, classes_seen):
      W: [K, 1000] weights per class
      classes_seen: [K] label IDs (order matches W rows)
    Prefers coefs_raw.npy; falls back to loading gpmd model joblib.
    """
    classes_paths = [
        model_root / "classes_seen_patient.npy",  # union model
        model_root / "classes_seen.npy",          # fallback
    ]
    classes_seen = None
    for p in classes_paths:
        if p.exists():
            classes_seen = np.load(p)
            break
    if classes_seen is None:
        raise FileNotFoundError(f"classes_seen_{'{patient,'.rstrip()}.npy not found in {model_root}")

    p_coefs = model_root / "coefs_raw.npy"
    if p_coefs.exists():
        W = np.load(p_coefs)  # [K, 1000]
        assert W.ndim == 2 and W.shape[1] == N_FEATS, f"Expected [K,{N_FEATS}], got {W.shape}"
        return W, classes_seen

    # fallback: load model and extract coefs_
    for name in ["gpmd_model_best.joblib", "gpmd_model_patient.joblib", "gpmd_model_final.joblib"]:
        mp = model_root / name
        if mp.exists():
            model = joblib.load(mp)
            W = getattr(model, "coefs_", None)
            if W is None:
                raise AttributeError(f"Model {mp} has no attribute 'coefs_'.")
            assert W.ndim == 2 and W.shape[1] == N_FEATS, f"Expected [K,{N_FEATS}] got {W.shape}"
            return W, classes_seen

    raise FileNotFoundError(f"No weights found in {model_root} (coefs_raw.npy or gpmd_model_*.joblib).")

def reshape_weights_lag_elec_band(W_flat: np.ndarray) -> np.ndarray:
    """
    Correct reshape for your feature order:
      idx = lag * 200 + electrode * 5 + band  (lag-major, then electrode, then band)
    So reshape to [K, 5, 40, 5] in C-order is correct.
    """
    K, F = W_flat.shape
    assert F == N_FEATS, f"Expected {N_FEATS}, got {F}"
    return W_flat.reshape(K, N_LAGS, N_ELECTRODES, N_BANDS)

def class_matrix_200x5(W4D_k: np.ndarray, use_abs: bool = False) -> np.ndarray:
    """
    One class: [5 lags, 40 elec, 5 bands] -> [200 rows (elec×band), 5 cols (lags)]
    """
    M = np.transpose(W4D_k, (1, 2, 0)).reshape(N_ELECTRODES * N_BANDS, N_LAGS)
    return np.abs(M) if use_abs else M

def row_labels():
    if ELECTRODE_NAMES is None or len(ELECTRODE_NAMES) != N_ELECTRODES:
        return [f"E{r//5:02d}" if r % 5 == 0 else "" for r in range(N_ELECTRODES * N_BANDS)]
    return [ELECTRODE_NAMES[r // 5] if (r % 5 == 0) else "" for r in range(N_ELECTRODES * N_BANDS)]

def plot_heatmap(mat_200x5: np.ndarray, title: str, save_path: Path, center_zero: bool):
    plt.figure(figsize=(9, 12))
    if center_zero and (np.nanmin(mat_200x5) < 0) and (np.nanmax(mat_200x5) > 0):
        im = plt.imshow(mat_200x5, aspect="auto", cmap="coolwarm", origin="upper")
        vmax = np.nanmax(np.abs(mat_200x5))
        plt.clim(-vmax, vmax)
    else:
        im = plt.imshow(mat_200x5, aspect="auto", cmap="viridis", origin="upper")
    plt.colorbar(im, fraction=0.025, pad=0.03)
    plt.xticks(range(N_LAGS), [f"Lag {i+1}" for i in range(N_LAGS)])
    plt.yticks(range(N_ELECTRODES * N_BANDS), row_labels())
    plt.xlabel("Time lags")
    plt.ylabel("Electrode × Band (δ..γ within each electrode)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()

def main():
    pid = os.environ.get("PATIENT_ID", "unknown")
    model_root = Path(MODEL_DIR) / "gdec_gpmd"
    out_root = Path(ATTRIBUTION_OUTPUT_DIR) / "gdec_weights"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[attr-gdec] Patient {pid} | loading from: {model_root}")
    W_flat, classes_seen = load_weights_and_classes_gdec(model_root)   # [K, 1000], [K]
    class_names = ids_to_names(list(classes_seen))
    print(f"[attr-gdec] weights shape: {W_flat.shape}, classes: {list(classes_seen)}")

    # reshape → [K, 5, 40, 5]
    W4D = reshape_weights_lag_elec_band(W_flat)
    np.save(out_root / "weights_reshaped_Kx5x40x5.npy", W4D)
    np.save(out_root / "weights_abs_reshaped_Kx5x40x5.npy", np.abs(W4D))

    # per-class 200x5 matrices
    mats = []
    for k in range(W4D.shape[0]):
        M = class_matrix_200x5(W4D[k], use_abs=False)   # signed version
        mats.append(M)
        plot_heatmap(
            M,
            f"GDEC weights — class {class_names[k]} (rows: Elec×Band, cols: Lags)",
            out_root / f"class_{classes_seen[k]}_weights_200x5.png",
            center_zero=True
        )
    mats = np.stack(mats, axis=0)  # [K, 200, 5]
    np.save(out_root / "weights_per_class_200x5.npy", mats)

    # aggregates across classes
    l2_agg = np.sqrt(np.sum(mats**2, axis=0))  # [200, 5]
    signed_mean = np.mean(mats, axis=0)        # [200, 5]
    np.save(out_root / "weights_l2_agg_200x5.npy", l2_agg)
    np.save(out_root / "weights_signed_mean_200x5.npy", signed_mean)

    plot_heatmap(
        l2_agg,
        "GDEC weights — L2 across classes (rows: Elec×Band, cols: Lags)",
        out_root / "weights_l2_agg_200x5.png",
        center_zero=False
    )
    plot_heatmap(
        signed_mean,
        "GDEC weights — signed mean across classes",
        out_root / "weights_signed_mean_200x5.png",
        center_zero=True
    )

    print(f"[attr-gdec] saved arrays and figures to: {out_root}")

if __name__ == "__main__":
    main()