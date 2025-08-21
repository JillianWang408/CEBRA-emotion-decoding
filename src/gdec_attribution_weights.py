# RUN: python -m src.gdec_attribution_weights --feature-mode {lags|cov_ut}
import os
from pathlib import Path
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    MODEL_DIR, ATTRIBUTION_OUTPUT_DIR, OUT_DIR, N_ELECTRODES,
    ELECTRODE_NAMES, EMOTION_MAP
)

# ----- constants (lags mode) -----
N_LAGS        = 5
N_BANDS       = 5
F_PER_LAG     = N_ELECTRODES * N_BANDS               # 200
N_FEATS_LAGS  = N_LAGS * F_PER_LAG                   # 1000
N_FEATS_COVUT = (N_ELECTRODES * (N_ELECTRODES + 1)) // 2  # 820

def ids_to_names(id_list):
    if EMOTION_MAP is None:
        return [str(int(i)) for i in id_list]
    if isinstance(EMOTION_MAP, dict):
        return [EMOTION_MAP.get(int(i), str(int(i))) for i in id_list]
    return [EMOTION_MAP[int(i)] if int(i) < len(EMOTION_MAP) else str(int(i)) for i in id_list]

def _first_existing(*paths: Path) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None

def load_weights_and_classes(mode_tag: str, expected_F: int):
    ROOTS_COEFS = [
        Path(OUT_DIR) / mode_tag,
        Path(OUT_DIR),
        Path(MODEL_DIR) / "gdec_gpmd" / mode_tag,
        Path(MODEL_DIR) / "gdec_gpmd",
    ]
    ROOTS_CLASSES = ROOTS_COEFS

    p_classes = _first_existing(
        *(r / "classes_seen_patient.npy" for r in ROOTS_CLASSES),
        *(r / "classes_seen.npy" for r in ROOTS_CLASSES),
    )
    if p_classes is None:
        raise FileNotFoundError(
            f"[attr] Could not find classes_seen_(patient|final).npy in any of: "
            + ", ".join(str(r) for r in ROOTS_CLASSES)
        )
    classes_seen = np.load(p_classes)

    p_coefs = _first_existing(*(r / "coefs_raw.npy" for r in ROOTS_COEFS))
    if p_coefs is not None:
        W = np.load(p_coefs)
    else:
        W = None
        for r in ROOTS_COEFS:
            for name in ["gpmd_model_best.joblib", "gpmd_model_patient.joblib", "gpmd_model_final.joblib"]:
                mp = r / name
                if mp.exists():
                    model = joblib.load(mp)
                    W = getattr(model, "coefs_", None)
                    if W is None:
                        continue
                    break
            if W is not None:
                break
        if W is None:
            raise FileNotFoundError(
                f"[attr] No weights found (coefs_raw.npy or gpmd_model_*.joblib) in any of: "
                + ", ".join(str(r) for r in ROOTS_COEFS)
            )

    assert W.ndim == 2 and W.shape[1] == expected_F, f"[attr] Expected [K,{expected_F}] got {W.shape}"
    return W, classes_seen

def reshape_weights_lags(W_flat: np.ndarray) -> np.ndarray:
    # [K, 1000] → [K, 5 (lags), 40 (elec), 5 (bands)] for lag-major layout
    K, F = W_flat.shape
    assert F == N_FEATS_LAGS
    return W_flat.reshape(K, N_LAGS, N_ELECTRODES, N_BANDS)

def class_matrix_200x5(W4D_k: np.ndarray, use_abs: bool = False) -> np.ndarray:
    # [lags, elec, band] → [200 rows (elec×band), 5 cols (lags)]
    M = np.transpose(W4D_k, (1, 2, 0)).reshape(N_ELECTRODES * N_BANDS, N_LAGS)
    return np.abs(M) if use_abs else M

def row_labels():
    # label each block of 5 rows (bands) with the electrode name
    if ELECTRODE_NAMES is None or len(ELECTRODE_NAMES) != N_ELECTRODES:
        names = [f"E{i:02d}" for i in range(N_ELECTRODES)]
    else:
        names = ELECTRODE_NAMES
    out = []
    for e in range(N_ELECTRODES):
        out.extend([names[e]] + [""] * (N_BANDS - 1))
    return out  # length 200

def plot_heatmap(
    mat,
    title: str,
    save_path: Path,
    center_zero: bool,
    xlabels=None,  # FIX: default to None; caller supplies correct labels per mode
    ylabels=None,  # FIX: default to None; caller supplies correct labels per mode
    max_xticks: int = 9999,  # FIX: show all by default
    max_yticks: int = 9999,  # FIX: show all by default
    xrotation: int = 45,     # 45° tilt for x-axis labels
    tick_fontsize: int = 8,
):
    fig, ax = plt.subplots(figsize=(10, 10 if mat.shape[0] == mat.shape[1] else 12), dpi=160)

    if center_zero and (np.nanmin(mat) < 0) and (np.nanmax(mat) > 0):
        im = ax.imshow(mat, aspect="auto", cmap="coolwarm", origin="upper")
        vmax = np.nanmax(np.abs(mat))
        im.set_clim(-vmax, vmax)
    else:
        im = ax.imshow(mat, aspect="auto", cmap="viridis", origin="upper")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.03)
    cbar.ax.tick_params(labelsize=tick_fontsize)

    def _set_ticks(axis: str, labels, max_n, rotation=0):
        if labels is None:
            return
        n = len(labels)
        step = max(1, int(np.ceil(n / max_n)))
        idx = np.arange(0, n, step)
        show = [labels[i] for i in idx]
        if axis == "x":
            ax.set_xticks(idx)
            ax.set_xticklabels(show, rotation=rotation, ha="right", fontsize=tick_fontsize)
        else:
            ax.set_yticks(idx)
            ax.set_yticklabels(show, fontsize=tick_fontsize)

    _set_ticks("x", xlabels, max_xticks, rotation=xrotation)
    _set_ticks("y", ylabels, max_yticks)

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def vech_to_sym_matrix(w_ut: np.ndarray) -> np.ndarray:
    """Map upper-triangular vector (diag included) back to symmetric [n,n]."""
    n = N_ELECTRODES
    iu = np.triu_indices(n, k=0)
    M = np.zeros((n, n), dtype=w_ut.dtype)
    M[iu] = w_ut
    M = M + np.triu(M, 1).T
    return M

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-mode", choices=["lags", "cov"], default="lags",
                        help="Which feature basis to use for loading/reshaping weights.")
    args = parser.parse_args()

    pid = os.environ.get("PATIENT_ID", "unknown")
    mode_tag = args.feature_mode  # "lags" or "cov"

    # Mode-specific output root for attribution artifacts
    subdir = "gdec_weights_lags" if mode_tag == "lags" else "gdec_weights_cov"
    out_root = Path(ATTRIBUTION_OUTPUT_DIR) / mode_tag / subdir
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[attr-gdec] writing to: {out_root}")

    expected_F = N_FEATS_LAGS if mode_tag == "lags" else N_FEATS_COVUT
    W_flat, classes_seen = load_weights_and_classes(mode_tag, expected_F)
    class_names = ids_to_names(list(classes_seen))
    print(f"[attr-gdec] Patient {pid} | feature-mode={mode_tag} | weights {W_flat.shape} | classes {list(classes_seen)}")
    print(f"[attr-gdec] writing to: {out_root}")

    # Electrode names helper (40 length)
    elect_names = ELECTRODE_NAMES if (ELECTRODE_NAMES and len(ELECTRODE_NAMES) == N_ELECTRODES) \
        else [f"E{i:02d}" for i in range(N_ELECTRODES)]

    if mode_tag == "lags":
        # ---- LAGS: per-class 200x5 + aggregates ----
        W4D = reshape_weights_lags(W_flat)  # [K, 5, 40, 5]
        np.save(out_root / "weights_reshaped_Kx5x40x5.npy", W4D)
        np.save(out_root / "weights_abs_reshaped_Kx5x40x5.npy", np.abs(W4D))

        mats = []
        for k in range(W4D.shape[0]):
            M = class_matrix_200x5(W4D[k], use_abs=False)  # [200,5]
            mats.append(M)
            # FIX: xlabels are lag names; ylabels are 200-row electrode-block labels
            plot_heatmap(
                M,
                f"GDEC weights — class {class_names[k]} (rows: Elec×Band, cols: Lags)",
                out_root / f"class_{classes_seen[k]}_weights_200x5.png",
                center_zero=True,
                xlabels=[f"Lag {i+1}" for i in range(N_LAGS)],
                ylabels=row_labels(),
                max_xticks=N_LAGS,
                max_yticks=N_ELECTRODES * N_BANDS,
            )
        mats = np.stack(mats, axis=0)  # [K, 200, 5]
        np.save(out_root / "weights_per_class_200x5.npy", mats)

        l2_agg = np.sqrt(np.sum(mats**2, axis=0))  # [200, 5]
        signed_mean = np.mean(mats, axis=0)        # [200, 5]
        np.save(out_root / "weights_l2_agg_200x5.npy", l2_agg)
        np.save(out_root / "weights_signed_mean_200x5.npy", signed_mean)

        plot_heatmap(
            l2_agg, "GDEC weights — L2 across classes",
            out_root / "weights_l2_agg_200x5.png",
            center_zero=False,
            xlabels=[f"Lag {i+1}" for i in range(N_LAGS)],
            ylabels=row_labels(),
            max_xticks=N_LAGS,
            max_yticks=N_ELECTRODES * N_BANDS,
        )
        plot_heatmap(
            signed_mean, "GDEC weights — signed mean across classes",
            out_root / "weights_signed_mean_200x5.png",
            center_zero=True,
            xlabels=[f"Lag {i+1}" for i in range(N_LAGS)],
            ylabels=row_labels(),
            max_xticks=N_LAGS,
            max_yticks=N_ELECTRODES * N_BANDS,
        )

    else:
        # ---- COVARIANCE (upper-triangular): per-class 40x40 + aggregates ----
        K, F = W_flat.shape
        assert F == N_FEATS_COVUT
        mats = []
        for k in range(K):
            M = vech_to_sym_matrix(W_flat[k])  # [40,40]
            mats.append(M)
            plot_heatmap(
                M, f"GDEC weights (cov) — class {class_names[k]}",
                out_root / f"class_{classes_seen[k]}_weights_40x40.png",
                center_zero=True,
                xlabels=elect_names,   # electrode names on both axes
                ylabels=elect_names,
                max_xticks=N_ELECTRODES,
                max_yticks=N_ELECTRODES,
            )
        mats = np.stack(mats, axis=0)  # [K, 40, 40]
        np.save(out_root / "weights_per_class_40x40.npy", mats)

        l2_agg = np.sqrt(np.sum(mats**2, axis=0))  # [40,40]
        signed_mean = np.mean(mats, axis=0)        # [40,40]
        np.save(out_root / "weights_l2_agg_40x40.npy", l2_agg)
        np.save(out_root / "weights_signed_mean_40x40.npy", signed_mean)

        plot_heatmap(
            l2_agg, "GDEC weights (cov) — L2 across classes",
            out_root / "weights_l2_agg_40x40.png",
            center_zero=False,
            xlabels=elect_names,
            ylabels=elect_names,
            max_xticks=N_ELECTRODES,
            max_yticks=N_ELECTRODES,
        )
        plot_heatmap(
            signed_mean, "GDEC weights (cov) — signed mean across classes",
            out_root / "weights_signed_mean_40x40.png",
            center_zero=True,
            xlabels=elect_names,
            ylabels=elect_names,
            max_xticks=N_ELECTRODES,
            max_yticks=N_ELECTRODES,
        )

    print(f"[attr-gdec] saved arrays and figures to: {out_root}")

if __name__ == "__main__":
    main()
