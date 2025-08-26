import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- Your config with the patient mapping ---
PATIENT_CONFIG = {
    1:    ("EC238", "238"),
    2:    ("EC239", "239"),
    9:    ("EC272", "272"),
    27:   ("EC301", "301"),
    28:   ("EC304", "304"),

    15: ("EC280", "280"), 
    22: ("EC288", "288"),
    24: ("EC293", "293"),
    29: ("PR06", "PR06"),
    30: ("EC325", "325"),
    31: ("EC326", "326"),
}

OUTPUT_DIR_NAME = "model_comparison"
MODELS = ["xcebra", "glmnet", "gdec"]
FEATURE_MODES = ["lags", "cov"]
LABELS = {
    ("xcebra", "lags"): "xCEBRA — lags",
    ("xcebra", "cov"):  "xCEBRA — cov",
    ("glmnet", "lags"): "GLMNET — lags",
    ("glmnet", "cov"):  "GLMNET — cov",
    ("gdec",   "lags"): "GDEC — lags",
    ("gdec",   "cov"):  "GDEC — cov",
}

# --- style maps: same color per model; dashed for lags, solid for cov (虚线/实线) ---
MODEL_COLORS = {
    "xcebra": "C0",
    "glmnet": "C1",
    "gdec":   "C2",
}
FEATURE_LINESTYLE = {
    "lags": "--",   # 虚线
    "cov":  "-",    # 实线
}

# --- options ---
SHOW_EMPTY_SERIES_IN_LEGEND = True   # show "(no data)" legend entries
DEBUG = True                         # print a per-series debug report

def _looks_like_project_root(p: Path) -> bool:
    return any((p / name).exists() for name in ["output_glmnet", "output_gdec", "output_xCEBRA_lags", "output_xCEBRA_cov"])

def find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for up in [here, *here.parents]:
        if _looks_like_project_root(up):
            return up
    return Path.cwd()

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", find_project_root())).resolve()
OUTPUT_DIR = PROJECT_ROOT / OUTPUT_DIR_NAME
PNG_PATH = OUTPUT_DIR / "accuracy_across_patients.png"
CSV_PATH = OUTPUT_DIR / "accuracy_across_patients.csv"

print(f"[one-click] PROJECT_ROOT = {PROJECT_ROOT}")
print(f"[one-click] Output dir   = {OUTPUT_DIR}")

# ---------- Build mapping from PATIENT_CONFIG ----------
_raw_to_display: Dict[str, str] = {}
_display_order: List[str] = []
_seen = set()
for k, (code, alt) in PATIENT_CONFIG.items():
    display = code  # show "EC…" on axis; change to `alt` to show numeric instead
    if display not in _seen:
        _display_order.append(display); _seen.add(display)
    _raw_to_display[str(k)] = display
    if isinstance(k, (int, float)):
        _raw_to_display[str(int(k))] = display
        _raw_to_display[str(float(k))] = display
    _raw_to_display[str(code)] = display
    _raw_to_display[str(alt)] = display

# also prepare sets we can search for in paths
ALL_TOKENS = set(_raw_to_display.keys())

# ---------- parsing ----------
PATIENT_LINE_RE = re.compile(r"^\s*Patient\s+(.+?)\s*$", re.IGNORECASE)
MEAN_STD_RE = re.compile(r"Mean\s*[±+\-/]\s*SD\s*:\s*([0-9]*\.?[0-9]+)\s*[±+\-/]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

def parse_summary(summary_path: Path) -> Optional[Tuple[str, float, float]]:
    try:
        lines = summary_path.read_text().strip().splitlines()
    except Exception:
        return None
    raw = None
    mean = std = None
    for ln in lines[:3]:
        m = PATIENT_LINE_RE.match(ln)
        if m:
            raw = m.group(1).strip()
            break
    for ln in lines:
        m = MEAN_STD_RE.search(ln)
        if m:
            mean, std = float(m.group(1)), float(m.group(2))
            break
    if raw is None or mean is None or std is None:
        return None
    return raw, float(np.clip(mean, 0.0, 1.0)), float(max(0.0, std))

def _coerce_numeric(s: str):
    try:
        f = float(s)
        if np.isfinite(f):
            return f
    except Exception:
        pass
    try:
        return int(s)
    except Exception:
        return None

def to_display_label(raw_patient: str, summary_path: Optional[Path] = None) -> Optional[str]:
    """Map 'Patient …' token to display label; fallback: infer from path segments."""
    # direct match
    if raw_patient in _raw_to_display:
        return _raw_to_display[raw_patient]
    # tolerate 'P09', spaces, etc.
    rp = raw_patient.strip().replace("P", "").replace("p", "")
    if rp in _raw_to_display:
        return _raw_to_display[rp]
    num = _coerce_numeric(rp)
    if num is not None:
        for key in (str(num), str(int(num)), str(float(num))):
            if key in _raw_to_display:
                return _raw_to_display[key]
    # last resort: scan the path segments for any known token (EC*, PR06, numeric alt)
    if summary_path is not None:
        parts = " ".join(p.name for p in summary_path.parents)
        for tok in ALL_TOKENS:
            if tok and tok in parts:
                return _raw_to_display[tok]
    return None

# ---------- collectors ----------
def collect_glmnet_or_gdec(model: str, feature_mode: str) -> Tuple[Dict[str, Tuple[float, float]], List[str]]:
    base = PROJECT_ROOT / f"output_{model}"
    got: Dict[str, Tuple[float, float]] = {}
    notes: List[str] = []
    if not base.exists():
        notes.append(f"base missing: {base}")
        return got, notes
    for outdir in base.iterdir():
        if not outdir.is_dir():
            continue
        p = outdir / "evaluation_outputs" / feature_mode / "cv" / "summary.txt"
        if not p.exists():
            notes.append(f"missing: {p}")
            continue
        parsed = parse_summary(p)
        if not parsed:
            notes.append(f"parse fail: {p}")
            continue
        raw, mean, std = parsed
        disp = to_display_label(raw, p)
        if disp is None:
            notes.append(f"unmapped '{raw}' in {p}")
            continue
        got[disp] = (mean, std)
    return got, notes

def collect_xcebra(feature_mode: str) -> Tuple[Dict[str, Tuple[float, float]], List[str]]:
    base = PROJECT_ROOT / ("output_xCEBRA_lags" if feature_mode == "lags" else "output_xCEBRA_cov")
    got: Dict[str, Tuple[float, float]] = {}
    notes: List[str] = []
    if not base.exists():
        notes.append(f"base missing: {base}")
        return got, notes
    for outdir in base.iterdir():
        if not outdir.is_dir():
            continue
        p = outdir / "evaluation_outputs" / "summary.txt"
        if not p.exists():
            notes.append(f"missing: {p}")
            continue
        parsed = parse_summary(p)
        if not parsed:
            notes.append(f"parse fail: {p}")
            continue
        raw, mean, std = parsed
        disp = to_display_label(raw, p)
        if disp is None:
            notes.append(f"unmapped '{raw}' in {p}")
            continue
        got[disp] = (mean, std)
    return got, notes

def collect_all():
    data: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]] = {}
    dbg: Dict[Tuple[str, str], List[str]] = {}
    for model in ["glmnet", "gdec"]:
        for feat in FEATURE_MODES:
            m, notes = collect_glmnet_or_gdec(model, feat)
            data[(model, feat)] = m
            dbg[(model, feat)] = notes
    for feat in FEATURE_MODES:
        m, notes = collect_xcebra(feat)
        data[("xcebra", feat)] = m
        dbg[("xcebra", feat)] = notes
    return data, dbg

# ---------- series ----------
def build_series(axis_labels: List[str],
                 data_map: Dict[str, Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    means = np.array([data_map.get(lbl, (np.nan, np.nan))[0] for lbl in axis_labels], dtype=float)
    stds  = np.array([data_map.get(lbl, (np.nan, np.nan))[1] for lbl in axis_labels], dtype=float)
    stds[np.isnan(means)] = np.nan
    return means, stds

# ---------- plot & csv ----------
def plot_lines_with_errorbars(axis_labels: List[str],
                              acc_series: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]],
                              save_png: Path):
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(axis_labels))
    empty_entries = []  # (label, color, linestyle)

    for model in MODELS:
        for feat in FEATURE_MODES:
            key = (model, feat)
            means, stds = acc_series.get(key, (None, None))
            color = MODEL_COLORS.get(model, None)
            linestyle = FEATURE_LINESTYLE.get(feat, "-")

            if means is None or np.all(np.isnan(means)):
                empty_entries.append((LABELS.get(key, f"{model}-{feat}"), color, linestyle))
                continue

            ax.errorbar(
                x, means, yerr=stds,
                marker="o",
                linestyle=linestyle,
                color=color,
                capsize=3, elinewidth=1, linewidth=1.7,
                label=LABELS.get(key, f"{model}-{feat}")
            )


    ax.set_xticks(x)
    ax.set_xticklabels(axis_labels, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Patient (correspondence ID)")
    ax.set_ylabel("Accuracy (mean ± SD)")
    ax.set_title("Evaluation Accuracy by Patient — xCEBRA • GLMNET • GDEC (lags & cov)")
    ax.grid(True, linestyle="--", alpha=0.35)
    legend = ax.legend(ncol=3, frameon=False)

    # Add placeholders for empty series to keep all six in legend with correct style
    if SHOW_EMPTY_SERIES_IN_LEGEND and empty_entries:
        handles = list(legend.legend_handles) if legend else []
        labels = [t.get_text() for t in legend.texts] if legend else []
        for lab, col, ls in empty_entries:
            handles.append(Line2D([0], [0], linestyle=ls, marker="o",
                                  alpha=0.5, color=col))
            labels.append(f"{lab} (no data)")
        if legend:
            legend.remove()
        ax.legend(handles, labels, ncol=3, frameon=False)

    fig.tight_layout()
    save_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_png, dpi=200, bbox_inches="tight")
    print(f"[one-click] Saved figure → {save_png}")

def export_csv(axis_labels: List[str],
               acc_series: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]],
               out_csv: Path):
    header = ["patient"] + [f"{m}_{f}_mean" for m in MODELS for f in FEATURE_MODES] + \
                           [f"{m}_{f}_std"  for m in MODELS for f in FEATURE_MODES]
    rows = []
    for i, lbl in enumerate(axis_labels):
        row = [lbl]
        for m in MODELS:
            for f in FEATURE_MODES:
                means, stds = acc_series.get((m, f), (None, None))
                mv = "" if means is None or np.isnan(means[i]) else f"{means[i]:.6f}"
                row.append(mv)
        for m in MODELS:
            for f in FEATURE_MODES:
                means, stds = acc_series.get((m, f), (None, None))
                sv = "" if stds is None or np.isnan(stds[i]) else f"{stds[i]:.6f}"
                row.append(sv)
        rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
        pd.DataFrame(rows, columns=header).to_csv(out_csv, index=False)
    except Exception:
        import csv
        with out_csv.open("w", newline="") as f:
            csv.writer(f).writerows([header] + rows)
    print(f"[one-click] Saved CSV → {out_csv}")

# ---------- debug print ----------
def debug_report(axis_labels: List[str],
                 data_maps: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]],
                 dbg_notes: Dict[Tuple[str, str], List[str]]):
    print("\n[debug] Series coverage:")
    for model in MODELS:
        for feat in FEATURE_MODES:
            key = (model, feat)
            m = data_maps.get(key, {})
            have = sorted([k for k in m.keys() if k in axis_labels], key=lambda s: axis_labels.index(s))
            print(f"  - {LABELS[key]}: {len(have)} patients → {have}")
            notes = dbg_notes.get(key, [])
            for n in notes:
                print(f"      · {n}")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    axis_labels = list(_display_order)
    if not axis_labels:
        raise SystemExit("PATIENT_CONFIG produced no axis labels — check src.config.PATIENT_CONFIG")

    data_maps, dbg_notes = collect_all()

    if DEBUG:
        debug_report(axis_labels, data_maps, dbg_notes)

    acc_series: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
    for key, m in data_maps.items():
        acc_series[key] = build_series(axis_labels, m)

    plot_lines_with_errorbars(axis_labels, acc_series, save_png=PNG_PATH)
    export_csv(axis_labels, acc_series, CSV_PATH)

if __name__ == "__main__":
    main()
