# RUN:
# Default (use caches):           python -m src.main
# Force everything:               python -m src.main --feature-mode cov --force
# Only redo grid search:          python -m src.main --feature-mode lags --force-grid
# Change shift sweep window:      SHIFT_MIN=-8 SHIFT_MAX=8 python -m src.main
# Pick dataset feature structure: python -m src.main --feature-mode lags (or cov)
# after changing kernal: python -m src.main --feature-mode lags --force-grid --force-fold-models --force-final --force-shift

import os
import argparse
import subprocess

# Default patients
#PATIENT_IDS = [1, 2, 9, 27, 28, 15, 22, 24, 29, 30, 31]
#PATIENT_IDS = [1, 9, 27, 28] #2(239) not working well
PATIENT_IDS = [29, 30, 31]

PIPELINE_SCRIPTS = [
    "single_node_training"
    #"single_node_eval",
    #"single_node_pairing_training"
    #"single_node_pairing_eval",
    #"single_node_aggregation"
]

def build_forwarded_args_for(script: str, args: argparse.Namespace):
    """Build per-script CLI args (only pass what each child understands)."""
    forwarded = []

    # All scripts understand --feature-mode
    forwarded += ["--feature-mode", args.feature_mode]

    if script == "gdec_model_finetune":
        # training accepts the force/shift flags
        for flag in [
            "force",
            "force_grid",
            "force_fold_models",
            "force_final",
            "force_shift",
            "force_scaler",
        ]:
            if getattr(args, flag):
                forwarded.append("--" + flag.replace("_", "-"))

        # numeric shift window
        forwarded += ["--shift-min", str(args.shift_min), "--shift-max", str(args.shift_max)]

    # evaluation & attribution only need feature-mode, so nothing else is added
    return forwarded

def run_pipeline_for_patient(pid: int, args: argparse.Namespace):
    env = os.environ.copy()
    env["PATIENT_ID"] = str(pid)

    print(f"\n====================== Running pipeline for Patient {pid} ======================")
    for script in PIPELINE_SCRIPTS:
        forwarded_args = build_forwarded_args_for(script, args)
        print(f"\n--- Running {script}.py for Patient {pid} ---")
        cmd = ["caffeinate", "-dimsu", "python", "-m", f"src.{script}", *forwarded_args]
        print("[CMD]", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to run {script}.py for Patient {pid}: {e}")
            return  # stop for this patient if one step fails

def main():
    parser = argparse.ArgumentParser(description="Run xCEBRA training/eval pipeline")
    parser.add_argument("--patients", nargs="*", type=int, default=PATIENT_IDS,
                        help="Patient IDs to process (space-separated)")
    # Global “force” that enables all sub-forces (for training only)
    parser.add_argument("--force", action="store_true",
                        help="Force re-run of ALL training stages")
    # Fine-grained flags (used only by the training module)
    parser.add_argument("--force-grid", action="store_true", help="Force re-run grid search")
    parser.add_argument("--force-fold-models", action="store_true", help="Force re-train per-fold models")
    parser.add_argument("--force-final", action="store_true", help="Force re-train final patient model")
    parser.add_argument("--force-shift", action="store_true", help="Force recompute label shift")
    parser.add_argument("--force-scaler", action="store_true", help="Force recompute scalers")
    parser.add_argument("--shift-min", type=int, default=-5)
    parser.add_argument("--shift-max", type=int, default=5)
    parser.add_argument("--feature-mode", choices=["lags", "cov"], default="lags",
                        help="Choose feature structure for child steps.")
    args = parser.parse_args()

    # Env overrides for shift window (so SHIFT_MIN/MAX=... work as in the header)
    if "SHIFT_MIN" in os.environ:
        args.shift_min = int(os.environ["SHIFT_MIN"])
    if "SHIFT_MAX" in os.environ:
        args.shift_max = int(os.environ["SHIFT_MAX"])

    # If --force, turn on all training sub-forces
    if args.force:
        args.force_grid = True
        args.force_fold_models = True
        args.force_final = True
        args.force_shift = True
        args.force_scaler = True

    print(f"[main] feature-mode={args.feature_mode} | shift window [{args.shift_min}, {args.shift_max}]")
    for pid in args.patients:
        run_pipeline_for_patient(pid, args)

if __name__ == "__main__":
    main()
