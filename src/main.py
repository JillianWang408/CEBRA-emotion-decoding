'''
Default (use caches, no recompute): python -m src.main
Force everything for all patients: FORCE=1 python -m src.main
Only redo grid search (keep shifts/scalers/models if present): FORCE_GRID=1 python -m src.main
Change shift sweep window: SHIFT_MIN=-8 SHIFT_MAX=8 python -m src.main
'''

import subprocess
import os
import argparse
import os
import subprocess

# List of patient IDs you want to run the pipeline on
#PATIENT_IDS = [1, 2, 9, 27, 28, 15, 22, 24, 29, 30, 31]
PATIENT_IDS = [1, 2, 9, 27, 28]  # full coverage

# List of scripts to run sequentially for each patient
PIPELINE_SCRIPTS = [
    #"gdec_model_finetune",
    #"gdec_evaluation",
    "gdec_attribution_weights"
]

def build_forwarded_args(args: argparse.Namespace):
    """Translate top-level flags into a list passed to child scripts."""
    forwarded = []
    # boolean flags
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
    # numeric flags
    forwarded += ["--shift-min", str(args.shift_min), "--shift-max", str(args.shift_max)]
    return forwarded

def run_pipeline_for_patient(pid: int, forwarded_args):
    env = os.environ.copy()
    env["PATIENT_ID"] = str(pid)

    print(f"\n====================== Running pipeline for Patient {pid} ======================")
    for script in PIPELINE_SCRIPTS:
        print(f"\n--- Running {script}.py for Patient {pid} ---")
        cmd = ["caffeinate", "-dimsu", "python", "-m", f"src.{script}", *forwarded_args]
        print("[CMD]", " ".join(cmd))  # helpful to verify flags are forwarded
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to run {script}.py for Patient {pid}: {e}")
            return  # stop for this patient if one step fails

def main():
    parser = argparse.ArgumentParser(description="Run xCEBRA training/eval pipeline")
    parser.add_argument("--patients", nargs="*", type=int, default=PATIENT_IDS,
                        help="Patient IDs to process (space-separated)")
    # Global “force” that enables all sub-forces
    parser.add_argument("--force", action="store_true",
                        help="Force re-run of ALL stages for child scripts")
    # Fine-grained flags (forwarded to child scripts)
    parser.add_argument("--force-grid", action="store_true", help="Force re-run grid search")
    parser.add_argument("--force-fold-models", action="store_true", help="Force re-train per-fold models")
    parser.add_argument("--force-final", action="store_true", help="Force re-train final patient model")
    parser.add_argument("--force-shift", action="store_true", help="Force recompute label shift")
    parser.add_argument("--force-scaler", action="store_true", help="Force recompute scalers")
    parser.add_argument("--shift-min", type=int, default=-5)
    parser.add_argument("--shift-max", type=int, default=5)
    args = parser.parse_args()

    # If --force, turn on all sub-forces
    if args.force:
        args.force_grid = True
        args.force_fold_models = True
        args.force_final = True
        args.force_shift = True
        args.force_scaler = True

    forwarded_args = build_forwarded_args(args)

    for pid in args.patients:
        run_pipeline_for_patient(pid, forwarded_args)

if __name__ == "__main__":
    main()