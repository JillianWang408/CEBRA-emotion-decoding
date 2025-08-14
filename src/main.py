import subprocess
import os

# List of patient IDs you want to run the pipeline on
# PATIENT_IDS = [1, 2, 9, 27, 28, 15, 22, 24, 29, 30, 31] 
PATIENT_IDS = [1, 2, 9, 27, 28] #full coverage

# List of scripts to run sequentially for each patient
PIPELINE_SCRIPTS = [
    #"gdec_training",
    "gdec_evaluation"
]

def run_pipeline_for_patient(pid):
    env = os.environ.copy()
    env["PATIENT_ID"] = str(pid)  # Set patient ID for config.py to use
    print(f"\n====================== Running pipeline for Patient {pid} ======================")

    for script in PIPELINE_SCRIPTS:
        print(f"\n--- Running {script}.py for Patient {pid} ---")
        # Run each script using `caffeinate` to keep system awake
        try:
            subprocess.run(["caffeinate", "-dimsu", "python", "-m", f"src.{script}"], check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to run {script}.py for Patient {pid}: {e}")
            return  # Skip remaining scripts for this patient if one fails

def main():
    for pid in PATIENT_IDS:
        run_pipeline_for_patient(pid)

if __name__ == "__main__":
    main()
