"""Load config paths for CEBRA based on patient_id and target."""

import os
from pathlib import Path


def load_cebra_config(patient_id: int, target: str):
    """
    Load config with PATIENT_ID set. Returns paths for neural, labels, output.
    target: "9emotion" | "arousal" | "valence" | "categories"
    """
    os.environ["PATIENT_ID"] = str(patient_id)
    import importlib.util
    project_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "config", project_root / "src" / "config.py"
    )
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    if target == "9emotion":
        neural_train = config.NEURAL_PATH
        label_train = config.EMOTION_PATH
        neural_test = config.NEURAL_PRED_PATH
        label_test = config.EMOTION_PRED_PATH
        label_map = config.EMOTION_MAP
        dc_name = "DC5"
    elif target == "arousal":
        neural_train = config.VALENCE_CALC_NEURAL_DC6
        label_train = config.VALENCE_CALC_RESP_DC6
        neural_test = config.VALENCE_PRED_NEURAL_DC6
        label_test = config.VALENCE_PRED_RESP_DC6
        label_map = config.AROUSAL_MAP
        dc_name = "DC6"
    elif target == "valence":
        neural_train = config.VALENCE_CALC_NEURAL_DC7
        label_train = config.VALENCE_CALC_RESP_DC7
        neural_test = config.VALENCE_PRED_NEURAL_DC7
        label_test = config.VALENCE_PRED_RESP_DC7
        label_map = config.VALENCE_MAP
        dc_name = "DC7"
    elif target == "categories":
        neural_train = config.VALENCE_CALC_NEURAL_DC9
        label_train = config.VALENCE_CALC_RESP_DC9
        neural_test = config.VALENCE_PRED_NEURAL_DC9
        label_test = config.VALENCE_PRED_RESP_DC9
        label_map = config.CATEGORY_MAP
        dc_name = "DC9"
    else:
        raise ValueError(f"Unknown target: {target}")

    output_base = project_root / "output_CEBRA" / config.output_dir / target
    model_dir = output_base / "models"

    return {
        "neural_train": neural_train,
        "label_train": label_train,
        "neural_test": neural_test,
        "label_test": label_test,
        "label_map": label_map,
        "output_base": output_base,
        "model_dir": model_dir,
        "patient_code": config.output_dir,
        "dc_name": dc_name,
    }
