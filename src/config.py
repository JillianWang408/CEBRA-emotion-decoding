from pathlib import Path
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# === Get PATIENT_ID from environment variable ===
PATIENT_ID = float(os.environ.get("PATIENT_ID", -1))
if PATIENT_ID == -1:
    raise ValueError("PATIENT_ID environment variable not set.")

# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# Patient configuration dictionary
PATIENT_CONFIG = {
    272.0:      ("EC272", "272", "272.0"),
    301.0:      ("EC301", "301", "301.0"),

    272:    ("EC272", "272", "272"),
    301:    ("EC301", "301", "301"),
    304:    ("EC304", "304", "304"),
    239:    ("EC239", "239", "239"),
    238:    ("EC238", "238", "238"),
}

ec_code, patient_dir, output_dir = PATIENT_CONFIG[PATIENT_ID]

DATA_ROOT = Path(f"/Users/wangzihan/Desktop/Projects/xCEBRA/data/{ec_code}/{patient_dir}")
DATA_DIR = PROJECT_ROOT / "data" / ec_code / patient_dir
MODEL_DIR = PROJECT_ROOT / "output" / output_dir / "models"
EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "output" / output_dir / "evaluation_outputs"
ATTRIBUTION_OUTPUT_DIR = PROJECT_ROOT / "output" / output_dir / "attribution_outputs"

# File paths
NEURAL_PATH = DATA_DIR / "nrcRF_calc_Stim_StimNum_5_Nr_1_msBW_1000_movHeldOut_1.mat"
EMOTION_PATH = DATA_DIR / "nrcRF_calc_Resp_chan_1_movHeldOut_1.mat"
NEURAL_TENSOR_PATH = MODEL_DIR / "neural_tensor.pt"
EMOTION_TENSOR_PATH = MODEL_DIR / "emotion_tensor.pt"
MODEL_WEIGHTS_PATH = MODEL_DIR / "xcebra_weights.pt"
EMBEDDING_PATH = MODEL_DIR / "embedding.pt"

# Embedding dimensions
BEHAVIOR_INDICES = (0, 10)
N_LATENTS = 10

# Output plots
EVALUATION_EMBEDDING_PLOT = EVALUATION_OUTPUT_DIR / "embedding_summary.png"
EVALUATION_CONFUSION_PLOT = EVALUATION_OUTPUT_DIR / "confusion_matrix.png"

# Emotion map
EMOTION_MAP = {
    0: "No emotion", 1: "Amusement", 2: "Embarrassment", 3: "Anger", 4: "Confused",
    5: "Awe", 6: "Disgust", 7: "Fear", 8: "Affection", 9: "Sadness"
}

ELECTRODE_NAMES = [
    'LOFC7', 'LOFC8', 'LOFC9', 'LOFC10', 'LOFC1', 'LOFC2', 'LOFC3', 'LOFC4', 
    'ROFC1', 'ROFC2', 'ROFC3', 'ROFC4', 'ROFC7', 'ROFC8', 'ROFC9', 'ROFC10', 
    'LAD1', 'LAD2', 'LAD3', 'LAD4',
    'LINS1', 'LINS2', 'LINS3', 'LINS4',
    'LC1', 'LC2', 'LC3', 'LC4',
    'RC1', 'RC2', 'RC3', 'RC4',
    'RINS1', 'RINS2', 'RINS3', 'RINS4',
    'RAD1', 'RAD2', 'RAD3', 'RAD4'
]

N_ELECTRODES = 40
