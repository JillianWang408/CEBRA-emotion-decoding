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
    9.0:      ("EC272", "272.0"),
    27.0:      ("EC301","301.0"),

    1:    ("EC238", "238"),
    2:    ("EC239", "239"),
    9:    ("EC272", "272"),
    27:    ("EC301", "301"),
    28:    ("EC304", "304"),

    15: ("EC280", "280"), #280 noisiest
    22: ("EC288", "288"),
    24: ("EC293", "293"),
    29: ("PR06", "PR06"),
    30: ("EC325", "325"),
    31: ("EC326", "326"),
}

ec_code, output_dir = PATIENT_CONFIG[PATIENT_ID]

#40*40
#DATA_DIR = PROJECT_ROOT / "data" / ec_code / "nrcRF_stim_resp_5_Nfold_pairs_msBW_1000_wASpec16_v16_DC_1   2   5   6   7   8   9  10  11  12__wASpec16_v16_DC_1   2   5   6   7   8   9  10  11  12_5"
# MODEL_DIR = PROJECT_ROOT / "output_covariance" / output_dir / "models"
# EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "output_covariance" / output_dir / "evaluation_outputs"
# ATTRIBUTION_OUTPUT_DIR = PROJECT_ROOT / "output_covariance" / output_dir / "attribution_outputs"

# #40*5*5
# DATA_DIR = PROJECT_ROOT / "data" / ec_code / "nrcRF_stim_resp_5_Nfold_pairs_msBW_1000_wASpec16_v16_DC5_1   2   5   6   7   8   9  10  11  12__wASpec16_v16_DC5_1   2   5   6   7   8   9  10  11  12_5"
# MODEL_DIR = PROJECT_ROOT / "output" / output_dir / "models"
# EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "output" / output_dir / "evaluation_outputs"
# ATTRIBUTION_OUTPUT_DIR = PROJECT_ROOT / "output" / output_dir / "attribution_outputs"


#logistic
DATA_DIR = PROJECT_ROOT / "data" / ec_code / "nrcRF_stim_resp_5_Nfold_pairs_msBW_1000_wASpec16_v16_DC5_1   2   5   6   7   8   9  10  11  12__wASpec16_v16_DC5_1   2   5   6   7   8   9  10  11  12_5"

# gdec:
MODEL_DIR = PROJECT_ROOT / "output_gdec" / output_dir / "models"
OUT_DIR = Path(MODEL_DIR)  / "gdec_gpmd"
EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "output_gdec" / output_dir / "evaluation_outputs"
ATTRIBUTION_OUTPUT_DIR = PROJECT_ROOT / "output_gdec" / output_dir / "attribution_outputs"

#glmnet:
# MODEL_DIR = PROJECT_ROOT / "output_glmnet" / output_dir / "models"
# OUT_DIR = Path(MODEL_DIR) / "glmnet"
# EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "output_glmnet" / output_dir / "evaluation_outputs"
# ATTRIBUTION_OUTPUT_DIR = PROJECT_ROOT / "output_glmnet" / output_dir / "attribution_outputs"

FOLD_MODELS_DIR = OUT_DIR / "fold_models"
FOLDS_DIR = OUT_DIR / "folds"

# File paths
NEURAL_PATH = DATA_DIR / "nrcRF_calc_Stim_StimNum_5_Nr_1_msBW_1000_movHeldOut_1.mat"
EMOTION_PATH = DATA_DIR / "nrcRF_calc_Resp_chan_1_movHeldOut_1.mat"

NEURAL_TENSOR_PATH = MODEL_DIR / "neural_tensor.pt"
EMOTION_TENSOR_PATH = MODEL_DIR / "emotion_tensor.pt"
FULL_NEURAL_PATH = MODEL_DIR / "full_neural_tensor.pt"
FULL_EMOTION_PATH = MODEL_DIR / "full_emotion_tensor.pt"

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
