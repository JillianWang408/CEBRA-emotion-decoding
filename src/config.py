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

#-----------XCEBRA-----------
# cov
# DATA_DIR = PROJECT_ROOT / "data" / ec_code / "nrcRF_stim_resp_5_Nfold_pairs_msBW_1000_wASpec16_v16_DC_1   2   5   6   7   8   9  10  11  12__wASpec16_v16_DC_1   2   5   6   7   8   9  10  11  12_5"
# MODEL_DIR = PROJECT_ROOT / "output_xCEBRA_cov" / output_dir / "models"
# EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "output_xCEBRA_cov" / output_dir / "evaluation_outputs"
# ATTRIBUTION_OUTPUT_DIR = PROJECT_ROOT / "output_xCEBRA_cov" / output_dir / "attribution_outputs"

# lags
DATA_DIR = PROJECT_ROOT / "data" / ec_code / "nrcRF_stim_resp_5_Nfold_pairs_msBW_1000_wASpec16_v16_DC5_1   2   5   6   7   8   9  10  11  12__wASpec16_v16_DC5_1   2   5   6   7   8   9  10  11  12_5"
MODEL_DIR = PROJECT_ROOT / "output_xCEBRA_lags" / output_dir / "models"
EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "output_xCEBRA_lags" / output_dir / "evaluation_outputs"
ATTRIBUTION_OUTPUT_DIR = PROJECT_ROOT / "output_xCEBRA_lags" / output_dir / "attribution_outputs"


#----------logistic------------
#lags
#DATA_DIR = PROJECT_ROOT / "data" / ec_code / "nrcRF_stim_resp_5_Nfold_pairs_msBW_1000_wASpec16_v16_DC5_1   2   5   6   7   8   9  10  11  12__wASpec16_v16_DC5_1   2   5   6   7   8   9  10  11  12_5"

#cov
#DATA_DIR = PROJECT_ROOT / "data" / ec_code / "nrcRF_stim_resp_5_Nfold_pairs_msBW_1000_wASpec16_v16_DC_1   2   5   6   7   8   9  10  11  12__wASpec16_v16_DC_1   2   5   6   7   8   9  10  11  12_5"

# gdec:
# MODEL_DIR = PROJECT_ROOT / "output_gdec" / output_dir / "models"
# OUT_DIR = Path(MODEL_DIR)  / "gdec_gpmd"
# EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "output_gdec" / output_dir / "evaluation_outputs"
# ATTRIBUTION_OUTPUT_DIR = PROJECT_ROOT / "output_gdec" / output_dir / "attribution_outputs"

# glmnet:
# MODEL_DIR = PROJECT_ROOT / "output_glmnet" / output_dir / "models"
# OUT_DIR = Path(MODEL_DIR)  / "glmnet"
# EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "output_glmnet" / output_dir / "evaluation_outputs"
# ATTRIBUTION_OUTPUT_DIR = PROJECT_ROOT / "output_glmnet" / output_dir / "attribution_outputs"

# FOLD_MODELS_DIR = OUT_DIR / "fold_models" 
# FOLDS_DIR = OUT_DIR / "folds"

# File paths (9emotion: calc=train, pred=test, same as single_emotion)
NEURAL_PATH = DATA_DIR / "nrcRF_calc_Stim_StimNum_5_Nr_1_msBW_1000_movHeldOut_1.mat"
EMOTION_PATH = DATA_DIR / "nrcRF_calc_Resp_chan_1_movHeldOut_1.mat"
NEURAL_PRED_PATH = DATA_DIR / "nrcRF_pred_Stim_StimNum_5_Nr_1_msBW_1000_movHeldOut_1.mat"
EMOTION_PRED_PATH = DATA_DIR / "nrcRF_pred_Resp_chan_1_movHeldOut_1.mat"

# Single-emotion trial-level pipeline: uses DC8 folder (separate from main DATA_DIR)
SINGLE_EMOTION_DATA_DIR = PROJECT_ROOT / "data" / ec_code / "nrcRF_stim_resp_5_Nfold_pairs_msBW_1000_wASpec16_v16_DC8_1   2   5   6   7   8   9  10  11  12__wASpec16_v16_DC8_1   2   5   6   7   8   9  10  11  12_5"
SINGLE_EMOTION_CALC_NEURAL = SINGLE_EMOTION_DATA_DIR / "nrcRF_calc_Stim_StimNum_5_Nr_1_msBW_1000_movHeldOut_1.mat"
SINGLE_EMOTION_CALC_RESP = SINGLE_EMOTION_DATA_DIR / "nrcRF_calc_Resp_chan_1_movHeldOut_1.mat"
SINGLE_EMOTION_PRED_NEURAL = SINGLE_EMOTION_DATA_DIR / "nrcRF_pred_Stim_StimNum_5_Nr_1_msBW_1000_movHeldOut_1.mat"
SINGLE_EMOTION_PRED_RESP = SINGLE_EMOTION_DATA_DIR / "nrcRF_pred_Resp_chan_1_movHeldOut_1.mat"
SINGLE_EMOTION_OUTPUT_DIR = PROJECT_ROOT / "output_single_emotion" / output_dir

# Valence/Arousal (DC6, DC7, DC9) - EC238 and EC239 do NOT have this data
_BASE_DC = "nrcRF_stim_resp_5_Nfold_pairs_msBW_1000_wASpec16_v16_DC{dc}_1   2   5   6   7   8   9  10  11  12__wASpec16_v16_DC{dc}_1   2   5   6   7   8   9  10  11  12_5"
VALENCE_AROUSAL_DATA_DIR_DC6 = PROJECT_ROOT / "data" / ec_code / _BASE_DC.format(dc="6")
VALENCE_AROUSAL_DATA_DIR_DC7 = PROJECT_ROOT / "data" / ec_code / _BASE_DC.format(dc="7")
VALENCE_AROUSAL_DATA_DIR_DC9 = PROJECT_ROOT / "data" / ec_code / _BASE_DC.format(dc="9")
# Valence/Arousal: calc = train, pred = test (same structure as single_emotion/9emotion)
VALENCE_CALC_NEURAL_DC6 = VALENCE_AROUSAL_DATA_DIR_DC6 / "nrcRF_calc_Stim_StimNum_5_Nr_1_msBW_1000_movHeldOut_1.mat"
VALENCE_CALC_RESP_DC6 = VALENCE_AROUSAL_DATA_DIR_DC6 / "nrcRF_calc_Resp_chan_1_movHeldOut_1.mat"
VALENCE_PRED_NEURAL_DC6 = VALENCE_AROUSAL_DATA_DIR_DC6 / "nrcRF_pred_Stim_StimNum_5_Nr_1_msBW_1000_movHeldOut_1.mat"
VALENCE_PRED_RESP_DC6 = VALENCE_AROUSAL_DATA_DIR_DC6 / "nrcRF_pred_Resp_chan_1_movHeldOut_1.mat"
VALENCE_CALC_NEURAL_DC7 = VALENCE_AROUSAL_DATA_DIR_DC7 / "nrcRF_calc_Stim_StimNum_5_Nr_1_msBW_1000_movHeldOut_1.mat"
VALENCE_CALC_RESP_DC7 = VALENCE_AROUSAL_DATA_DIR_DC7 / "nrcRF_calc_Resp_chan_1_movHeldOut_1.mat"
VALENCE_PRED_NEURAL_DC7 = VALENCE_AROUSAL_DATA_DIR_DC7 / "nrcRF_pred_Stim_StimNum_5_Nr_1_msBW_1000_movHeldOut_1.mat"
VALENCE_PRED_RESP_DC7 = VALENCE_AROUSAL_DATA_DIR_DC7 / "nrcRF_pred_Resp_chan_1_movHeldOut_1.mat"
VALENCE_CALC_NEURAL_DC9 = VALENCE_AROUSAL_DATA_DIR_DC9 / "nrcRF_calc_Stim_StimNum_5_Nr_1_msBW_1000_movHeldOut_1.mat"
VALENCE_CALC_RESP_DC9 = VALENCE_AROUSAL_DATA_DIR_DC9 / "nrcRF_calc_Resp_chan_1_movHeldOut_1.mat"
VALENCE_PRED_NEURAL_DC9 = VALENCE_AROUSAL_DATA_DIR_DC9 / "nrcRF_pred_Stim_StimNum_5_Nr_1_msBW_1000_movHeldOut_1.mat"
VALENCE_PRED_RESP_DC9 = VALENCE_AROUSAL_DATA_DIR_DC9 / "nrcRF_pred_Resp_chan_1_movHeldOut_1.mat"

NEURAL_TENSOR_PATH = MODEL_DIR / "neural_tensor.pt"
EMOTION_TENSOR_PATH = MODEL_DIR / "emotion_tensor.pt"
FULL_NEURAL_PATH = MODEL_DIR / "full_neural_tensor.pt"
FULL_EMOTION_PATH = MODEL_DIR / "full_emotion_tensor.pt"

MODEL_WEIGHTS_PATH = MODEL_DIR / "xcebra_weights.pt"
EMBEDDING_PATH = MODEL_DIR / "embedding.pt"


# Embedding dimensions
BEHAVIOR_INDICES = None
N_LATENTS = 10

# Output plots
EVALUATION_EMBEDDING_PLOT = EVALUATION_OUTPUT_DIR / "embedding_summary.png"
EVALUATION_CONFUSION_PLOT = EVALUATION_OUTPUT_DIR / "confusion_matrix.png"

# Emotion map
EMOTION_MAP = {
    0: "No emotion", 1: "Amusement", 2: "Embarrassment", 3: "Anger", 4: "Confused",
    5: "Awe", 6: "Disgust", 7: "Fear", 8: "Affection", 9: "Sadness"
}

# Valence/Arousal: EC238 and EC239 do NOT have this data
PATIENTS_WITH_VALENCE_AROUSAL = [9, 27, 28, 15, 22, 24, 29, 30, 31]  # exclude 1, 2

# DC6 Arousal: 0-6 scale (very calm to very activated, 3=neutral)
AROUSAL_MAP = {0: "Very calm", 1: "Calm", 2: "Slightly calm", 3: "Neutral", 4: "Slightly activated", 5: "Activated", 6: "Very activated"}

# DC7 Valence: 0-6 scale (very unpleasant to very pleasant, 3=neutral)
VALENCE_MAP = {0: "Very unpleasant", 1: "Unpleasant", 2: "Slightly unpleasant", 3: "Neutral", 4: "Slightly pleasant", 5: "Pleasant", 6: "Very pleasant"}

# DC9 Valence/Arousal Categories: 0-4 (quadrants)
CATEGORY_MAP = {
    0: "Neutral/Neutral",
    1: "Unpleasant/Activated",
    2: "Pleasant/Activated",
    3: "Pleasant/Calm",
    4: "Unpleasant/Calm",
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

NODE_MAP = {
    # Orbitofrontal Cortex — Left (LOFC)
    "LOFC_Medial":  ["LOFC1", "LOFC2", "LOFC3", "LOFC4"],
    "LOFC_Lateral": ["LOFC7", "LOFC8", "LOFC9", "LOFC10"],

    # Orbitofrontal Cortex — Right (ROFC)
    "ROFC_Medial":  ["ROFC1", "ROFC2", "ROFC3", "ROFC4"],
    "ROFC_Lateral": ["ROFC7", "ROFC8", "ROFC9", "ROFC10"],

    # Anterior Dorsal Cingulate
    "LAD": ["LAD1", "LAD2", "LAD3", "LAD4"],
    "RAD": ["RAD1", "RAD2", "RAD3", "RAD4"],

    # Insula
    "LINS": ["LINS1", "LINS2", "LINS3", "LINS4"],
    "RINS": ["RINS1", "RINS2", "RINS3", "RINS4"],

    # Cingulate
    "LC": ["LC1", "LC2", "LC3", "LC4"],
    "RC": ["RC1", "RC2", "RC3", "RC4"],
}

N_ELECTRODES = 40
