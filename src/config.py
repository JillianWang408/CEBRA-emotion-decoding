from pathlib import Path
import sys

# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

PATIENT_ID = 1

if PATIENT_ID == 1:
    DATA_ROOT = Path("/Users/wangzihan/Desktop/Projects/xCEBRA/data/EC272/Patient1")
    MODEL_DIR = PROJECT_ROOT /"output" / "patient_1" / "models"
    EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "output" / "patient_1" / "evaluation_outputs"
    ATTRIBUTION_OUTPUT_DIR = PROJECT_ROOT / "output" / "patient_1" / "attribution_outputs"
    DATA_DIR = PROJECT_ROOT / "data" / "EC272" / "Patient1"
    ELECTRODE_NAMES = [
    'LOFC7', 'LOFC8', 'LOFC9', 'LOFC10', 'LOFC2', 'LOFC3', 'LOFC4', 'LOFC5',
    'ROFC2', 'ROFC3', 'ROFC4', 'ROFC5', 'ROFC7', 'ROFC8', 'ROFC9', 'ROFC10',
    'LAMY1', 'LAMY2', 'LAMY3', 'LAMY4', 'LAINS1', 'LAINS2', 'LAINS3', 'LAINS4',
    'LCING2', 'LCING3', 'LCING4', 'LCING5','RCING2', 'RCING3', 'RCING4', 'RCING5',
    'RAINS1', 'RAINS2', 'RAINS3', 'RAINS4','RAMY1', 'RAMY2', 'RAMY3', 'RAMY4'
]   
if PATIENT_ID == 2:
    DATA_ROOT = Path("/Users/wangzihan/Desktop/Projects/xCEBRA/data/EC301/Patient2")
    MODEL_DIR = PROJECT_ROOT /"output" / "patient_2" / "models"
    EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "output" / "patient_2" / "evaluation_outputs"
    ATTRIBUTION_OUTPUT_DIR = PROJECT_ROOT / "output" / "patient_2" / "attribution_outputs"
    DATA_DIR = PROJECT_ROOT / "data" / "EC301" / "Patient1"
    ELECTRODE_NAMES = ['LOFC7', 'LOFC8', 'LOFC9', 'LOFC10', 'LOFC1', 'LOFC2', 'LOFC3', 'LOFC4',
                   'ROFC1', 'ROFC2', 'ROFC3', 'ROFC4', 'ROFC7', 'ROFC8', 'ROFC9', 'ROFC10',
                   'LPI1', 'LPI2', 'LPI3', 'LPI4', 'LC1', 'LC2', 'LC3', 'LC4', 
                   'RC1', 'RC2', 'RC3', 'RC4', 'RI1', 'RI2', 'RI3', 'RI4', 'RA1', 'RA2', 'RA3', 'RA4']



# Raw input .mat files
NEURAL_PATH = DATA_DIR / "nrcRF_calc_Stim_StimNum_5_Nr_1_msBW_1000_movHeldOut_1.mat"
EMOTION_PATH = DATA_DIR / "nrcRF_calc_Resp_chan_1_movHeldOut_1.mat"

# Saved PyTorch tensor files
NEURAL_TENSOR_PATH = MODEL_DIR / "neural_tensor.pt"
EMOTION_TENSOR_PATH = MODEL_DIR / "emotion_tensor.pt"

# Trained model
MODEL_WEIGHTS_PATH = MODEL_DIR / "xcebra_weights.pt"

# Inferred embedding output
EMBEDDING_PATH = MODEL_DIR / "embedding.pt"

# === Embedding dimensions ===
BEHAVIOR_INDICES = (0, 10)
TIME_INDICES = (10, 20)
N_LATENTS = 20

# Imaage Output path
EVALUATION_EMBEDDING_PLOT = EVALUATION_OUTPUT_DIR / "embedding_summary.png"
EVALUATION_CONFUSION_PLOT = EVALUATION_OUTPUT_DIR / "confusion_matrix.png"

# === Emotion label mapping ===
EMOTION_MAP = {
    0: "No emotion", 1: "Amusement", 2: "Embarrassment", 3: "Anger", 4: "Confused",
    5: "Awe", 6: "Disgust", 7: "Fear", 8: "Affection", 9: "Sadness"
}

N_ELECTRODES = len(ELECTRODE_NAMES)

