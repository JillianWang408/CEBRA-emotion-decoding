from pathlib import Path
import sys

# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

PATIENT_ID = 3.1

if PATIENT_ID == 1:
    DATA_ROOT = Path("/Users/wangzihan/Desktop/Projects/xCEBRA/data/EC272/Patient1")
    MODEL_DIR = PROJECT_ROOT /"output" / "patient_1" / "models"
    EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "output" / "patient_1" / "evaluation_outputs"
    ATTRIBUTION_OUTPUT_DIR = PROJECT_ROOT / "output" / "patient_1" / "attribution_outputs"
    DATA_DIR = PROJECT_ROOT / "data" / "EC272" / "Patient1"

if PATIENT_ID == 1.1:
    DATA_ROOT = Path("/Users/wangzihan/Desktop/Projects/xCEBRA/data/EC272/Patient1")
    MODEL_DIR = PROJECT_ROOT /"output" / "patient_1.1" / "models"
    EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "output" / "patient_1.1" / "evaluation_outputs"
    ATTRIBUTION_OUTPUT_DIR = PROJECT_ROOT / "output" / "patient_1.1" / "attribution_outputs"
    DATA_DIR = PROJECT_ROOT / "data" / "EC272" / "Patient1"

elif PATIENT_ID == 2:
    DATA_ROOT = Path("/Users/wangzihan/Desktop/Projects/xCEBRA/data/EC301/Patient2")
    MODEL_DIR = PROJECT_ROOT /"output" / "patient_2" / "models"
    EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "output" / "patient_2" / "evaluation_outputs"
    ATTRIBUTION_OUTPUT_DIR = PROJECT_ROOT / "output" / "patient_2" / "attribution_outputs"
    DATA_DIR = PROJECT_ROOT / "data" / "EC301" / "Patient1"

elif PATIENT_ID == 2.1:
    DATA_ROOT = Path("/Users/wangzihan/Desktop/Projects/xCEBRA/data/EC301/Patient2")
    MODEL_DIR = PROJECT_ROOT /"output" / "patient_2" / "models"
    EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "output" / "patient_2" / "evaluation_outputs"
    ATTRIBUTION_OUTPUT_DIR = PROJECT_ROOT / "output" / "patient_2" / "attribution_outputs"
    DATA_DIR = PROJECT_ROOT / "data" / "EC301" / "Patient1"

elif PATIENT_ID == 3:
    DATA_ROOT = Path("/Users/wangzihan/Desktop/Projects/xCEBRA/data/EC304/Patient3")
    MODEL_DIR = PROJECT_ROOT /"output" / "patient_3" / "models"
    DATA_DIR = PROJECT_ROOT / "data" / "EC304" / "Patient3"
    EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "output" / "patient_3" / "evaluation_outputs"
    ATTRIBUTION_OUTPUT_DIR = PROJECT_ROOT / "output" / "patient_3" / "attribution_outputs"
    
elif PATIENT_ID == 3.1:
    DATA_ROOT = Path("/Users/wangzihan/Desktop/Projects/xCEBRA/data/EC304/Patient3")
    MODEL_DIR = PROJECT_ROOT /"output" / "patient_3.1" / "models"
    DATA_DIR = PROJECT_ROOT / "data" / "EC304" / "Patient3"
    EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "output" / "patient_3.1" / "evaluation_outputs"
    ATTRIBUTION_OUTPUT_DIR = PROJECT_ROOT / "output" / "patient_3.1" / "attribution_outputs"
    
elif PATIENT_ID == 4.1:
    DATA_ROOT = Path("/Users/wangzihan/Desktop/Projects/xCEBRA/data/EC239/Patient4")
    MODEL_DIR = PROJECT_ROOT /"output" / "patient_4.1" / "models"
    DATA_DIR = PROJECT_ROOT / "data" / "EC239" / "Patient4"
    EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "output" / "patient_4.1" / "evaluation_outputs"
    ATTRIBUTION_OUTPUT_DIR = PROJECT_ROOT / "output" / "patient_4.1" / "attribution_outputs"
        
elif PATIENT_ID == 5.1:
    DATA_ROOT = Path("/Users/wangzihan/Desktop/Projects/xCEBRA/data/EC238/Patient5")
    MODEL_DIR = PROJECT_ROOT /"output" / "patient_5.1" / "models"
    DATA_DIR = PROJECT_ROOT / "data" / "EC238" / "Patient5"
    EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "output" / "patient_5.1" / "evaluation_outputs"
    ATTRIBUTION_OUTPUT_DIR = PROJECT_ROOT / "output" / "patient_5.1" / "attribution_outputs"


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
# #hybrid
# BEHAVIOR_INDICES = (0, 10)
# TIME_INDICES = (10, 20)
# N_LATENTS = 20

#supervised
BEHAVIOR_INDICES = (0, 10)
N_LATENTS = 10

# Imaage Output path
EVALUATION_EMBEDDING_PLOT = EVALUATION_OUTPUT_DIR / "embedding_summary.png"
EVALUATION_CONFUSION_PLOT = EVALUATION_OUTPUT_DIR / "confusion_matrix.png"

# === Emotion label mapping ===
EMOTION_MAP = {
    0: "No emotion", 1: "Amusement", 2: "Embarrassment", 3: "Anger", 4: "Confused",
    5: "Awe", 6: "Disgust", 7: "Fear", 8: "Affection", 9: "Sadness"
}

ELECTRODE_NAMES = [
        'LOFC7', 'LOFC8', 'LOFC9', 'LOFC10', 'LOFC1', 'LOFC2', 'LOFC3', 'LOFC4', 
        'ROFC1', 'ROFC2', 'ROFC3', 'ROFC4', 'ROFC7', 'ROFC8', 'ROFC9', 'ROFC10', 
        'LAD1', 'LAD2', 'LAD3', 'LAD4'
        'LINS1', 'LINS2', 'LINS3', 'LINS4',
        'LC1', 'LC2', 'LC3', 'LC4',
        'RC1', 'RC2', 'RC3', 'RC4',
        'RINS1', 'RINS2', 'RINS3', 'RINS4',
        'RAD1', 'RAD2', 'RAD3', 'RAD4']

N_ELECTRODES = 40

