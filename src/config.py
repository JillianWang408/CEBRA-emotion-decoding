from pathlib import Path
import sys

# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DATA_ROOT = Path("/Users/wangzihan/Desktop/Projects/xCEBRA/data/EC272/Patient1")
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "EC272" / "Patient1"

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
N_ELECTRODES = 40

# === Emotion label mapping ===
EMOTION_MAP = {
    0: "No emotion", 1: "Amusement", 2: "Embarrassment", 3: "Anger", 4: "Confused",
    5: "Awe", 6: "Disgust", 7: "Fear", 8: "Affection", 9: "Sadness"
}
# === Electrodes for the patient ===
ELECTRODE_NAMES = [
    'LOFC7', 'LOFC8', 'LOFC9', 'LOFC10', 'LOFC2', 'LOFC3', 'LOFC4', 'LOFC5',
    'ROFC2', 'ROFC3', 'ROFC4', 'ROFC5', 'ROFC7', 'ROFC8', 'ROFC9', 'ROFC10',
    'LAMY1', 'LAMY2', 'LAMY3', 'LAMY4',
    'LAINS1', 'LAINS2', 'LAINS3', 'LAINS4',
    'LCING2', 'LCING3', 'LCING4', 'LCING5',
    'RCING2', 'RCING3', 'RCING4', 'RCING5',
    'RAINS1', 'RAINS2', 'RAINS3', 'RAINS4',
    'RAMY1', 'RAMY2', 'RAMY3', 'RAMY4'
]


