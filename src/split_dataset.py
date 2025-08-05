# RUN WITH: python -m src.split_dataset

import numpy as np
import torch
from pathlib import Path
from src.config import NEURAL_PATH, EMOTION_PATH
import mat73
import scipy.io


# === Load data
neural_array = mat73.loadmat(NEURAL_PATH)['stim'].T  # shape [T, 1000]
emotion_array = scipy.io.loadmat(EMOTION_PATH)['resp'].flatten()

# === Create train/test split indices
train_frac = 0.8
n = len(emotion_array)
split_idx = int(n * train_frac)

train_idx = np.arange(0, split_idx)
test_idx = np.arange(split_idx, n)

# === Save
split_dir = Path("splits")
split_dir.mkdir(exist_ok=True)

np.save(split_dir / "train_idx.npy", train_idx)
np.save(split_dir / "test_idx.npy", test_idx)

print(f"✅ Split complete: {len(train_idx)} train, {len(test_idx)} test")
