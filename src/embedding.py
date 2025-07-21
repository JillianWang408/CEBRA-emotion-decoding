# RUN WITH: python -m src.embedding

import torch
from pathlib import Path
import json
from cebra.models import init as init_model
from cebra.data import DatasetxCEBRA
from src.config import (NEURAL_TENSOR_PATH, EMOTION_TENSOR_PATH, MODEL_WEIGHTS_PATH)
from src.utils import load_fixed_cebra_model


# === Load data ===
neural_tensor = torch.load(NEURAL_TENSOR_PATH)
emotion_tensor = torch.load(EMOTION_TENSOR_PATH)

# === Prepare dataset and model ===
datasets = DatasetxCEBRA(neural=neural_tensor, continuous=emotion_tensor)
model = load_fixed_cebra_model(MODEL_WEIGHTS_PATH, num_output=20)

datasets.configure_for(model)
data_input = datasets[torch.arange(len(datasets))]

# === Move to device and compute embedding ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.split_outputs = False

embedding = model(data_input.to(device)).detach().cpu()
torch.save(embedding, "models/embedding.pt")
print("Embedding shape:", embedding.shape)

