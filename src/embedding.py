# RUN WITH: python -m src.embedding

import torch
from pathlib import Path
import json
from cebra.models import init as init_model
from cebra.data import DatasetxCEBRA
from src.config import (NEURAL_TENSOR_PATH, EMOTION_TENSOR_PATH)


# === Load data ===
neural_tensor = torch.load(NEURAL_TENSOR_PATH)
emotion_tensor = torch.load(EMOTION_TENSOR_PATH)

# === Create embedding dataset ===
datasets = DatasetxCEBRA(neural=neural_tensor, continuous=emotion_tensor)
model = init_model(name="offset10-model", num_neurons=neural_tensor.shape[1], num_units=256, num_output=20)

state_dict = torch.load("models/xcebra_weights.pt")

model.eval()

datasets.configure_for(model)
data_input = datasets[torch.arange(len(datasets))]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.split_outputs = False

embedding = model(data_input.to(device)).detach().cpu()
torch.save(embedding, "models/embedding.pt")
print("Embedding shape:", embedding.shape)

