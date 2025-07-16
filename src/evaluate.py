# src/evaluate.py
import torch
import argparse
import numpy as np
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# === CLI ===
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to saved model weights (.pt)")
parser.add_argument("--data_path", type=str, default="./data/neural_data.mat", help="Path to neural data (.mat)")
parser.add_argument("--label_path", type=str, default="./data/emotion_labels.mat", help="Path to label data (.mat)")
args = parser.parse_args()

# === Load Data ===
import mat73, scipy.io
neural_array = mat73.loadmat(args.data_path)['stim'].T
emotion_array = scipy.io.loadmat(args.label_path)['resp'].flatten()

neural_tensor = torch.tensor(neural_array, dtype=torch.float32)
emotion_tensor = torch.tensor(emotion_array, dtype=torch.int64)  # assuming labels are integers

# === Load Model ===
from cebra.models import init as init_model
model = init_model(name="offset10-model", num_neurons=neural_tensor.shape[1], num_units=256, num_output=20)
model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
model.eval()

# === Format Input ===
if neural_tensor.dim() == 2:
    neural_tensor = neural_tensor.unsqueeze(1).repeat(1, 2, 1).permute(0, 2, 1)  # shape: [N, input, time]
    if neural_tensor.shape[2] < 3:
        pad = neural_tensor[:, :, :1].expand(-1, -1, 3 - neural_tensor.shape[2])
        neural_tensor = torch.cat([neural_tensor, pad], dim=2)

# === Get Embeddings ===
with torch.no_grad():
    embeddings = model(neural_tensor).cpu().numpy()

# === KNN Evaluation ===
y_true = emotion_tensor.numpy()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(KNeighborsClassifier(n_neighbors=5), embeddings, y_true, cv=cv)

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Emotion Decoding Confusion Matrix")
plt.tight_layout()
plt.show()