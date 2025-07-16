# src/attribution.py
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from cebra.models.jacobian import compute_jacobian
from cebra.models.jacobian_regularizer import compute_inverse_jacobian

# === CLI ===
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to saved model weights (.pt)")
parser.add_argument("--data_path", type=str, default="./data/neural_data.mat")
args = parser.parse_args()

# === Load Data ===
import mat73
neural_array = mat73.loadmat(args.data_path)['stim'].T
neural_tensor = torch.tensor(neural_array, dtype=torch.float32)

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

# === Compute Jacobians ===
with torch.no_grad():
    jacobian = compute_jacobian(model, neural_tensor)
    inv_jacobian = compute_inverse_jacobian(model, neural_tensor)

# === Visualize ===
# Absolute mean Jacobian
jf = jacobian
jf_abs = torch.abs(jf)
jf_mean = jf_abs.mean(dim=0).cpu().numpy()  # shape: [latent_dim, features]

plt.figure(figsize=(12, 6))
plt.imshow(jf_mean, aspect='auto', cmap='viridis')
plt.colorbar(label='Attribution Strength')
plt.xlabel("Input Features")
plt.ylabel("Latent Dimensions")
plt.title("Jacobian Attribution Map")
plt.tight_layout()
plt.show()
