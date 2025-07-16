import torch
import mat73
import scipy.io
from pathlib import Path
from cebra.data import DatasetxCEBRA, ContrastiveMultiObjectiveLoader
from cebra.solver import MultiObjectiveConfig
from cebra.solver.schedulers import LinearRampUp
from cebra.models import init as init_model
from cebra.models.jacobian_regularizer import JacobianReg
import os

data_root = Path("./data")
neural_path = data_root / "neural_data.mat"
emotion_path = data_root / "emotion_labels.mat"

# === Load Data ===
neural_array = mat73.loadmat(neural_path)['stim'].T
emotion_array = scipy.io.loadmat(emotion_path)['resp'].flatten()

neural_tensor = torch.tensor(neural_array, dtype=torch.float32)
emotion_tensor = torch.tensor(emotion_array, dtype=torch.float32).unsqueeze(1)

# === Dataset Setup ===
dataset = DatasetxCEBRA(neural=neural_tensor, position=emotion_tensor)

# === Loader ===
loader = ContrastiveMultiObjectiveLoader(dataset=dataset, batch_size=512, num_steps=1000)

# === Config ===
config = MultiObjectiveConfig(loader)
config.set_slice(0, 10)
config.set_loss("FixedCosineInfoNCE", temperature=1.0)
config.set_distribution("time_delta", time_delta=1, label_name="position")
config.push()

config.set_slice(10, 20)
config.set_loss("FixedCosineInfoNCE", temperature=1.0)
config.set_distribution("time", time_offset=10)
config.push()
config.finalize()

criterion = config.criterion
feature_ranges = config.feature_ranges

# === Model ===
model = init_model(name="offset10-model", num_neurons=neural_tensor.shape[1], num_units=256, num_output=20)
regularizer = JacobianReg()

optimizer = torch.optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=3e-4)

from cebra.solver import init as init_solver
solver = init_solver(name="multiobjective-solver", model=model, feature_ranges=feature_ranges,
                     criterion=criterion, optimizer=optimizer, regularizer=regularizer,
                     renormalize=True, use_sam=False, tqdm_on=True)

# === Regularizer scheduler ===
scheduler = LinearRampUp(n_splits=2, step_to_switch_on_reg=250, step_to_switch_off_reg=500,
                         start_weight=0.0, end_weight=0.1)

# === Train ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
solver.to(device)
solver.fit(loader=loader, valid_loader=None, scheduler_regularizer=scheduler)

# === Save ===
output_path = Path("./models/xcebra_weights.pt")
output_path.parent.mkdir(exist_ok=True, parents=True)
torch.save(model.state_dict(), output_path)
print(f"Model saved to {output_path}")

