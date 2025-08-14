# RUN THESE TWO LINES TOGETHER: 
# caffeinate -dimsu python -m src.train

import torch
import cebra
import mat73
import scipy.io
from cebra.data import DatasetxCEBRA, ContrastiveMultiObjectiveLoader
from cebra.solver import MultiObjectiveConfig
from cebra.solver.schedulers import LinearRampUp
from cebra.models import init as init_model
from cebra.models.jacobian_regularizer import JacobianReg

from src.config import (
    MODEL_DIR, NEURAL_PATH, EMOTION_PATH, MODEL_WEIGHTS_PATH, NEURAL_TENSOR_PATH, 
    EMOTION_TENSOR_PATH, BEHAVIOR_INDICES, TIME_INDICES, N_LATENTS
)

# === Load Data ===
neural_array = mat73.loadmat(NEURAL_PATH)['stim'].T
emotion_array = scipy.io.loadmat(EMOTION_PATH)['resp'].flatten()

neural_tensor = torch.tensor(neural_array, dtype=torch.float32)
emotion_tensor = torch.tensor(emotion_array, dtype=torch.float32).unsqueeze(1)

MODEL_DIR.mkdir(exist_ok=True, parents=True)
NEURAL_TENSOR_PATH.parent.mkdir(exist_ok=True, parents=True)
EMOTION_TENSOR_PATH.parent.mkdir(exist_ok=True, parents=True)
torch.save(neural_tensor, NEURAL_TENSOR_PATH)
torch.save(emotion_tensor, EMOTION_TENSOR_PATH)

print("Tensors saved")
print("neural_tensor shape:", neural_tensor.shape)

# === Dataset Setup ===
datasets = DatasetxCEBRA(neural=neural_tensor, position=emotion_tensor)

# === Loader ===
batch_size = 512
num_steps = 1000
n_latents = 20
behavior_indices = BEHAVIOR_INDICES # The embedding[:, 0:9] portion will be trained using emotion contrastive loss (e.g., close if same emotion).
time_indices = TIME_INDICES #The embedding[:, 9:18] portion will be trained using time contrastive loss (e.g., close if nearby in time).
loader = ContrastiveMultiObjectiveLoader(dataset=datasets, batch_size=batch_size, num_steps=num_steps)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Config ===
config = MultiObjectiveConfig(loader)
config.set_slice(*BEHAVIOR_INDICES)
config.set_loss("FixedCosineInfoNCE", temperature=1.0)
config.set_distribution("time_delta", time_delta=1, label_name="position")
config.push()

config.set_slice(*TIME_INDICES)
config.set_loss("FixedCosineInfoNCE", temperature=1.0)
config.set_distribution("time", time_offset=10)
config.push()
config.finalize()

criterion = config.criterion
feature_ranges = config.feature_ranges

# === Model ===
# Loader, prepare patches for training
neural_model = init_model(name="offset10-model", num_neurons=datasets.neural.shape[1], 
                          num_units=256, num_output=N_LATENTS).to(device)

# Assuming all datasets have the same configuration, configure using the first one
datasets.configure_for(neural_model)

# Define Optimizer
opt = torch.optim.Adam(list(neural_model.parameters()) + list(criterion.parameters()), lr=3e-4, weight_decay=0)

regularizer = cebra.models.jacobian_regularizer.JacobianReg()

#Create Solver (for actual training)
solver = cebra.solver.init(name="multiobjective-solver", model=neural_model, feature_ranges=feature_ranges,
                          regularizer=regularizer, renormalize=True, use_sam=False, criterion=criterion,
                          optimizer=opt, tqdm_on=True).to(device)

# === Regularizer scheduler ===
weight_scheduler = LinearRampUp(n_splits=2, step_to_switch_on_reg=num_steps // 4, step_to_switch_off_reg=num_steps // 2,
                         start_weight=0.0, end_weight=0.1)

# === Train ===
solver.to(device)
solver.fit(loader=loader, valid_loader=None, scheduler_regularizer=weight_scheduler)

# Save trained model
output_path = MODEL_WEIGHTS_PATH
output_path.parent.mkdir(exist_ok=True, parents=True)
torch.save(solver.model.state_dict(), output_path)
