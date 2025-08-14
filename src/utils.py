import torch
from cebra.models import init as init_model
from src.config import NEURAL_TENSOR_PATH


def load_fixed_cebra_model(model_path, name="offset10-model", num_units=256, num_output=20):
    # Infer input size
    num_neurons = torch.load(NEURAL_TENSOR_PATH).shape[1]

    # Initialize model
    model = init_model(
        name=name,
        num_neurons=num_neurons,
        num_units=num_units,
        num_output=num_output
    )

    # Load and fix state_dict
    raw_state_dict = torch.load(model_path, map_location="cpu")
    fixed_state_dict = {}

    for k, v in raw_state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        k = k.replace(".module.", ".")

        if k.startswith("net.2.0"):
            k = k.replace("net.2.0", "net.2.module.0")
        elif k.startswith("net.3.0"):
            k = k.replace("net.3.0", "net.3.module.0")
        elif k.startswith("net.4.0"):
            k = k.replace("net.4.0", "net.4.module.0")

        fixed_state_dict[k] = v

    model.load_state_dict(fixed_state_dict)
    model.eval()
    return model
