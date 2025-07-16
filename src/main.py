from evaluate import evaluate_embedding
from plot import plot_embedding, plot_confusion_matrix
from attribution import compute_and_plot_attribution
from config import NEURAL_PATH
import torch
import mat73

if __name__ == "__main__":
    # Load evaluation results and plot
    eval_result = evaluate_embedding()
    plot_embedding(eval_result)
    plot_confusion_matrix(eval_result)

    # Optional: load model and neural_tensor for attribution
    model = torch.load("models/xcebra_weights.pt")  # adjust if necessary
    neural_tensor = mat73.loadmat(NEURAL_PATH)['stim'].T
    neural_tensor = torch.tensor(neural_tensor, dtype=torch.float32)
    compute_and_plot_attribution(model, neural_tensor)
