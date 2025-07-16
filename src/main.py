# RUN WITH: python -m src.main

import torch
import os
import sys
import mat73

from evaluate import evaluate_embedding
from plot import plot_embedding, plot_confusion_matrix
from attribution import compute_and_plot_attribution
from config import (
    EMBEDDING_PATH, MODEL_WEIGHTS_PATH, NEURAL_PATH,
    EMOTION_TENSOR_PATH, NEURAL_TENSOR_PATH
)

def main():
    # Evaluate and plot
    print("Evaluating embedding...")
    eval_result = evaluate_embedding()

    print("Plotting embedding and confusion matrix...")
    plot_embedding(eval_result)
    plot_confusion_matrix(eval_result)

    # Load model and neural data
    print("Loading model and neural data for attribution...")
    try:
        model = torch.load(MODEL_WEIGHTS_PATH, map_location="cpu")
    except Exception as e:
        print(f"Failed to load model from {MODEL_WEIGHTS_PATH}: {e}")
        sys.exit(1)

    try:
        neural_tensor = torch.load(NEURAL_TENSOR_PATH)
    except FileNotFoundError:
        print("Neural tensor file not found, attempting to load from .mat file...")
        try:
            raw_data = mat73.loadmat(NEURAL_PATH)
            neural_tensor = torch.tensor(raw_data['stim'].T, dtype=torch.float32)
        except Exception as e:
            print(f"Failed to load neural data from {NEURAL_PATH}: {e}")
            sys.exit(1)

    print("Running attribution analysis...")
    compute_and_plot_attribution(model, neural_tensor)

if __name__ == "__main__":
    main()
