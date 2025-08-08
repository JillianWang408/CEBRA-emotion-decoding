import os
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np
import mat73
import scipy.io
from config import {N_NODES, NEURAL_PATH, EMOTION_PATH, MODEL_DIR}

def train_nodewise_logistic(neural_array, emotion_array, save_dir):
    """
    Train multinomial logistic regression for each node (4 electrodes each)
    and save weights for analysis.
    """

    train_idx = np.load("splits/train_idx.npy")
    neural_array = mat73.loadmat(NEURAL_PATH)['stim'].T
    emotion_array = scipy.io.loadmat(EMOTION_PATH)['resp'].flatten()
    
    N_NODES = neural_array.shape[1] // 4  # 40 electrodes, 10 nodes
    assert neural_array.shape[1] == 40, "Expected 40 electrodes (features)"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    weights_all_nodes = []  # For saving all weight matrices

    for node in range(N_NODES):
        cols = list(range(4 * node, 4 * (node + 1)))  # 4 electrodes/node
        X_node = neural_array[:, cols]
        y = emotion_array.astype(int)  # Shape: (n_samples,)
        # Remove possible NaNs in labels
        mask = ~np.isnan(y)
        X_node = X_node[mask]
        y = y[mask]

        # Train multinomial logistic regression
        clf = LogisticRegression(
            multi_class='multinomial', solver='lbfgs', max_iter=2000, C=1.0
        )
        clf.fit(X_node, y)
        weights = clf.coef_  # shape: (10 classes, 4 electrodes)
        intercept = clf.intercept_  # (10,)
        weights_all_nodes.append(weights)

        # Save each node's model and weights if needed
        with open(os.path.join(save_dir, f'node_{node+1}_logreg.pkl'), 'wb') as f:
            pickle.dump(clf, f)
        np.save(os.path.join(save_dir, f'node_{node+1}_weights.npy'), weights)
        np.save(os.path.join(save_dir, f'node_{node+1}_intercept.npy'), intercept)

        print(f"Node {node+1}: trained, weights shape = {weights.shape}")

    # Optionally save all weights together
    np.save(os.path.join(save_dir, 'all_node_weights.npy'), np.array(weights_all_nodes))
    print("All nodes processed and weights saved.")

def main():
    # === Load Data ===
    train_idx = np.load("splits/train_idx.npy")
    neural_array = mat73.loadmat(NEURAL_PATH)['stim'].T
    emotion_array = scipy.io.loadmat(EMOTION_PATH)['resp'].flatten()

    # Train and save nodewise logistic regression models
    train_nodewise_logistic(
        neural_array=neural_array[train_idx],  # training data only
        emotion_array=emotion_array[train_idx],
        save_dir=str(MODEL_DIR / "nodewise_logreg")
    )

if __name__ == "__main__":
    main()
