import numpy as np
import torch
import mat73
import scipy.io
from pathlib import Path
from cebra.data import DatasetxCEBRA, ContrastiveMultiObjectiveLoader
from cebra.models import init as init_model

from src.config import (
    MODEL_DIR, PROJECT_ROOT, output_dir,  
    NEURAL_PATH, EMOTION_PATH,         
    ELECTRODE_NAMES, NODE_MAP,
    N_LATENTS, BEHAVIOR_INDICES
)
# [CHANGE] remember to change paths in config.py too if needed

from src.utils_training import (
    make_and_save_split,
    feat_indices_for_electrodes,
    build_cebra_config,
    train_and_save,             
)

# DC5 layout constants
N_ELECTRODES = 40
N_BANDS = 5
N_LAGS  = 5

# [CHANGE] Output directory: output_xCEBRA/<id>/OFC_only_model/
#OUT_DIR = PROJECT_ROOT / "output_xCEBRA_lags" / output_dir / "OFC_only_model"
OUT_DIR = PROJECT_ROOT / "output_xCEBRA_cov" / output_dir / "OFC_only_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OFC_NODES = ["LOFC_Medial", "LOFC_Lateral", "ROFC_Medial", "ROFC_Lateral"]

def ofc_electrode_indices():
    """Return OFC electrode indices in ELECTRODE_NAMES order, plus their names."""
    wanted = []
    for node in OFC_NODES:
        wanted.extend(NODE_MAP[node])
    wanted_set = set(wanted) 
    ordered_names = [nm for nm in ELECTRODE_NAMES if nm in wanted_set]  # preserve ELECTRODE_NAMES order
    name_to_idx = {nm: i for i, nm in enumerate(ELECTRODE_NAMES)}
    idx = [name_to_idx[nm] for nm in ordered_names]
    if len(idx) != 16:
        print(f"[WARN] Expected 16 OFC electrodes, found {len(idx)}: {ordered_names}")
    else:
        print(f"[INFO] Using {len(idx)} OFC electrodes.")
    return idx, ordered_names

def cov_block_indices(row_idx, col_idx, n_electrodes=40, upper_only=False):
    """
    Flattened indices for a (row_idx × col_idx) block from a row-major [E*E] vector per time.
    If upper_only=True and row_idx==col_idx, keep only i<=j (unique symmetric entries).
    """
    idxs = []
    if upper_only and row_idx == col_idx:
        for off, i in enumerate(row_idx):
            for j in col_idx[off:]:               # j >= i within the block
                idxs.append(i * n_electrodes + j)
    else:
        for i in row_idx:
            base = i * n_electrodes               # row-major
            for j in col_idx:
                idxs.append(base + j)
    return np.asarray(idxs, dtype=int)


def main():
    # === Load data (from config data paths) ===
    neural_array  = mat73.loadmat(NEURAL_PATH)['stim'].T             
    emotion_array = scipy.io.loadmat(EMOTION_PATH)['resp'].flatten()
    T, D = neural_array.shape

    #[CHANGE]
    #assert D == N_ELECTRODES * N_BANDS * N_LAGS, f"D={D}, expected {N_ELECTRODES*N_BANDS*N_LAGS}"
    assert D == N_ELECTRODES * N_ELECTRODES, f"D={D}, expected {N_ELECTRODES*N_ELECTRODES} for covariance"

    # === Load the split ===
    train_idx, test_idx = make_and_save_split(T, OUT_DIR)

    #[CHANGE] 
    # === Select OFC features only (L–E–B ordering) ===
    # ofc_indices, ofc_names = ofc_electrode_indices()
    # feat_idx = feat_indices_for_electrodes(ofc_indices, n_electrodes=N_ELECTRODES,
    #                                        n_bands=N_BANDS, n_lags=N_LAGS)
    # assert len(feat_idx) == 16 * N_BANDS * N_LAGS, "feat_idx length mismatch"
    # print(f"[INFO] OFC feature cols = {len(feat_idx)} (16 * 5 * 5 = 400)")

    # === Select OFC×OFC covariance features ===
    ofc_indices, ofc_names = ofc_electrode_indices()
    feat_idx = cov_block_indices(ofc_indices, ofc_indices, n_electrodes=N_ELECTRODES, upper_only=False)
    assert len(feat_idx) == 16 * 16, "feat_idx length mismatch for OFC×OFC block"
    print(f"[INFO] OFC×OFC cov feature cols = {len(feat_idx)} (16×16 = 256)")


    # === Build training tensors (OFC only) ===
    X_train = torch.from_numpy(neural_array[:, feat_idx][train_idx]).float().contiguous()
    y_train = torch.from_numpy(emotion_array[train_idx]).float().unsqueeze(1).contiguous()
    print("[INFO] Train shapes:", X_train.shape, y_train.shape)

    # Full tensor for embedding export on WHOLE dataset with same feat_idx
    full_neural_tensor = torch.tensor(neural_array, dtype=torch.float32)
    emotion_tensor_full = torch.tensor(emotion_array, dtype=torch.float32).unsqueeze(1)

    # === Dataset & Loader ===
    ds = DatasetxCEBRA(neural=X_train, position=y_train)
    loader = ContrastiveMultiObjectiveLoader(dataset=ds, batch_size=512, num_steps=1000)

    # === Config ===
    config = build_cebra_config(loader, BEHAVIOR_INDICES, temperature=1.0)

    # === Model ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = init_model(
        name="offset10-model",
        num_neurons=ds.neural.shape[1],
        num_units=256,
        num_output=N_LATENTS
    ).to(device)

    ds.configure_for(model)

    # === Train & Save into OUT_DIR ===
    meta = {
        "feat_idx": feat_idx.tolist(),
        "electrodes": ofc_names,
        "nodes": OFC_NODES,
        "n_electrodes": N_ELECTRODES
    }

    _ = train_and_save(
        model=model,
        loader=loader,
        config=config,
        out_dir=OUT_DIR,               
        full_neural_tensor=full_neural_tensor,
        meta=meta,
        device=device,
        num_steps=1000
    )

    print(f"[DONE] Saved weights & embedding in: {OUT_DIR}")

if __name__ == "__main__":
    main()