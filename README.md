# 🧠 xCEBRA Emotion Decoding

This project uses xCEBRA to decode emotional states from ECoG data by learning contrastive latent embeddings. Supports both single-patient and multi-patient aggregated training pipelines.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/JillianWang408/emotion-decoding.git 

#OR you can specify a different directory name like this:
git clone https://github.com/JillianWang408/emotion-decoding.git my-folder-path

cd emotion-decoding  # ✅ Change this if your root folder is located elsewhere
```

### 2. Create a Python 3.12 virtual environment

> Make sure you have Python 3.12 installed. Check with:
>
> ```bash
> python3.12 --version
> ```

**Using `venv`:**

```bash
python3.12 -m venv venv
source venv/bin/activate
```

**Or using `conda` (optional):**

```bash
conda create -n emotion-decoding python=3.12
conda activate emotion-decoding
```

### 3. Install project dependencies

```bash
pip install -r requirements.txt
```

---

## Directory Structure

- `src/`: Core scripts for training, evaluation, and attribution
  - `patient_aggregation.py`: Aggregate and z-score neural data across multiple patients
  - `patient_aggreagation_encoding_finetune.py`: Train encoder on aggregated multi-patient data
  - `full_encoding.py`: Single-patient unsupervised + supervised training
  - `full_encoding_finetune.py`: Single-patient two-head finetuning
  - `full_decoding.py`: Single-patient decoding pipeline
  - `full_decoding_finetune.py`: Single-patient decoding finetuning
  - `main.py`: Batch processing for multiple patients
- `output_xCEBRA_lags/`: Single-patient model outputs (models, embeddings, evaluations)
- `output_xCEBRA_cov/`: Single-patient covariance-based outputs
- `output_patient_aggregation/`: Aggregated multi-patient data and training outputs
- `data/`: ECoG and label files (.mat) organized by patient

---

## Workflows

### Workflow 1: Multi-Patient Aggregated Training (Recommended for larger datasets)

This workflow combines data from multiple patients, z-scores each patient independently, and trains a unified model.

#### Step 1: Aggregate Patient Data

Combine and z-score neural data from multiple patients:

```bash
python src/patient_aggregation.py --patient-ids 1 2 9 27
```

**Options:**
- `--patient-ids`: List of patient IDs to aggregate (default: all configured patients)
- `--output-dir`: Output directory (default: `output_patient_aggregation/`)
- `--no-save`: Don't save arrays/plot (display only)

**Outputs:**
- `aggregated_patient_data_<patient_codes>.npz`: Combined neural data, emotion labels, patient IDs
- `concatenated_heatmap_<patient_codes>.png`: Visualization of concatenated z-scored signals

**Note:** Patient 239 is automatically trimmed to first 630 timepoints to exclude artifacts.

#### Step 2: Train on Aggregated Data

Train encoder from scratch on aggregated data (unsupervised → supervised → two-head finetuning):

```bash
python src/patient_aggreagation_encoding_finetune.py \
  --aggregated-npz output_patient_aggregation/aggregated_patient_data_238_239_272_301.npz
```

**Key Hyperparameters:**
- `--unsup-steps`: Unsupervised CEBRA-Time steps (default: 3500)
- `--sup-steps`: Supervised CEBRA-TimeDelta steps (default: 2500)
- `--latent-dim`: Embedding dimension (default: 16)
- `--seq-len`: Sequence length for two-head training (default: 64)
- `--batch-size`: Batch size (default: 16)
- `--epochs`: Max epochs for two-head training (default: 20)
- `--lr-enc`: Encoder learning rate (default: 1e-5)
- `--lr-head`: Head learning rate (default: 3e-4)
- `--lambda-tc`: Temporal consistency weight (default: 0.1)

**Outputs:**
- `xcebra_unsupervised/`: Phase A outputs (model_weights.pt, embedding.pt)
- `xcebra_supervised/`: Phase B outputs (model_weights.pt, embedding.pt)
- `encoder_finetuned.pt`, `gate_head.pt`, `emo_head.pt`: Phase C model weights
- `embedding_finetuned.pt`: Final embeddings
- `finetune_logs.pt`: Training history and best tau threshold
- `finetune_meta.pt`: Hyperparameters and best results
- `finetune_curves.png`: Training curves

**Grid Search:** The tau threshold (for combining gate + emotion predictions) is automatically grid-searched during validation: `[0.1, 0.3, 0.4, 0.45, 0.5, 0.55]`

---

### Workflow 2: Single-Patient Training

Train models on individual patient data.

#### Full Pipeline (Multiple Patients)

Set patient IDs in `src/main.py`:

```python
PATIENT_IDS = [1, 2, 9, 27]  # Patient IDs to process
```

Then run:

```bash
caffeinate -dimsu python -m src.main
```

#### Individual Steps

1. **Set patient ID:**

```bash
export PATIENT_ID=1
```

2. **Train encoder (unsupervised + supervised):**

```bash
python src/full_encoding.py
```

3. **Finetune with two-head classifier:**

```bash
python src/full_encoding_finetune.py
```

4. **Decoding (optional):**

```bash
python src/full_decoding.py
python src/full_decoding_finetune.py
```

---

## Configuration

Patient configuration is defined in `src/config.py`:

```python
PATIENT_CONFIG = {
    1: ("EC238", "238"),
    2: ("EC239", "239"),
    9: ("EC272", "272"),
    # ... etc
}
```

Data paths are automatically resolved based on patient codes.

---

## Key Features

- **Per-patient z-scoring**: Each patient's neural features are z-scored independently before aggregation
- **Temporal consistency**: Two-head training includes temporal smoothing losses
- **Class weighting**: Automatic class weights for imbalanced emotion labels
- **Tau grid search**: Automatic threshold optimization for gate + emotion prediction combination
- **Comprehensive logging**: Training history, best hyperparameters, and metadata saved

---

## Output Structure

### Aggregated Training Outputs

```
output_patient_aggregation/
└── <patient_codes>/  # e.g., 238_239_272_301/
    ├── aggregated_patient_data_<codes>.npz
    ├── concatenated_heatmap_<codes>.png
    ├── xcebra_unsupervised/
    │   ├── model_weights.pt
    │   └── embedding.pt
    ├── xcebra_supervised/
    │   ├── model_weights.pt
    │   └── embedding.pt
    ├── encoder_finetuned.pt
    ├── gate_head.pt
    ├── emo_head.pt
    ├── embedding_finetuned.pt
    ├── finetune_logs.pt
    ├── finetune_meta.pt
    └── finetune_curves.png
```

### Single-Patient Outputs

```
output_xCEBRA_lags/
└── <patient_id>/
    ├── models/
    ├── evaluation_outputs/
    └── attribution_outputs/
```

---

## License

MIT
