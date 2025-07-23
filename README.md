# 🧠 xCEBRA Emotion Decoding

This project uses xCEBRA to decode emotional states from ECoG data by learning contrastive latent embeddings.

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
- `src/`: Core scripts for training, evaluation, and attribution (Omit the hybrid_training subfolder containing outdated codes.)
- `output`: Models, embedding, evaluation, and attribution maps
    - patient_x.0: Output for patient X after hybrid training (outdated)
    - patient_x: Output for patient X after supervised training (all latent based on emotion labels) ✅
        - `models/`: Saved model weights
        - `evaluation_outputs/`: embedding 3D visualization and confusion matrix
        - `attibution_outputs/`: output images from running the attribution.py module, including attribution maps and covariance matrix for each latent dimension
    - aggreate_outputs: aggregated normalized covariance matrix from selected patients
- `data/`: ECoG and label files (.mat)

## How to run (Full Pipeline)

To run training → embedding → evaluation → attribution for multiple patients, set the list of patients in src/main.py:

src/main.py

```python
PATIENT_IDS = [1.1, 2.1, 3.1, 4.1, 5.1] #only put in the patient ID you want to train
```

Then run:

```bash
caffeinate -dimsu python -m src.main
```

Each patient’s results will be saved to output/patient\_{id}/ as defined in config.py.

## How to run (Individual Steps)

To run each module manually, first update the line in src/config.py:

```bash
export PATIENT_ID=3.1
```

Then run:

1. Train supervised model:

```bash
caffeinate -dimsu python -m src.train_supervised
```

2. Compute embedding:

```bash
python -m src.embedding
```

3. Evaluate embedding:

```bash
python -m src.evaluate_supervised
```

4. Run attribution analysis:

```bash
python -m src.attribution_supervised
```

Outputs will be saved in the patient-specific folders under output/.

## License
MIT
