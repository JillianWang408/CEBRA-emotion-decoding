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
- `src/`: training, evaluation, visualization code
- `models/`: saved .pt model files
- `data/`: ECoG and label files (.mat)
- `notebooks/`: optional Colab/Jupyter files
- `attibution_outputs/`: output images from running the attribution.py module

## How to run .py files under src
Copy these command line into terminal when you open each file

1. `train.py`: 
```bash
caffeinate -i python train.py
python -m src.embedding
```
2. `embedding.py`: 
```bash
python -m src.embedding
```
2. `evaluate.py`: 
```bash
python -m src.evaluate
```
3. `attribution.py`:
```bash
python -m src.attibution
```

## License
MIT
