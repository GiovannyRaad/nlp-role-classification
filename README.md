# Role Classification Model

This repository contains a small NLP project for role classification in classroom-like discussion posts. It implements three models: a DistilBERT fine-tuned classifier, a Kim-style TextCNN, and a baseline logistic regression.

**Quick Start**

- **Create virtualenv & install**:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # PowerShell
pip install -r requirements.txt
```

- **Prepare data (default paths):**

```bash
python src/run_data_pipeline.py        # cleans/merges/splits data under data/
```

- **Train DistilBERT**

```bash
python src/distilbert/train.py
```

- **Train CNN**

```bash
python src/cnn/train.py
```

- **Run inference (DistilBERT)**

```python
from src.distilbert.predict_text import RoleClassifier
rc = RoleClassifier()
rc.predict("Example post text here")
```

**Project Layout**

- **data/**: raw, cleaned, and split CSVs. See data/split/ for `train.csv` and `test.csv`.
- **embeddings/**: (not checked into git) put pretrained GloVe files here (e.g. `glove.6B.200d.txt`). See `embeddings/DOWNLOAD_LINKS.txt` for links.
- **models/**: saved model artifacts. DistilBERT output: `models/distilbert_role_classifier/`.
- **src/**: main code. Key modules:
  - [src/distilbert/config.py](src/distilbert/config.py) — DistilBERT config and paths
  - [src/distilbert/train.py](src/distilbert/train.py) — DistilBERT training (custom loss)
  - [src/distilbert/predict_text.py](src/distilbert/predict_text.py) — inference util
  - [src/cnn/config.py](src/cnn/config.py) — CNN hyperparams & embeddings settings
  - [src/cnn/train.py](src/cnn/train.py) — TextCNN training and pretrained embedding loader
  - [src/merge_labeled_datasets.py](src/merge_labeled_datasets.py) — merges labeled CSVs in `data/`
  - [scripts/download_glove.py](scripts/download_glove.py) — helper to download GloVe (run locally)

**What to know about implementation**

- Task: 3-class classification with labels `learner`, `teacher`, `troll`.
- DistilBERT details:
  - Tokenizer: `distilbert-base-uncased` with `max_length=256`.
  - Training: label smoothing = 0.1 and an asymmetric false-troll penalty added to the loss to reduce false troll positives. The penalty weight is set in `src/distilbert/train.py` (`FALSE_TROLL_PENALTY_WEIGHT`).
  - Inference: a predicted `troll` is only accepted if its probability > 0.8; otherwise the highest non-troll class is returned. See [src/distilbert/predict_text.py](src/distilbert/predict_text.py).

- TextCNN details:
  - Kim-style multiple kernel sizes, global max pooling, dropout, linear output head.
  - Pretrained embeddings supported (GloVe / word2vec). Put GloVe file(s) in `embeddings/`, set `PRETRAINED_EMBEDDINGS['path']` and `MODEL_CONFIG['embedding_dim']` in [src/cnn/config.py](src/cnn/config.py).
  - You can choose to freeze pretrained embeddings or fine-tune them by toggling `PRETRAINED_EMBEDDINGS['freeze']` in `src/cnn/config.py`.

**Reproducibility & paths**

- All scripts resolve paths relative to their script/project root (so run them from anywhere). Default data directory is `data/` in the repository root. Models are saved to `models/`.
- Embeddings are ignored via `.gitignore` — place large pretrained files in `embeddings/` locally.

**Experiments to try (recommended)**

- Compare CNN variants:
  - Randomly initialized embeddings
  - GloVe frozen (no fine-tuning)
  - GloVe fine-tuned (allow gradients)
- Compare DistilBERT runs with/without asymmetric penalty and with different `FALSE_TROLL_PENALTY_WEIGHT` values.

**Troubleshooting**

- GPU note: Some new consumer GPUs (e.g., RTX 5060 / compute sm_120) may not be supported by the currently installed PyTorch build; if CUDA kernels fail, training will fall back to CPU. See [src/distilbert/config.py](src/distilbert/config.py) for training settings.
- If you need GloVe files, see `embeddings/DOWNLOAD_LINKS.txt` for URLs and instructions.

**Where to change hyperparameters**

- DistilBERT hyperparams: [src/distilbert/config.py](src/distilbert/config.py) and [src/distilbert/train.py](src/distilbert/train.py).
- CNN hyperparams and embedding settings: [src/cnn/config.py](src/cnn/config.py).

**Contact / Notes**

- If your teacher wants the exact commands I ran or model-run logs, tell me which model and I will add run commands and example outputs to this README.

---

Generated for teacher evaluation. File: [README.md](README.md)

## Example Commands

Below are concrete commands and quick edits you can use to run the pipeline and experiments.

- Prepare the environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- Prepare the data (cleans, merges, splits):

```bash
python src/run_data_pipeline.py
```

- Train DistilBERT (uses defaults in `src/distilbert/config.py`):

```bash
python src/distilbert/train.py
```

- Adjust the asymmetric false-troll penalty:

Open `src/distilbert/train.py` and change the `FALSE_TROLL_PENALTY_WEIGHT` constant to experiment with values (e.g. `1.5`).

- Single-shot DistilBERT inference from the command line:

```bash
python -c "from src.distilbert.predict_text import RoleClassifier; print(RoleClassifier().predict('I have a question about the homework'))"
```

- Train TextCNN with pretrained embeddings:

1. Put `glove.6B.200d.txt` (or another dimension) into `embeddings/`.
2. Edit `src/cnn/config.py`: set `PRETRAINED_EMBEDDINGS['path']` to the file path and set `MODEL_CONFIG['embedding_dim']` to the file's dimension. Toggle `PRETRAINED_EMBEDDINGS['freeze']` to `True` or `False`.

```bash
python src/cnn/train.py
```

- Download GloVe manually (example PowerShell):

```powershell
(New-Object System.Net.WebClient).DownloadFile('https://nlp.stanford.edu/data/glove.6B.zip','glove.6B.zip')
Expand-Archive glove.6B.zip -DestinationPath embeddings\
```

- Artifact locations:
  - DistilBERT model directory: `models/distilbert_role_classifier/` (see `src/distilbert/config.py`).
  - CNN checkpoint: `models/cnn_text_classifier.pt` (see `src/cnn/config.py`).

---

# nlp-role-classification
