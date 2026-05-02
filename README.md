# Role Classification Model

This project implements a natural language processing system for classifying discussion posts into three roles: **learner**, **teacher**, and **troll**.

Three models are included:

* Logistic Regression (baseline)
* CNN-based text classifier
* DistilBERT (final model)

---

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

---

## Data

The dataset is already included in the repository.

Training and test files are located in:

```
data/split/
```

---

## Training

### Train DistilBERT

```bash
python src/distilbert/train.py
```

### Train CNN

```bash
python src/cnn/train.py
```

### Train Logistic Regression

```bash
python src/logreg/train.py
```

---

## Project Structure

```
data/       → dataset (train/test)
models/     → saved models
src/        → training and model code
```

---

## Notes

* Task: multiclass classification (learner / teacher / troll)
* DistilBERT is the best-performing model
* All models use the same dataset split for fair comparison

---

## Authors

Giovanny Raad, Karim Hawa
