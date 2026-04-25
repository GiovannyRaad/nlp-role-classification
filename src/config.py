"""Configuration settings for the multiclass logistic regression model."""

from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SRC_DIR = PROJECT_ROOT / "src"

# Model hyperparameters
MODEL_CONFIG = {
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": 42,
    
    "verbose": 1,
}

# TF-IDF vectorizer configuration
TFIDF_CONFIG = {
    "lowercase": True,
    "ngram_range": (1, 2),
    "min_df": 1,
    "max_features": 5000,
}

# Training configuration
TRAIN_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
}

# Role label mapping for multiclass training
ROLE_LABELS = {
    0: "learner",
    1: "teacher",
    2: "troll",
}
