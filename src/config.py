"""Configuration settings for the logistic regression model"""

import os
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

# Training configuration
TRAIN_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
}
