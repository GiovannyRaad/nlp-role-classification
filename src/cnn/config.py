"""Configuration for CNN text role classification."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

ROLE_LABELS = {
    0: "learner",
    1: "teacher",
    2: "troll",
}

TRAINING_CONFIG = {
    "train_file": DATA_DIR / "split" / "train.csv",
    "test_file": DATA_DIR / "split" / "test.csv",
    "batch_size": 32,
    "epochs": 20,
    "learning_rate": 5e-4,
    "weight_decay": 5e-5,
    "grad_clip_norm": 1.0,
    "lr_scheduler_factor": 0.5,
    "lr_scheduler_patience": 2,
    "lr_scheduler_min_lr": 1e-6,
    "seed": 42,
    "device": "auto",  # auto | cuda | dml | cpu
    "dml_device_index": 1,
    "use_amp": True,
    "num_workers": 2,
    "optimizer": "adam",  # adam | adamw | adagrad | sgd
    "dml_optimizer": "adam",  # adam | adamw | adagrad | sgd
    "use_class_weights": False,
    "monitor_metric": "accuracy",  # accuracy | f1
    "early_stopping_patience": 4,
    "sgd_learning_rate": 1e-2,
    "sgd_momentum": 0.9,
    "sgd_nesterov": True,
}

TEXT_CONFIG = {
    "max_vocab_size": 40000,
    "min_freq": 1,
    "max_length": 256,
    "unk_token": "<UNK>",
    "pad_token": "<PAD>",
}

MODEL_CONFIG = {
    "embedding_dim": 200,
    "num_filters": 128,
    "kernel_sizes": (2, 3, 4, 5),
    "dropout": 0.4,
}

ARTIFACT_CONFIG = {
    "checkpoint_file": MODELS_DIR / "cnn_text_classifier.pt",
}

