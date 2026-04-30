"""
Configuration for DistilBERT finetuning for role classification.
Best practices parameters for 3-class classification task (learner, teacher, troll).
"""

class DistilBertConfig:
    """Configuration parameters for DistilBERT model training."""
    
    # Model configuration
    MODEL_NAME = "distilbert-base-uncased"
    NUM_LABELS = 3
    LABELS = ["learner", "teacher", "troll"]
    LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
    ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}
    
    # Training hyperparameters
    LEARNING_RATE = 2e-5
    EPOCHS = 4  # Usually 3-5 epochs for transfer learning
    BATCH_SIZE = 32  # DistilBERT is smaller, can handle larger batches
    EVAL_BATCH_SIZE = 64
    WARMUP_STEPS = 0 
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    
    # Text preprocessing
    MAX_LENGTH = 256  # DistilBERT max is 512, but 256 is usually sufficient for this task
    TRUNCATION = True
    PADDING = "max_length"
    
    # Optimization
    OPTIMIZER = "adamw_torch"  # AdamW is best for transformer finetuning
    SCHEDULER_TYPE = "linear"  # Linear warmup schedule is standard
    
    # Training/eval configuration
    SEED = 42
    EVAL_STEPS = 100  # Evaluate every N steps during training
    SAVE_STEPS = 100
    LOG_LEVEL = "info"
    REPORT_TO = ["tensorboard"]
    
    # Paths
    OUTPUT_DIR = "../../models/distilbert_role_classifier"
    TRAIN_DATA_PATH = "../../data/split/train.csv"
    TEST_DATA_PATH = "../../data/split/test.csv"
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 3
    EARLY_STOPPING_THRESHOLD = 0.001
