"""
Training script for DistilBERT role classifier.
Finetunes DistilBERT on the role classification task (learner, teacher, troll).
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from datasets import Dataset
import logging

from config import DistilBertConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RoleClassificationDataset(Dataset):
    """Custom dataset class for role classification."""
    
    def __init__(self, data_path, tokenizer, config):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to CSV file with 'text' and 'label' columns
            tokenizer: HuggingFace tokenizer
            config: Configuration object
        """
        self._data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.config = config
        
        # Verify required columns exist
        if 'text' not in self._data.columns or 'label' not in self._data.columns:
            raise ValueError("CSV must contain 'text' and 'label' columns")
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        # Support both single index and batch index (list/array/Series)
        if isinstance(idx, (list, np.ndarray, pd.Series)):
            rows = self._data.iloc[idx]
            texts = rows['text'].astype(str).tolist()
            labels = rows['label'].tolist()
            # Convert labels to IDs
            label_ids = [self.config.LABEL_TO_ID.get(l, 0) if isinstance(l, str) else int(l) for l in labels]
            # Tokenize batch
            encoding = self.tokenizer(
                texts,
                max_length=self.config.MAX_LENGTH,
                truncation=self.config.TRUNCATION,
                padding=self.config.PADDING,
                return_tensors="pt"
            )
            return {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "labels": torch.tensor(label_ids, dtype=torch.long),
            }
        else:
            row = self._data.iloc[idx]
            text = str(row['text'])
            label = row['label']
            # Convert label to ID if it's a string
            if isinstance(label, str):
                label_id = self.config.LABEL_TO_ID.get(label, 0)
            else:
                label_id = int(label)
            # Tokenize single
            encoding = self.tokenizer(
                text,
                max_length=self.config.MAX_LENGTH,
                truncation=self.config.TRUNCATION,
                padding=self.config.PADDING,
                return_tensors="pt"
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": torch.tensor(label_id, dtype=torch.long),
            }


def compute_metrics(pred):
    """Compute metrics for evaluation."""
    predictions = np.argmax(pred.predictions, axis=1)
    references = pred.label_ids
    
    accuracy = accuracy_score(references, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        references, predictions, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def train_distilbert(config=None):
    """
    Main training function.
    
    Args:
        config: Configuration object (uses DistilBertConfig if None)
    """
    if config is None:
        config = DistilBertConfig()
    
    # Set random seed
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)
    
    logger.info("=" * 50)
    logger.info("DistilBERT Role Classification Training")
    logger.info("=" * 50)
    
    # Create output directory
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer and model
    logger.info(f"Loading tokenizer and model from {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_LABELS,
        id2label=config.ID_TO_LABEL,
        label2id=config.LABEL_TO_ID,
    )
    # Log device info
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        logger.info("CUDA is NOT available. Using CPU.")
    
    # Load datasets
    logger.info(f"Loading training data from {config.TRAIN_DATA_PATH}")
    train_dataset = RoleClassificationDataset(config.TRAIN_DATA_PATH, tokenizer, config)
    logger.info(f"Training set size: {len(train_dataset)}")
    
    logger.info(f"Loading test data from {config.TEST_DATA_PATH}")
    eval_dataset = RoleClassificationDataset(config.TEST_DATA_PATH, tokenizer, config)
    logger.info(f"Evaluation set size: {len(eval_dataset)}")
    
    # Calculate warmup steps (10% of total training steps)
    num_train_epochs = config.EPOCHS
    total_train_steps = (len(train_dataset) // config.BATCH_SIZE + 1) * num_train_epochs
    warmup_steps = int(total_train_steps * 0.1)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        warmup_steps=warmup_steps,
        weight_decay=config.WEIGHT_DECAY,
        max_grad_norm=config.MAX_GRAD_NORM,
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=config.EVAL_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=50,
        seed=config.SEED,
        optim=config.OPTIMIZER,
        lr_scheduler_type=config.SCHEDULER_TYPE,
        report_to=config.REPORT_TO,
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=config.EARLY_STOPPING_THRESHOLD,
            )
        ],
    )
    
    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    
    logger.info(f"Training completed in {train_result.training_loss:.2f} seconds")
    logger.info(f"Final training loss: {train_result.training_loss:.4f}")
    
    # Evaluate
    logger.info("Evaluating on test set...")
    eval_result = trainer.evaluate()
    
    logger.info("=" * 50)
    logger.info("Final Results:")
    logger.info("=" * 50)
    for key, value in eval_result.items():
        logger.info(f"{key}: {value:.4f}")
    
    # Save model and tokenizer
    logger.info(f"Saving model to {config.OUTPUT_DIR}")
    model.save_pretrained(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    
    logger.info("Training completed successfully!")
    
    return model, tokenizer, eval_result


if __name__ == "__main__":
    train_distilbert()
