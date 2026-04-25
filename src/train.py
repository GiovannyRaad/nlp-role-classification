"""Training script for logistic regression model"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from config import MODEL_CONFIG, TRAIN_CONFIG, MODELS_DIR, DATA_DIR


def load_data(X_path, y_path):
    """Load features and labels from numpy files"""
    X = np.load(X_path)
    y = np.load(y_path)
    return X, y


def train_logistic_regression(X, y):
    """Train logistic regression model"""
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, **TRAIN_CONFIG
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    print("\nTraining logistic regression model...")
    model = LogisticRegression(**MODEL_CONFIG)
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    
    print(f"\nModel Performance:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    return model, X_test, y_test, y_pred


def save_model(model, model_name="logistic_regression.pkl"):
    """Save trained model to disk"""
    model_path = MODELS_DIR / model_name
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {model_path}")
    return model_path


def main():
    """Main training pipeline"""
    # Example: Load your data
    # X, y = load_data(DATA_DIR / "features.npy", DATA_DIR / "labels.npy")
    
    # For now, create dummy data for testing
    print("Creating dummy data for testing...")
    np.random.seed(42)
    X = np.random.randn(100, 20)  # 100 samples, 20 features
    y = np.random.randint(0, 2, 100)  # Binary classification
    
    # Train model
    model, X_test, y_test, y_pred = train_logistic_regression(X, y)
    
    # Save model
    save_model(model)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
