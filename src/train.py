"""Training script for multiclass role detection using TF-IDF + Logistic Regression."""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from config import DATA_DIR, MODEL_CONFIG, MODELS_DIR, ROLE_LABELS, TFIDF_CONFIG


def dataframe_to_texts_and_labels(df: pd.DataFrame, source_name: str):
    """Validate and convert a dataframe into texts + numeric labels."""
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Dataset {source_name} must contain 'text' and 'label' columns. "
            f"Found columns: {list(df.columns)}"
        )

    role_to_id = {role: idx for idx, role in ROLE_LABELS.items()}
    labels_text = df["label"].astype(str).str.strip().str.lower()
    unknown_labels = sorted(set(labels_text) - set(role_to_id))
    if unknown_labels:
        raise ValueError(f"Unknown labels found in dataset {source_name}: {unknown_labels}")

    texts = df["text"].astype(str).tolist()
    labels = labels_text.map(role_to_id).tolist()
    return texts, labels


def load_split_datasets(split_dir: Path):
    """Load train/test CSV files from data/split and return mapped arrays."""
    train_path = split_dir / "train.csv"
    test_path = split_dir / "test.csv"

    if not train_path.exists():
        raise ValueError(f"Missing split file: {train_path}")
    if not test_path.exists():
        raise ValueError(f"Missing split file: {test_path}")

    print(f"Using train split: {train_path}")
    print(f"Using test split: {test_path}")

    train_df = pd.read_csv(train_path, dtype=str, keep_default_na=False)
    test_df = pd.read_csv(test_path, dtype=str, keep_default_na=False)

    X_train, y_train = dataframe_to_texts_and_labels(train_df, str(train_path))
    X_test, y_test = dataframe_to_texts_and_labels(test_df, str(test_path))

    return X_train, X_test, y_train, y_test


def train_role_classifier(X_train, X_test, y_train, y_test):
    """Train multiclass role classifier using pre-split datasets."""
    print("Label mapping:")
    for idx, role in ROLE_LABELS.items():
        print(f"  {idx} = {role}")

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    print("\nFitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training logistic regression model...")
    model = LogisticRegression(**MODEL_CONFIG)
    model.fit(X_train_vec, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\nModel Performance:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    return model, vectorizer


def save_artifacts(model, vectorizer):
    """Save trained model and TF-IDF vectorizer."""
    model_path = MODELS_DIR / "logistic_regression.pkl"
    vectorizer_path = MODELS_DIR / "tfidf_vectorizer.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"\nModel saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")


def main():
    """Main training pipeline."""
    split_dir = DATA_DIR / "split"
    X_train, X_test, y_train, y_test = load_split_datasets(split_dir)

    model, vectorizer = train_role_classifier(X_train, X_test, y_train, y_test)
    save_artifacts(model, vectorizer)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
