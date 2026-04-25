"""Training script for multiclass role detection using TF-IDF + Logistic Regression."""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from config import DATA_DIR, MODEL_CONFIG, MODELS_DIR, ROLE_LABELS, TFIDF_CONFIG, TRAIN_CONFIG


def load_first_output_dataset(output_dir: Path):
    """Load the first CSV file found in data/output and return texts + numeric labels."""
    csv_files = sorted(output_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in output folder: {output_dir}")

    selected_file = csv_files[0]
    print(f"Using dataset file: {selected_file}")

    df = pd.read_csv(selected_file, dtype=str, keep_default_na=False)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Dataset {selected_file} must contain 'text' and 'label' columns. "
            f"Found columns: {list(df.columns)}"
        )

    role_to_id = {role: idx for idx, role in ROLE_LABELS.items()}
    labels_text = df["label"].astype(str).str.strip().str.lower()
    unknown_labels = sorted(set(labels_text) - set(role_to_id))
    if unknown_labels:
        raise ValueError(f"Unknown labels found in dataset: {unknown_labels}")

    texts = df["text"].astype(str).tolist()
    labels = labels_text.map(role_to_id).tolist()
    return texts, labels


def train_role_classifier(texts, labels):
    """Train multiclass role classifier with TF-IDF features."""
    print("Label mapping:")
    for idx, role in ROLE_LABELS.items():
        print(f"  {idx} = {role}")

    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        stratify=labels,
        **TRAIN_CONFIG,
    )

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
    output_dir = DATA_DIR / "output"
    texts, labels = load_first_output_dataset(output_dir)

    model, vectorizer = train_role_classifier(texts, labels)
    save_artifacts(model, vectorizer)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
