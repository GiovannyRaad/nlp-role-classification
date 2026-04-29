"""Run inference on external text using saved role classification artifacts."""

import argparse
import pickle
from pathlib import Path

try:
    from .config import MODELS_DIR, ROLE_LABELS
except ImportError:
    from config import MODELS_DIR, ROLE_LABELS


def load_artifacts(models_dir: Path):
    """Load trained model and vectorizer from disk."""
    model_path = models_dir / "logistic_regression.pkl"
    vectorizer_path = models_dir / "tfidf_vectorizer.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


def predict_text(text: str, model, vectorizer):
    """Predict role label for one text sample."""
    text_vector = vectorizer.transform([text])
    predicted_id = int(model.predict(text_vector)[0])
    predicted_label = ROLE_LABELS.get(predicted_id, f"unknown({predicted_id})")

    probs = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(text_vector)[0]
        probs = {
            ROLE_LABELS.get(int(class_id), f"unknown({int(class_id)})"): float(prob)
            for class_id, prob in zip(model.classes_, probabilities)
        }

    return predicted_id, predicted_label, probs


def interactive_mode(model, vectorizer):
    """Prompt user for text repeatedly until exit."""
    print("Interactive mode. Type your text and press Enter.")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        text = input("\nText> ").strip()
        if text.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        if not text:
            print("Please enter non-empty text.")
            continue

        _, label, probs = predict_text(text, model, vectorizer)
        print(f"Predicted label: {label}")
        if probs is not None:
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            print("Probabilities:")
            for role, score in sorted_probs:
                print(f"  {role}: {score:.4f}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Predict role for external text")
    parser.add_argument(
        "--text",
        help="Text to classify. If omitted, script starts interactive mode.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    model, vectorizer = load_artifacts(MODELS_DIR)

    if args.text:
        _, label, probs = predict_text(args.text, model, vectorizer)
        print(f"Predicted label: {label}")
        if probs is not None:
            sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            print("Probabilities:")
            for role, score in sorted_probs:
                print(f"  {role}: {score:.4f}")
        return

    interactive_mode(model, vectorizer)


if __name__ == "__main__":
    main()
