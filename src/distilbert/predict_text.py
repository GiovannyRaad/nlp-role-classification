"""
Inference script for the finetuned DistilBERT role classifier.
Allows making predictions on new text samples.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

logger = logging.getLogger(__name__)


class RoleClassifier:
    """DistilBERT-based role classifier for inference."""
    
    def __init__(self, model_dir=None):
        """
        Initialize the classifier with a pretrained model.
        Args:
            model_dir: Path to the directory containing the saved model and tokenizer (default: relative to this script)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Set default model_dir relative to this script if not provided
        if model_dir is None:
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "models", "distilbert_role_classifier"))

        # Try loading the fast tokenizer, fallback to slow if needed
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        except Exception as e:
            logger.warning(f"Fast tokenizer failed to load: {e}\nTrying to load the slow tokenizer...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
            except Exception as e2:
                logger.error(f"Failed to load both fast and slow tokenizers.\nFast error: {e}\nSlow error: {e2}\nMake sure you have 'tokenizers' and 'sentencepiece' installed, and your model/tokenizer files are not corrupted.")
                raise

        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        # Get label mappings
        self.id_to_label = self.model.config.id2label
        self.label_to_id = self.model.config.label2id

        logger.info(f"Model loaded successfully from {model_dir}")
        logger.info(f"Labels: {list(self.id_to_label.values())}")
    
    def predict(self, text, return_probabilities=False):
        """
        Predict the role for a given text.
        
        Args:
            text: Input text to classify
            return_probabilities: If True, return probabilities for all classes
        
        Returns:
            Dictionary with 'label' and optionally 'probabilities'
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Get probabilities
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        
        # Get predicted class
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        predicted_label = self.id_to_label[predicted_class_id]
        
        # Apply troll confidence threshold (must be > 80%)
        troll_id = self.label_to_id.get("troll", 2)
        troll_confidence = probabilities[troll_id]
        
        if predicted_class_id == troll_id and troll_confidence <= 0.8:
            # If troll prediction but below 80% confidence, use argmax of other classes
            other_indices = [i for i in range(len(probabilities)) if i != troll_id]
            best_other_idx = max(other_indices, key=lambda i: probabilities[i])
            predicted_class_id = best_other_idx
            predicted_label = self.id_to_label[predicted_class_id]
        
        result = {"label": predicted_label}
        
        if return_probabilities:
            result["probabilities"] = {
                self.id_to_label[i]: float(prob)
                for i, prob in enumerate(probabilities)
            }
        
        return result
    
    def predict_batch(self, texts, return_probabilities=False):
        """
        Predict roles for a batch of texts.
        
        Args:
            texts: List of input texts to classify
            return_probabilities: If True, return probabilities for all classes
        
        Returns:
            List of dictionaries with predictions
        """
        results = []
        for text in texts:
            results.append(self.predict(text, return_probabilities))
        return results


def main():
    """Interactive usage of the RoleClassifier."""
    classifier = RoleClassifier()
    print("=" * 60)
    print("Role Classification Interactive Prediction")
    print("Type your text and press Enter (or just press Enter to exit)")
    print("=" * 60)
    while True:
        text = input("\nEnter text: ").strip()
        if not text:
            print("Exiting.")
            break
        prediction = classifier.predict(text, return_probabilities=True)
        print(f"Predicted Role: {prediction['label']}")
        print("Probabilities:")
        for role, prob in prediction['probabilities'].items():
            print(f"  {role}: {prob:.4f}")


if __name__ == "__main__":
    main()
