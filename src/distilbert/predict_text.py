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
    
    def __init__(self, model_dir="models/distilbert_role_classifier"):
        """
        Initialize the classifier with a pretrained model.
        
        Args:
            model_dir: Path to the directory containing the saved model and tokenizer
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
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
        
        # Get predicted class
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        predicted_label = self.id_to_label[predicted_class_id]
        
        result = {"label": predicted_label}
        
        if return_probabilities:
            # Get probabilities
            probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()
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
    """Example usage of the RoleClassifier."""
    
    # Initialize classifier
    classifier = RoleClassifier()
    
    # Example texts
    example_texts = [
        "I'm learning about machine learning and would like some recommendations.",
        "Based on my experience, here's what I recommend for best practices.",
        "You're all stupid and don't know what you're talking about.",
    ]
    
    print("=" * 60)
    print("Role Classification Predictions")
    print("=" * 60)
    
    for text in example_texts:
        print(f"\nText: {text}")
        prediction = classifier.predict(text, return_probabilities=True)
        print(f"Predicted Role: {prediction['label']}")
        print("Probabilities:")
        for role, prob in prediction['probabilities'].items():
            print(f"  {role}: {prob:.4f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
