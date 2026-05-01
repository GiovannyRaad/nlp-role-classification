"""Run inference with the saved CNN text classifier checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    from .config import ARTIFACT_CONFIG, TEXT_CONFIG, TRAINING_CONFIG
    from .train import TextCNN, encode_text
except ImportError:
    from config import ARTIFACT_CONFIG, TEXT_CONFIG, TRAINING_CONFIG
    from train import TextCNN, encode_text


def load_model_and_assets(checkpoint_path: Path) -> tuple[TextCNN, dict[str, int], dict[int, str]]:
    """Load model, vocabulary, and labels from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu")
    vocab: dict[str, int] = payload["vocab"]
    role_labels_raw = payload["role_labels"]
    role_labels = {int(k): str(v) for k, v in role_labels_raw.items()}
    text_cfg = payload["text_config"]
    model_cfg = payload["model_config"]

    model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=int(model_cfg["embedding_dim"]),
        num_classes=len(role_labels),
        num_filters=int(model_cfg["num_filters"]),
        kernel_sizes=tuple(model_cfg["kernel_sizes"]),
        dropout=float(model_cfg["dropout"]),
        pad_idx=vocab[text_cfg["pad_token"]],
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, vocab, role_labels


def resolve_dml_device(dml_device_index: int) -> tuple[torch.device, str]:
    """Resolve and validate a specific DirectML adapter index."""
    try:
        import torch_directml
    except ImportError as exc:
        raise RuntimeError(
            "DML was requested but torch-directml is not installed. Install with: pip install torch-directml"
        ) from exc

    count = int(torch_directml.device_count())
    if count <= 0:
        raise RuntimeError("No DirectML adapters were found.")
    if dml_device_index < 0 or dml_device_index >= count:
        raise RuntimeError(f"Invalid DML device index {dml_device_index}. Available range: 0..{count - 1}")

    adapter_name = str(torch_directml.device_name(dml_device_index))
    return torch_directml.device(dml_device_index), adapter_name


def select_device(device_pref: str, dml_device_index: int) -> tuple[torch.device, str, str]:
    """Resolve inference device from CLI preference."""
    normalized = str(device_pref).strip().lower()
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no GPU is available.")
        return torch.device("cuda"), "cuda", "cuda"
    if normalized == "dml":
        device, adapter_name = resolve_dml_device(dml_device_index)
        return device, "dml", adapter_name
    if normalized == "cpu":
        return torch.device("cpu"), "cpu", "cpu"
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda", "cuda"
    try:
        device, adapter_name = resolve_dml_device(dml_device_index)
        return device, "dml", adapter_name
    except RuntimeError:
        return torch.device("cpu"), "cpu", "cpu"


def predict_one(
    text: str,
    model: TextCNN,
    vocab: dict[str, int],
    role_labels: dict[int, str],
    device: torch.device,
    max_length: int,
) -> tuple[str, dict[str, float]]:
    """Predict one label and class probabilities."""
    input_ids = encode_text(text=text, vocab=vocab, max_length=max_length)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
        pred_id = int(torch.argmax(logits, dim=1).item())

    pred_label = role_labels.get(pred_id, f"unknown({pred_id})")
    prob_map = {role_labels[idx]: float(score) for idx, score in enumerate(probs)}
    return pred_label, prob_map


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Predict role for text using TextCNN")
    parser.add_argument("--text", help="Text to classify. If omitted, starts interactive mode.")
    parser.add_argument("--device", choices=["auto", "cuda", "dml", "cpu"], default="auto")
    parser.add_argument(
        "--dml-device-index",
        type=int,
        default=int(TRAINING_CONFIG.get("dml_device_index", 0)),
        help="DirectML adapter index",
    )
    return parser.parse_args()


def interactive_mode(
    model: TextCNN,
    vocab: dict[str, int],
    role_labels: dict[int, str],
    device: torch.device,
    max_length: int,
) -> None:
    """Interactive terminal mode."""
    print("Interactive mode. Type text and press Enter.")
    print("Type 'exit' or 'quit' to stop.")
    while True:
        text = input("\nText> ").strip()
        if text.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        if not text:
            print("Please enter non-empty text.")
            continue
        label, probs = predict_one(text, model, vocab, role_labels, device, max_length)
        print(f"Predicted label: {label}")
        print("Probabilities:")
        for role, score in sorted(probs.items(), key=lambda item: item[1], reverse=True):
            print(f"  {role}: {score:.4f}")


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    checkpoint_path = Path(ARTIFACT_CONFIG["checkpoint_file"])
    model, vocab, role_labels = load_model_and_assets(checkpoint_path)
    device, backend, adapter_name = select_device(args.device, args.dml_device_index)
    model.to(device)
    print(f"Using device: {device} ({backend}) | Adapter: {adapter_name}")
    max_length = int(TEXT_CONFIG["max_length"])

    if args.text:
        label, probs = predict_one(args.text, model, vocab, role_labels, device, max_length)
        print(f"Predicted label: {label}")
        print("Probabilities:")
        for role, score in sorted(probs.items(), key=lambda item: item[1], reverse=True):
            print(f"  {role}: {score:.4f}")
        return

    interactive_mode(model, vocab, role_labels, device, max_length)


if __name__ == "__main__":
    main()

