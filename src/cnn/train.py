"""Train a CNN text classifier on data/split train/test CSVs."""

from __future__ import annotations

import argparse
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from .config import ARTIFACT_CONFIG, MODEL_CONFIG, ROLE_LABELS, TEXT_CONFIG, TRAINING_CONFIG
except ImportError:
    from config import ARTIFACT_CONFIG, MODEL_CONFIG, ROLE_LABELS, TEXT_CONFIG, TRAINING_CONFIG


def set_seed(seed: int) -> None:
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    """Resolve runtime device from config/CLI preference."""
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


def tokenize(text: str) -> list[str]:
    """Very simple whitespace tokenizer."""
    return str(text).strip().lower().split()


def read_split(path: Path) -> pd.DataFrame:
    """Read and validate one split file."""
    if not path.exists():
        raise ValueError(f"Missing split file: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} must contain text,label columns. Found: {list(df.columns)}")
    df = df[["text", "label"]].copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["text"].str.len() > 0].copy()
    return df


def build_vocab(train_texts: list[str]) -> dict[str, int]:
    """Build token->id vocabulary from training texts only."""
    counter: Counter[str] = Counter()
    for text in train_texts:
        counter.update(tokenize(text))

    pad_token = TEXT_CONFIG["pad_token"]
    unk_token = TEXT_CONFIG["unk_token"]
    min_freq = int(TEXT_CONFIG["min_freq"])
    max_vocab_size = int(TEXT_CONFIG["max_vocab_size"])

    vocab = {pad_token: 0, unk_token: 1}
    for token, count in counter.most_common():
        if count < min_freq:
            continue
        if len(vocab) >= max_vocab_size:
            break
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_length: int) -> list[int]:
    """Convert text to fixed-length token id sequence."""
    unk_id = vocab[TEXT_CONFIG["unk_token"]]
    pad_id = vocab[TEXT_CONFIG["pad_token"]]
    token_ids = [vocab.get(tok, unk_id) for tok in tokenize(text)]
    token_ids = token_ids[:max_length]
    if len(token_ids) < max_length:
        token_ids.extend([pad_id] * (max_length - len(token_ids)))
    return token_ids


def role_to_id_map() -> dict[str, int]:
    """Build label->id map from ROLE_LABELS config."""
    return {label: idx for idx, label in ROLE_LABELS.items()}


class RoleTextDataset(Dataset):
    """Torch dataset with pre-encoded texts and numeric labels."""

    def __init__(self, texts: list[str], labels: list[int], vocab: dict[str, int], max_length: int):
        self.features = [encode_text(text, vocab, max_length) for text in texts]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.features[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class TextCNN(nn.Module):
    """Kim-style CNN for sentence classification."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_classes: int,
        num_filters: int,
        kernel_sizes: tuple[int, ...],
        dropout: float,
        pad_idx: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)  # [batch, seq_len, emb]
        embedded = embedded.transpose(1, 2)  # [batch, emb, seq_len]
        pooled_outputs = []
        for conv in self.convs:
            conved = torch.relu(conv(embedded))  # [batch, filters, seq']
            pooled = torch.max(conved, dim=2).values  # [batch, filters]
            pooled_outputs.append(pooled)
        cat = torch.cat(pooled_outputs, dim=1)  # [batch, filters * kernels]
        cat = self.dropout(cat)
        return self.classifier(cat)


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> dict[str, float]:
    """Run model evaluation and return core classification metrics."""
    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    return {
        "accuracy": accuracy_score(all_targets, all_preds),
        "precision": precision_score(all_targets, all_preds, average="weighted", zero_division=0),
        "recall": recall_score(all_targets, all_preds, average="weighted", zero_division=0),
        "f1": f1_score(all_targets, all_preds, average="weighted", zero_division=0),
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler,
    use_amp: bool,
    grad_clip_norm: float | None = None,
) -> float:
    """Train one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    total_steps = 0
    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        if use_amp:
            with torch.autocast(device_type="cuda", enabled=True):
                logits = model(inputs)
                loss = criterion(logits, targets)
        else:
            logits = model(inputs)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        if grad_clip_norm is not None and grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item())
        total_steps += 1
    return total_loss / max(1, total_steps)


def create_grad_scaler(use_amp: bool):
    """Create a GradScaler compatible with current torch APIs."""
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def create_optimizer(model: nn.Module, backend: str) -> tuple[torch.optim.Optimizer, str]:
    """Create optimizer based on backend + config."""
    requested = str(TRAINING_CONFIG.get("optimizer", "adam")).strip().lower()
    if backend == "dml":
        requested = str(TRAINING_CONFIG.get("dml_optimizer", requested)).strip().lower()

    if requested == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(TRAINING_CONFIG.get("sgd_learning_rate", 1e-2)),
            momentum=float(TRAINING_CONFIG.get("sgd_momentum", 0.9)),
            weight_decay=float(TRAINING_CONFIG["weight_decay"]),
            nesterov=bool(TRAINING_CONFIG.get("sgd_nesterov", True)),
        )
        return optimizer, "sgd"

    if requested == "adagrad":
        optimizer = torch.optim.Adagrad(
            model.parameters(),
            lr=float(TRAINING_CONFIG["learning_rate"]),
            weight_decay=float(TRAINING_CONFIG["weight_decay"]),
        )
        return optimizer, "adagrad"

    if requested == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(TRAINING_CONFIG["learning_rate"]),
            weight_decay=float(TRAINING_CONFIG["weight_decay"]),
            foreach=False,
        )
        return optimizer, "adamw"

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(TRAINING_CONFIG["learning_rate"]),
        weight_decay=float(TRAINING_CONFIG["weight_decay"]),
        foreach=False,
    )
    return optimizer, "adam"


def prepare_dataloaders(use_gpu: bool) -> tuple[DataLoader, DataLoader, dict[str, int], torch.Tensor]:
    """Load split CSVs, build vocab, create dataloaders, and class weights."""
    train_df = read_split(Path(TRAINING_CONFIG["train_file"]))
    test_df = read_split(Path(TRAINING_CONFIG["test_file"]))

    role_to_id = role_to_id_map()
    unknown_train = sorted(set(train_df["label"]) - set(role_to_id))
    unknown_test = sorted(set(test_df["label"]) - set(role_to_id))
    if unknown_train or unknown_test:
        raise ValueError(f"Unknown labels found. train={unknown_train}, test={unknown_test}")

    vocab = build_vocab(train_df["text"].tolist())
    max_length = int(TEXT_CONFIG["max_length"])
    batch_size = int(TRAINING_CONFIG["batch_size"])
    num_workers = int(TRAINING_CONFIG.get("num_workers", 0))
    pin_memory = bool(use_gpu)

    train_labels = [role_to_id[label] for label in train_df["label"].tolist()]
    test_labels = [role_to_id[label] for label in test_df["label"].tolist()]

    train_dataset = RoleTextDataset(
        texts=train_df["text"].tolist(),
        labels=train_labels,
        vocab=vocab,
        max_length=max_length,
    )
    test_dataset = RoleTextDataset(
        texts=test_df["text"].tolist(),
        labels=test_labels,
        vocab=vocab,
        max_length=max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    label_counts = Counter(train_labels)
    total = float(len(train_labels))
    num_classes = float(len(ROLE_LABELS))
    class_weights = []
    for class_id in range(len(ROLE_LABELS)):
        count = float(label_counts.get(class_id, 1))
        class_weights.append(total / (num_classes * count))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    return train_loader, test_loader, vocab, class_weights_tensor


def save_checkpoint(model: nn.Module, vocab: dict[str, int]) -> Path:
    """Save model weights + preprocessing metadata in one artifact."""
    out_path = Path(ARTIFACT_CONFIG["checkpoint_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "role_labels": ROLE_LABELS,
        "text_config": TEXT_CONFIG,
        "model_config": MODEL_CONFIG,
    }
    torch.save(payload, out_path)
    return out_path


def main() -> None:
    """Train a CNN classifier and save checkpoint artifact."""
    parser = argparse.ArgumentParser(description="Train TextCNN role classifier")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "dml", "cpu"],
        default=str(TRAINING_CONFIG.get("device", "auto")),
    )
    parser.add_argument(
        "--dml-device-index",
        type=int,
        default=int(TRAINING_CONFIG.get("dml_device_index", 0)),
        help="DirectML adapter index (use only with --device dml/auto)",
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision when running on CUDA")
    args = parser.parse_args()

    set_seed(int(TRAINING_CONFIG["seed"]))
    device, backend, adapter_name = select_device(args.device, args.dml_device_index)
    use_cuda = backend == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True
    train_loader, test_loader, vocab, class_weights = prepare_dataloaders(use_gpu=use_cuda)
    use_amp = bool(TRAINING_CONFIG.get("use_amp", True)) and use_cuda and (not args.no_amp)
    scaler = create_grad_scaler(use_amp=use_amp)

    model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=int(MODEL_CONFIG["embedding_dim"]),
        num_classes=len(ROLE_LABELS),
        num_filters=int(MODEL_CONFIG["num_filters"]),
        kernel_sizes=tuple(MODEL_CONFIG["kernel_sizes"]),
        dropout=float(MODEL_CONFIG["dropout"]),
        pad_idx=vocab[TEXT_CONFIG["pad_token"]],
    ).to(device)

    optimizer, optimizer_name = create_optimizer(model=model, backend=backend)
    print(
        f"Using device: {device} ({backend}) | Adapter: {adapter_name} | "
        f"AMP: {use_amp} | Optimizer: {optimizer_name}"
    )
    if bool(TRAINING_CONFIG.get("use_class_weights", False)):
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(TRAINING_CONFIG["lr_scheduler_factor"]),
        patience=int(TRAINING_CONFIG["lr_scheduler_patience"]),
        min_lr=float(TRAINING_CONFIG["lr_scheduler_min_lr"]),
    )
    grad_clip_norm = float(TRAINING_CONFIG["grad_clip_norm"])

    epochs = int(TRAINING_CONFIG["epochs"])
    monitor_metric = str(TRAINING_CONFIG.get("monitor_metric", "accuracy")).strip().lower()
    if monitor_metric not in {"accuracy", "f1"}:
        monitor_metric = "accuracy"
    early_stopping_patience = int(TRAINING_CONFIG.get("early_stopping_patience", 0))
    no_improve_epochs = 0
    best_metric = -1.0
    best_state_dict = None
    for epoch in range(1, epochs + 1):
        avg_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip_norm=grad_clip_norm,
        )
        metrics = evaluate(model, test_loader, device)
        current_metric = float(metrics[monitor_metric])
        scheduler.step(current_metric)
        current_lr = optimizer.param_groups[0]["lr"]
        if current_metric > best_metric:
            best_metric = current_metric
            best_state_dict = deepcopy(model.state_dict())
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        print(
            f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | "
            f"Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | LR: {current_lr:.6f}"
        )
        if early_stopping_patience > 0 and no_improve_epochs >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch} (no {monitor_metric} improvement).")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    final_metrics = evaluate(model, test_loader, device)
    out_path = save_checkpoint(model, vocab)
    print("\nFinal test metrics:")
    print(f"Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall:    {final_metrics['recall']:.4f}")
    print(f"F1-Score:  {final_metrics['f1']:.4f}")
    print(f"Saved checkpoint: {out_path}")


if __name__ == "__main__":
    main()

