"""Split a merged labeled dataset into train/test CSV files.

Edit SPLIT_CONFIG below to tweak split behavior.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_DIR


SPLIT_CONFIG = {
    "input_file": DATA_DIR / "output" / "merged_labeled_dataset.csv",
    "output_dir": DATA_DIR / "split",
    "train_file": "train.csv",
    "test_file": "test.csv",
    "test_size": 0.2,
    "random_state": 42,
    "shuffle": True,
    "stratify": True,
}


def validate_dataset(df: pd.DataFrame, input_file: Path) -> pd.DataFrame:
    """Validate required columns and normalize text/label values."""
    required_columns = {"text", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Dataset {input_file} must contain columns {sorted(required_columns)}. "
            f"Found: {list(df.columns)}"
        )

    cleaned = pd.DataFrame(
        {
            "text": df["text"].astype(str).fillna(""),
            "label": df["label"].astype(str).str.strip().str.lower().fillna(""),
        }
    )

    cleaned = cleaned[cleaned["text"].str.len() > 0].copy()
    if cleaned.empty:
        raise ValueError("No non-empty text rows found after cleaning.")

    return cleaned


def split_dataset() -> None:
    """Split configured dataset and save train/test CSV files."""
    input_file = Path(SPLIT_CONFIG["input_file"])
    output_dir = Path(SPLIT_CONFIG["output_dir"])
    train_path = output_dir / str(SPLIT_CONFIG["train_file"])
    test_path = output_dir / str(SPLIT_CONFIG["test_file"])

    if not input_file.exists():
        raise ValueError(f"Input dataset not found: {input_file}")

    print(f"Reading merged dataset: {input_file}")
    df = pd.read_csv(input_file, dtype=str, keep_default_na=False)
    df = validate_dataset(df, input_file)

    stratify_values = df["label"] if SPLIT_CONFIG["stratify"] else None

    print("Splitting dataset...")
    train_df, test_df = train_test_split(
        df,
        test_size=SPLIT_CONFIG["test_size"],
        random_state=SPLIT_CONFIG["random_state"],
        shuffle=SPLIT_CONFIG["shuffle"],
        stratify=stratify_values,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Total rows: {len(df)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")


def main() -> None:
    """CLI entrypoint."""
    split_dataset()


if __name__ == "__main__":
    main()