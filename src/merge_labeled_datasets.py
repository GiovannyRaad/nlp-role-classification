"""Merge labeled CSV files (text, label), shuffle, and save a single output file.

This script:
1) Scans a specific folder (no subfolders)
2) Reads CSV files that contain text and label columns
3) Keeps only text and label columns
4) Merges all valid files into one dataframe
5) Shuffles rows
6) Saves the result to data/output/
"""

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"text", "label"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def find_matching_column(columns: pd.Index, target: str) -> str | None:
    """Find a column by case-insensitive exact match."""
    target_lower = target.lower()
    for col in columns:
        if str(col).lower() == target_lower:
            return str(col)
    return None


def read_csv_with_fallback(file_path: Path) -> pd.DataFrame:
    """Read CSV with common encoding fallbacks."""
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            return pd.read_csv(
                file_path,
                dtype=str,
                keep_default_na=False,
                encoding=encoding,
                on_bad_lines="skip",
            )
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        "csv",
        b"",
        0,
        1,
        f"Unable to decode file {file_path}. Last error: {last_error}",
    )


def collect_labeled_files(input_dir: Path) -> list[Path]:
    """Collect only CSV files directly inside the given folder (no subfolders)."""
    return sorted([path for path in input_dir.glob("*.csv") if path.is_file()])


def merge_labeled_files(input_dir: Path, output_file: Path, seed: int = 42) -> None:
    """Merge all valid text/label CSV files from a folder into one shuffled CSV."""
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input folder does not exist or is not a directory: {input_dir}")

    csv_files = collect_labeled_files(input_dir)
    if not csv_files:
        raise ValueError(f"No CSV files found in: {input_dir}")

    print(f"Found {len(csv_files)} CSV files in {input_dir}")

    merged_frames: list[pd.DataFrame] = []
    used_files = 0

    for csv_file in csv_files:
        print(f"Reading: {csv_file.name}")
        df = read_csv_with_fallback(csv_file)

        text_col = find_matching_column(df.columns, "text")
        label_col = find_matching_column(df.columns, "label")

        if text_col is None or label_col is None:
            print(f"Skipping {csv_file.name}: missing text/label columns")
            continue

        standardized_df = pd.DataFrame(
            {
                "text": df[text_col].fillna(""),
                "label": df[label_col].fillna(""),
            }
        )

        merged_frames.append(standardized_df)
        used_files += 1

    if not merged_frames:
        raise ValueError("No valid CSV files with text and label columns were found.")

    merged_df = pd.concat(merged_frames, ignore_index=True)
    merged_df = merged_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_file, index=False)

    print(f"Used files: {used_files}")
    print(f"Merged rows: {len(merged_df)}")
    print(f"Saved merged file to: {output_file}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Merge and shuffle labeled datasets")
    parser.add_argument(
        "--input-dir",
        default=str(DATA_DIR),
        help="Folder containing CSV files to merge (no subfolders)",
    )
    parser.add_argument(
        "--output-file",
        default=str(DATA_DIR / "output" / "merged_labeled_dataset.csv"),
        help="Output CSV path. Should be inside data/output",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling rows",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    project_root = DATA_DIR.parent
    if not input_dir.is_absolute():
        input_dir = project_root / input_dir
    if not output_file.is_absolute():
        output_file = project_root / output_file
    merge_labeled_files(input_dir, output_file, args.seed)


if __name__ == "__main__":
    main()
