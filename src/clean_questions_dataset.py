"""Clean and format a questions dataset for role classification.

This script:
1) Reads a CSV dataset
2) Keeps only the first 3000 rows
3) Merges Title and Body into a single text column
4) Removes HTML tags/components from the merged text
5) Discards all other columns
6) Adds a label column set to learner for all rows
"""

import argparse
import html
import re
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def strip_html(text: str) -> str:
    """Remove HTML tags and normalize whitespace."""
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def clean_texts_with_progress(texts: pd.Series) -> pd.Series:
    """Clean text values while printing progress updates."""
    total = len(texts)
    if total == 0:
        return texts

    step = max(1, total // 20)  # ~5% increments
    cleaned = []

    print("Cleaning merged text (0%)")
    for idx, value in enumerate(texts, start=1):
        cleaned.append(strip_html(str(value)))
        if idx % step == 0 or idx == total:
            progress = int((idx / total) * 100)
            print(f"Cleaning merged text ({progress}%)")

    return pd.Series(cleaned)


def find_column(columns: pd.Index, target: str) -> str:
    """Find a column name by case-insensitive exact match."""
    target_lower = target.lower()
    for col in columns:
        if str(col).lower() == target_lower:
            return str(col)
    raise ValueError(f"Required column '{target}' was not found in the input dataset.")


def read_csv_with_fallback(input_csv: Path) -> pd.DataFrame:
    """Read CSV with common encoding fallbacks."""
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            df = pd.read_csv(
                input_csv,
                dtype=str,
                keep_default_na=False,
                encoding=encoding,
                on_bad_lines="skip",
            )
            print(f"Loaded dataset using encoding: {encoding}")
            return df
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        "csv",
        b"",
        0,
        1,
        f"Unable to decode file {input_csv}. Tried encodings: {', '.join(encodings)}. Last error: {last_error}",
    )


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve CLI paths relative to the project root when needed."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def clean_questions_dataset(input_csv: Path, output_csv: Path, row_limit: int = 3000) -> None:
    """Process the dataset according to project requirements."""
    print("Starting dataset cleaning pipeline...")
    df = read_csv_with_fallback(input_csv)

    print("Finding required columns...")
    title_col = find_column(df.columns, "Title")
    body_col = find_column(df.columns, "Body")

    print(f"Selecting first {row_limit} rows...")
    df = df.head(row_limit).copy()

    print("Merging Title + Body columns...")
    merged_text = (df[title_col].fillna("") + " " + df[body_col].fillna("")).str.strip()
    cleaned_text = clean_texts_with_progress(merged_text)

    print("Building output dataframe...")
    output_df = pd.DataFrame(
        {
            "text": cleaned_text,
            "label": "learner",
        }
    )

    print("Saving cleaned dataset...")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Output rows: {len(output_df)}")
    print(f"Saved cleaned dataset to: {output_csv}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    default_input = PROJECT_ROOT / "data" / "archive" / "Questions.csv"
    default_output = PROJECT_ROOT / "data" / "cleaned_questions_3k.csv"
    parser = argparse.ArgumentParser(description="Clean and format questions dataset")
    parser.add_argument(
        "--input",
        default=str(default_input),
        help="Path to input CSV containing Title and Body columns",
    )
    parser.add_argument(
        "--output",
        default=str(default_output),
        help="Path to output cleaned CSV",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3000,
        help="Maximum number of rows to process (default: 3000)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    clean_questions_dataset(resolve_project_path(args.input), resolve_project_path(args.output), args.limit)


if __name__ == "__main__":
    main()
