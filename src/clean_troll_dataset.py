"""Clean and format a troll dataset for role classification.

This script:
1) Reads a CSV dataset
2) Keeps only rows where any toxicity attribute is 1:
   toxic, severe_toxic, obscene, threat, insult, identity_hate
3) Keeps only comment_text and renames it to text
4) Removes HTML tags/components from text
5) Adds label column set to troll
6) Saves only text and label columns
"""

import argparse
import html
import re
from pathlib import Path

import pandas as pd


HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
TOXICITY_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


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

    print("Cleaning troll text (0%)")
    for idx, value in enumerate(texts, start=1):
        cleaned.append(strip_html(str(value)))
        if idx % step == 0 or idx == total:
            progress = int((idx / total) * 100)
            print(f"Cleaning troll text ({progress}%)")

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


def filter_troll_rows(df: pd.DataFrame, toxicity_cols: list[str]) -> pd.DataFrame:
    """Keep rows where at least one toxicity attribute equals 1."""
    numeric_flags = df[toxicity_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    keep_mask = numeric_flags.eq(1).any(axis=1)
    return df[keep_mask].copy()


def clean_troll_dataset(input_csv: Path, output_csv: Path, row_limit: int = 3000) -> None:
    """Process troll dataset according to project requirements."""
    print("Starting troll dataset cleaning pipeline...")
    df = read_csv_with_fallback(input_csv)

    print("Finding required columns...")
    comment_col = find_column(df.columns, "comment_text")
    matched_toxicity_cols = [find_column(df.columns, col) for col in TOXICITY_COLUMNS]

    print("Filtering rows where any toxicity flag is 1...")
    filtered_df = filter_troll_rows(df, matched_toxicity_cols)

    print(f"Selecting first {row_limit} filtered rows...")
    filtered_df = filtered_df.head(row_limit).copy()

    print("Cleaning comment_text column...")
    cleaned_text = clean_texts_with_progress(filtered_df[comment_col].fillna(""))

    print("Building output dataframe...")
    output_df = pd.DataFrame(
        {
            "text": cleaned_text,
            "label": "troll",
        }
    )

    print("Saving cleaned dataset...")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Filtered troll rows (after limit): {len(filtered_df)}")
    print(f"Output rows: {len(output_df)}")
    print(f"Saved cleaned dataset to: {output_csv}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Clean and format troll dataset")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV containing comment_text and toxicity columns",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output cleaned CSV",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3000,
        help="Maximum number of filtered rows to keep (default: 3000)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    clean_troll_dataset(Path(args.input), Path(args.output), args.limit)


if __name__ == "__main__":
    main()
