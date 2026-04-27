"""Run full dataset prep pipeline with one shared row limit.

This script runs, in order:
1) answers dataset cleaning
2) questions dataset cleaning
3) troll dataset cleaning
4) labeled datasets merge
"""

import argparse
from pathlib import Path

from clean_answers_dataset import clean_answers_dataset
from clean_questions_dataset import clean_questions_dataset
from clean_troll_dataset import clean_troll_dataset
from merge_labeled_datasets import merge_labeled_files


def positive_int(value: str) -> int:
    """Argparse type that only accepts positive integers."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--limit must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Run dataset cleaning + merge pipeline with one row limit"
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        required=True,
        help="Maximum rows per cleaner output (required)",
    )
    parser.add_argument(
        "--answers-input",
        default=str(project_root / "data" / "archive" / "Answers.csv"),
        help="Path to raw answers CSV",
    )
    parser.add_argument(
        "--questions-input",
        default=str(project_root / "data" / "archive" / "Questions.csv"),
        help="Path to raw questions CSV",
    )
    parser.add_argument(
        "--troll-input",
        default=str(
            project_root
            / "data"
            / "jigsaw-toxic-comment-classification-challenge"
            / "train.csv"
        ),
        help="Path to raw troll CSV",
    )
    parser.add_argument(
        "--answers-output",
        default=str(project_root / "data" / "cleaned_answers_3k.csv"),
        help="Path to cleaned answers output CSV",
    )
    parser.add_argument(
        "--questions-output",
        default=str(project_root / "data" / "cleaned_questions_3k.csv"),
        help="Path to cleaned questions output CSV",
    )
    parser.add_argument(
        "--troll-output",
        default=str(project_root / "data" / "cleaned_troll_dataset_3k.csv"),
        help="Path to cleaned troll output CSV",
    )
    parser.add_argument(
        "--merge-input-dir",
        default=str(project_root / "data"),
        help="Folder containing labeled CSV files for merging",
    )
    parser.add_argument(
        "--merge-output",
        default=str(project_root / "data" / "output" / "merged_labeled_dataset.csv"),
        help="Output path for merged labeled CSV",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for merged row shuffle",
    )
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full cleaning and merging pipeline."""
    answers_input = Path(args.answers_input)
    questions_input = Path(args.questions_input)
    troll_input = Path(args.troll_input)

    answers_output = Path(args.answers_output)
    questions_output = Path(args.questions_output)
    troll_output = Path(args.troll_output)

    merge_input_dir = Path(args.merge_input_dir)
    merge_output = Path(args.merge_output)

    print("Running answers cleaner...")
    clean_answers_dataset(answers_input, answers_output, row_limit=args.limit)

    print("Running questions cleaner...")
    clean_questions_dataset(questions_input, questions_output, row_limit=args.limit)

    print("Running troll cleaner...")
    clean_troll_dataset(troll_input, troll_output, row_limit=args.limit)

    print("Running labeled merge...")
    merge_labeled_files(merge_input_dir, merge_output, seed=args.seed)

    print("Pipeline complete.")


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()