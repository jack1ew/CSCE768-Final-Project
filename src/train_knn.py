#!/usr/bin/env python3
"""Train and evaluate the k-nearest neighbors baseline."""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
for candidate in (SCRIPT_DIR, ROOT_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from models.architectures.knn import build_model
import train_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train k-NN on a dataset")
    parser.add_argument(
        "--dataset",
        default="esta",
        choices=["esta", "kaggle", "combined"],
        help="Which processed dataset to use",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_utils.train_and_evaluate("knn", build_model, dataset=args.dataset)


if __name__ == "__main__":
    main()
