#!/usr/bin/env python3
"""Merge ESTA and Kaggle processed datasets into a single CSV."""

from pathlib import Path

import pandas as pd

ESTA_PATH = Path("data/processed/team_round_features.csv")
KAGGLE_PATH = Path("data/processed/team_round_features_kaggle.csv")
OUTPUT_PATH = Path("data/processed/team_round_features_combined.csv")


def _load_with_dataset(path: Path, dataset_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing processed dataset: {path}")
    df = pd.read_csv(path)
    if "dataset" not in df.columns:
        df["dataset"] = dataset_name
    else:
        df["dataset"] = df["dataset"].fillna(dataset_name)
    return df


def main() -> None:
    esta_df = _load_with_dataset(ESTA_PATH, "esta")
    kaggle_df = _load_with_dataset(KAGGLE_PATH, "kaggle")
    shared_cols = sorted(set(esta_df.columns).union(kaggle_df.columns))
    esta_aligned = esta_df.reindex(columns=shared_cols)
    kaggle_aligned = kaggle_df.reindex(columns=shared_cols)
    combined = pd.concat([esta_aligned, kaggle_aligned], ignore_index=True)
    combined.fillna(0, inplace=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote combined dataset with {len(combined)} samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
