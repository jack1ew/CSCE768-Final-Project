#!/usr/bin/env python3
"""Shared helpers for the CSCE768 training scripts."""

from pathlib import Path
from typing import Any, Callable, Tuple

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATASET_PATHS = {
    "esta": Path("data/processed/team_round_features_esta_regulation.csv"),
    "kaggle": Path("data/processed/team_round_features_kaggle_regulation.csv"),
    "combined": Path("data/processed/team_round_features_combined_regulation.csv"),
}
RESULTS_DIR = Path("results")
MODEL_DIR = Path("models/saved_models")

FEATURE_COLUMNS = [
    "round_num",
    "round_duration_seconds",
    "team_score_before",
    "opp_score_before",
    "score_diff",
    "team_eq_start",
    "team_eq_end",
    "team_eq_spend",
    "team_alive_end",
    "team_total_utility",
    "team_kills",
    "team_deaths",
    "team_damage",
    "team_round_result",
    "team_cumulative_kills",
    "team_cumulative_deaths",
    "team_cumulative_damage",
    "team_cumulative_eq_spend",
]
TARGET_COLUMN = "team_is_match_winner"


def _resolve_dataset_path(dataset: str) -> Path:
    if dataset in DATASET_PATHS:
        return DATASET_PATHS[dataset]
    return Path(dataset)


def load_processed_dataframe(dataset: str = "esta") -> pd.DataFrame:
    path = _resolve_dataset_path(dataset)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset '{dataset}' not found at {path}. Run the appropriate preprocessing script first."
        )
    df = pd.read_csv(path)
    if "dataset" not in df.columns:
        df["dataset"] = dataset
    else:
        df["dataset"] = df["dataset"].fillna(dataset).replace({0: dataset})
    return df


def split_matches(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    match_ids = df["match_id"].dropna().unique()
    train_ids, test_ids = train_test_split(match_ids, test_size=test_size, random_state=random_state)
    train_df = df[df["match_id"].isin(train_ids)].copy()
    test_df = df[df["match_id"].isin(test_ids)].copy()
    return train_df, test_df


def prepare_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    features = df[FEATURE_COLUMNS].copy()
    features["is_ct"] = (df["team_side"] == "CT").astype(int)
    features = features.fillna(0).astype(float)
    target = df[TARGET_COLUMN].astype(int)
    return features, target


def _evaluate_predictions(y_true, y_pred, y_proba) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "log_loss": log_loss(y_true, y_proba),
        "brier_score": brier_score_loss(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }
    return metrics


def _save_metrics(metrics: dict, model_name: str, dataset: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics["model"] = model_name
    metrics["dataset"] = dataset
    results_path = RESULTS_DIR / f"{model_name}_metrics.csv"
    frame = pd.DataFrame([metrics])
    if results_path.exists():
        existing = pd.read_csv(results_path)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(results_path, index=False)


def _save_pipeline(pipeline: Pipeline, model_name: str, dataset: str) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = MODEL_DIR / f"{model_name}_{dataset}_pipeline.joblib"
    joblib.dump(pipeline, artifact_path)
    print(f"Saved pipeline to {artifact_path}")
    if dataset == "esta":
        legacy_path = MODEL_DIR / f"{model_name}_pipeline.joblib"
        joblib.dump(pipeline, legacy_path)


def train_and_evaluate(
    model_name: str, model_factory: Callable[[], Any], dataset: str = "esta"
) -> None:
    df = load_processed_dataframe(dataset)
    train_df, test_df = split_matches(df)
    X_train, y_train = prepare_feature_matrix(train_df)
    X_test, y_test = prepare_feature_matrix(test_df)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", model_factory()),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = _evaluate_predictions(y_test, y_pred, y_proba)
    metrics.update(
        {
            "train_matches": train_df["match_id"].nunique(),
            "test_matches": test_df["match_id"].nunique(),
            "train_samples": len(train_df),
            "test_samples": len(test_df),
        }
    )

    _save_pipeline(pipeline, model_name, dataset)
    _save_metrics(metrics, model_name, dataset)

    print(
        f"[{model_name}] accuracy={metrics['accuracy']:.3f}, log_loss={metrics['log_loss']:.3f}, "
        f"brier={metrics['brier_score']:.3f}, roc_auc={metrics['roc_auc']:.3f}"
    )
    print(
        f"    train matches={metrics['train_matches']}, test matches={metrics['test_matches']}"
    )


if __name__ == "__main__":
    raise SystemExit("train_utils is a helper module and should not be run directly.")
