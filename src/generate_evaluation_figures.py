#!/usr/bin/env python3
"""Produce extended diagnostic figures for every model/dataset combination."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
import sys

for candidate in (SCRIPT_DIR, ROOT_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import train_utils
from models.architectures.transformer import build_model

sns.set_style("whitegrid")

RESULTS_DIR = Path("results/figures")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE_DIR = Path("models/saved_models")
MAX_ROUNDS = 30
MODELS = ["logistic", "random_forest", "knn", "mlp", "transformer"]
COLORS = {
    "logistic": "#1f77b4",
    "random_forest": "#ff7f0e",
    "knn": "#2ca02c",
    "mlp": "#d62728",
    "transformer": "#9467bd",
}


def _pad_sequence(features: np.ndarray, max_len: int) -> np.ndarray:
    if features.shape[0] >= max_len:
        return features[:max_len]
    pad = np.zeros((max_len - features.shape[0], features.shape[1]), dtype=np.float32)
    return np.vstack([features, pad])


def _build_sequences_with_keys(
    df: pd.DataFrame, feature_columns: List[str]
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    sequences: List[np.ndarray] = []
    labels: List[float] = []
    keys: List[Tuple[str, str]] = []
    grouped = df.sort_values("round_num").groupby(["match_id", "team_name"])
    for (match_id, team_name), group in grouped:
        feats = group[feature_columns].to_numpy(dtype=np.float32)
        if feats.size == 0:
            continue
        sequences.append(_pad_sequence(feats, MAX_ROUNDS))
        labels.append(group[train_utils.TARGET_COLUMN].iloc[0])
        keys.append((match_id, team_name))
    if not sequences:
        return np.zeros((0, MAX_ROUNDS, len(feature_columns)), dtype=np.float32), np.zeros(
            0, dtype=np.float32
        ), []
    return np.stack(sequences), np.array(labels, dtype=np.float32), keys


def _scale_sequences(arr: np.ndarray, scaler) -> np.ndarray:
    flat = arr.reshape(-1, arr.shape[-1])
    scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    scaled = (flat - scaler.mean_) / scale
    return scaled.reshape(arr.shape)


def _evaluate_transformer(dataset: str, test_df: pd.DataFrame, feature_columns: List[str]):
    model_path = PIPELINE_DIR / f"transformer_{dataset}.pt"
    scaler_path = PIPELINE_DIR / f"transformer_{dataset}_scaler.joblib"
    if not model_path.exists() or not scaler_path.exists():
        return None

    sequences, labels, keys = _build_sequences_with_keys(test_df, feature_columns)
    if sequences.size == 0:
        return None

    scaler = joblib.load(scaler_path)
    sequences = _scale_sequences(sequences, scaler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(feature_dim=sequences.shape[-1], seq_len=MAX_ROUNDS)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(sequences, dtype=torch.float32, device=device))
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)

    tf_df = pd.DataFrame(keys, columns=["match_id", "team_name"])
    tf_df["prob"] = probs
    tf_df["pred"] = preds
    tf_df["y_true"] = labels
    return tf_df


def _prepare_grouped_df(test_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        test_df.sort_values(["match_id", "team_name", "round_num"])
        .groupby(["match_id", "team_name"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return grouped


def collect_predictions(dataset: str) -> Dict[str, Dict[str, np.ndarray]]:
    df = train_utils.load_processed_dataframe(dataset)
    _, test_df = train_utils.split_matches(df)
    test_df = test_df.copy()
    test_df["is_ct"] = (test_df["team_side"] == "CT").astype(int)
    grouped_df = _prepare_grouped_df(test_df)
    features, target = train_utils.prepare_feature_matrix(grouped_df)
    predictions: Dict[str, Dict[str, np.ndarray]] = {}

    for model_name in ["logistic", "random_forest", "knn", "mlp"]:
        pipeline_path = PIPELINE_DIR / f"{model_name}_{dataset}_pipeline.joblib"
        if not pipeline_path.exists():
            continue
        pipeline = joblib.load(pipeline_path)
        probs = pipeline.predict_proba(features)[:, 1]
        preds = (probs >= 0.5).astype(int)
        predictions[model_name] = {
            "probs": probs,
            "preds": preds,
        }

    feature_columns = train_utils.FEATURE_COLUMNS + ["is_ct"]
    tf_df = _evaluate_transformer(dataset, test_df, feature_columns)
    if tf_df is not None:
        merged = grouped_df.merge(tf_df, on=["match_id", "team_name"], how="left")
        if merged["prob"].notna().all():
            predictions["transformer"] = {
                "probs": merged["prob"].to_numpy(),
                "preds": merged["pred"].astype(int).to_numpy(),
            }

    for model_data in predictions.values():
        model_data["y_true"] = target.to_numpy()

    return {
        "predictions": predictions,
        "grouped_df": grouped_df,
        "y_true": target.to_numpy(),
    }


def plot_roc_curves(dataset: str, predictions: Dict[str, Dict[str, np.ndarray]]) -> None:
    plt.figure(figsize=(7, 5))
    for model, data in predictions.items():
        fpr, tpr, _ = roc_curve(data["y_true"], data["probs"])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model.title()} (AUC={roc_auc:.3f})", color=COLORS.get(model))
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves ({dataset.title()})")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"roc_curves_{dataset}.png", dpi=300)
    plt.close()


def plot_precision_recall(dataset: str, predictions: Dict[str, Dict[str, np.ndarray]]):
    plt.figure(figsize=(7, 5))
    for model, data in predictions.items():
        precision, recall, _ = precision_recall_curve(data["y_true"], data["probs"])
        ap = auc(recall, precision)
        plt.plot(recall, precision, label=f"{model.title()} (AP={ap:.3f})", color=COLORS.get(model))
    baseline = data["y_true"].mean()
    plt.hlines(baseline, 0, 1, colors="gray", linestyles="dashed", label="Baseline")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curves ({dataset.title()})")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"precision_recall_{dataset}.png", dpi=300)
    plt.close()


def plot_calibration(dataset: str, predictions: Dict[str, Dict[str, np.ndarray]]):
    plt.figure(figsize=(6, 6))
    for model, data in predictions.items():
        prob_true, prob_pred = calibration_curve(
            data["y_true"], data["probs"], n_bins=15, strategy="uniform"
        )
        plt.plot(prob_pred, prob_true, marker="o", label=model.title(), color=COLORS.get(model))
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfectly Calibrated")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.title(f"Reliability Diagram ({dataset.title()})")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"calibration_{dataset}.png", dpi=300)
    plt.close()


def _compute_gain_curve(y_true: np.ndarray, probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(probs)[::-1]
    y_sorted = y_true[order]
    cum_pos = np.cumsum(y_sorted)
    total_pos = y_true.sum()
    percentages = np.arange(1, len(y_true) + 1) / len(y_true)
    gains = cum_pos / total_pos
    return percentages, gains


def plot_gain_and_lift(dataset: str, predictions: Dict[str, Dict[str, np.ndarray]]):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model, data in predictions.items():
        pct, gains = _compute_gain_curve(data["y_true"], data["probs"])
        axes[0].plot(pct, gains, label=model.title(), color=COLORS.get(model))
        lift = gains / (pct + 1e-8)
        axes[1].plot(pct, lift, label=model.title(), color=COLORS.get(model))
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axes[0].set_title("Cumulative Gain")
    axes[0].set_xlabel("Sample Fraction")
    axes[0].set_ylabel("Fraction of Positives Captured")
    axes[1].set_title("Lift Curve")
    axes[1].set_xlabel("Sample Fraction")
    axes[1].set_ylabel("Lift")
    axes[1].axhline(1.0, color="k", linestyle="--", linewidth=0.8)
    axes[1].set_ylim(0, None)
    axes[0].legend(loc="lower right")
    fig.suptitle(f"Gain & Lift Curves ({dataset.title()})", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"gain_lift_{dataset}.png", dpi=300)
    plt.close(fig)


def plot_threshold_metrics(dataset: str, predictions: Dict[str, Dict[str, np.ndarray]]):
    thresholds = np.linspace(0.1, 0.9, 9)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    for model, data in predictions.items():
        precisions = []
        recalls = []
        specificities = []
        for thresh in thresholds:
            preds = (data["probs"] >= thresh).astype(int)
            precisions.append(precision_score(data["y_true"], preds, zero_division=0))
            recalls.append(recall_score(data["y_true"], preds))
            tn = ((preds == 0) & (data["y_true"] == 0)).sum()
            fp = ((preds == 1) & (data["y_true"] == 0)).sum()
            specificity = tn / (tn + fp + 1e-8)
            specificities.append(specificity)
        axes[0].plot(thresholds, precisions, marker="o", label=model.title(), color=COLORS.get(model))
        axes[1].plot(thresholds, recalls, marker="o", label=model.title(), color=COLORS.get(model))
        axes[2].plot(
            thresholds, specificities, marker="o", label=model.title(), color=COLORS.get(model)
        )
    axes[0].set_ylabel("Precision")
    axes[1].set_ylabel("Recall (TPR)")
    axes[2].set_ylabel("Specificity (TNR)")
    for ax in axes:
        ax.set_xlabel("Decision Threshold")
        ax.set_ylim(0, 1.05)
    axes[0].legend(loc="lower left")
    fig.suptitle(f"Threshold Trade-offs ({dataset.title()})", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"threshold_tradeoffs_{dataset}.png", dpi=300)
    plt.close(fig)


def plot_confusion_matrices(dataset: str, predictions: Dict[str, Dict[str, np.ndarray]]):
    n_models = len(predictions)
    n_cols = 3
    n_rows = int(np.ceil(n_models / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    for ax in axes[n_models:]:
        ax.axis("off")
    for idx, (model, data) in enumerate(predictions.items()):
        disp = ConfusionMatrixDisplay.from_predictions(
            data["y_true"],
            data["preds"],
            ax=axes[idx],
            cmap="Blues",
            colorbar=False,
        )
        axes[idx].set_title(model.title())
    fig.suptitle(f"Confusion Matrices ({dataset.title()})", y=1.02)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"confusion_matrices_{dataset}.png", dpi=300)
    plt.close(fig)


def plot_probability_hist(dataset: str, predictions: Dict[str, Dict[str, np.ndarray]]):
    fig, axes = plt.subplots(len(predictions), 1, figsize=(6, 3 * len(predictions)), sharex=True)
    if len(predictions) == 1:
        axes = [axes]
    for ax, (model, data) in zip(axes, predictions.items()):
        sns.histplot(data["probs"][data["y_true"] == 1], color="#2ca02c", alpha=0.6, ax=ax, label="Winners")
        sns.histplot(
            data["probs"][data["y_true"] == 0],
            color="#d62728",
            alpha=0.6,
            ax=ax,
            label="Losers",
        )
        ax.set_title(f"Probability Histogram - {model.title()} ({dataset.title()})")
        ax.set_ylabel("Count")
        ax.legend()
    axes[-1].set_xlabel("Predicted Probability")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"probability_hist_{dataset}.png", dpi=300)
    plt.close(fig)


def generate_all_figures(dataset: str):
    payload = collect_predictions(dataset)
    predictions = payload["predictions"]
    if not predictions:
        print(f"No predictions found for dataset '{dataset}', skipping.")
        return

    plot_roc_curves(dataset, predictions)
    plot_precision_recall(dataset, predictions)
    plot_calibration(dataset, predictions)
    plot_gain_and_lift(dataset, predictions)
    plot_threshold_metrics(dataset, predictions)
    plot_confusion_matrices(dataset, predictions)
    plot_probability_hist(dataset, predictions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate extended evaluation figures.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["esta", "kaggle", "combined"],
        help="Datasets to visualize (choices: esta, kaggle, combined)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        print(f"Generating figures for {dataset} ...")
        generate_all_figures(dataset)


if __name__ == "__main__":
    main()
