#!/usr/bin/env python3
"""Train a Transformer encoder on per-round sequences."""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
for candidate in (SCRIPT_DIR, ROOT_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import joblib
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.architectures.transformer import build_model
import train_utils

MAX_ROUNDS = 30
BATCH_SIZE = 64
EPOCHS = 12
LEARNING_RATE = 1e-3
MODEL_DIR = Path("models/saved_models")


def _artifact_paths(dataset: str) -> tuple[Path, Path]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"transformer_{dataset}.pt"
    scaler_path = MODEL_DIR / f"transformer_{dataset}_scaler.joblib"
    return model_path, scaler_path


def _pad_sequence(features: np.ndarray, max_len: int) -> np.ndarray:
    if features.shape[0] >= max_len:
        return features[:max_len]
    pad = np.zeros((max_len - features.shape[0], features.shape[1]), dtype=np.float32)
    return np.vstack([features, pad])


def _build_sequences(df: pd.DataFrame, feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    sequences = []
    labels = []
    grouped = df.sort_values("round_num").groupby(["match_id", "team_name"])
    for (_, _), group in grouped:
        features = group[feature_columns].to_numpy(dtype=np.float32)
        if features.size == 0:
            continue
        sequences.append(_pad_sequence(features, MAX_ROUNDS))
        labels.append(group[train_utils.TARGET_COLUMN].iloc[0])
    if not sequences:
        return np.zeros((0, MAX_ROUNDS, len(feature_columns)), dtype=np.float32), np.zeros(0, dtype=np.float32)
    return np.stack(sequences), np.array(labels, dtype=np.float32)


def _scale_sequences(arr: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    flat = arr.reshape(-1, arr.shape[-1])
    scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    scaled = (flat - scaler.mean_) / scale
    return scaled.reshape(arr.shape)


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray) -> None:
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0
    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * features.size(0)
        total += features.size(0)
    return running_loss / max(total, 1)


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            logits = model(features)
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
    if not all_logits:
        return {}
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets).numpy()
    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(float)
    metrics = {
        "accuracy": accuracy_score(targets, preds),
        "log_loss": log_loss(targets, probs),
        "brier_score": brier_score_loss(targets, probs),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(targets, probs)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the transformer on a dataset")
    parser.add_argument(
        "--dataset",
        default="esta",
        choices=["esta", "kaggle", "combined"],
        help="Which processed dataset to use",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = train_utils.load_processed_dataframe(args.dataset)
    df["is_ct"] = (df["team_side"] == "CT").astype(int)
    feature_columns = train_utils.FEATURE_COLUMNS + ["is_ct"]

    train_df, test_df = train_utils.split_matches(df)
    sequences_train, labels_train = _build_sequences(train_df, feature_columns)
    sequences_test, labels_test = _build_sequences(test_df, feature_columns)

    if sequences_train.size == 0 or sequences_test.size == 0:
        raise SystemExit("Not enough data to build transformer sequences.")

    scaler = StandardScaler()
    scaler.fit(sequences_train.reshape(-1, sequences_train.shape[-1]))
    sequences_train = _scale_sequences(sequences_train, scaler)
    sequences_test = _scale_sequences(sequences_test, scaler)

    train_dataset = SequenceDataset(sequences_train, labels_train)
    test_dataset = SequenceDataset(sequences_test, labels_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(feature_dim=sequences_train.shape[-1], seq_len=MAX_ROUNDS)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, EPOCHS + 1):
        epoch_loss = _train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = _evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch}/{EPOCHS} loss={epoch_loss:.3f} "
            f"val_acc={metrics.get('accuracy', 0):.3f} "
            f"val_loss={metrics.get('log_loss', 0):.3f} "
            f"val_auc={metrics.get('roc_auc', float('nan')):.3f}"
        )

    model_path, scaler_path = _artifact_paths(args.dataset)
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    final_metrics = _evaluate(model, test_loader, device)
    summary = {
        "model": "transformer",
        "dataset": args.dataset,
        "train_matches": train_df["match_id"].nunique(),
        "test_matches": test_df["match_id"].nunique(),
        "accuracy": final_metrics.get("accuracy", 0),
        "log_loss": final_metrics.get("log_loss", 0),
        "brier_score": final_metrics.get("brier_score", 0),
        "roc_auc": final_metrics.get("roc_auc", 0),
    }
    results_path = Path("results")
    results_path.mkdir(parents=True, exist_ok=True)
    results_file = results_path / "transformer_metrics.csv"
    df_results = pd.DataFrame([summary])
    if results_file.exists():
        existing = pd.read_csv(results_file)
        df_results = pd.concat([existing, df_results], ignore_index=True)
    df_results.to_csv(results_file, index=False)
    print(f"Saved transformer metrics to {results_file}")
    print(f"Saved transformer weights to {model_path} and scaler to {scaler_path}")
    if args.dataset == "esta":
        legacy_model = MODEL_DIR / "transformer_transformer.pt"
        legacy_scaler = MODEL_DIR / "transformer_scaler.joblib"
        torch.save(model.state_dict(), legacy_model)
        joblib.dump(scaler, legacy_scaler)


if __name__ == "__main__":
    main()
