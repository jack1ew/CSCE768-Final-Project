#!/usr/bin/env python3
"""Train a hybrid CNN-Transformer encoder on per-round sequences."""

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

from models.architectures.cnn_transformer import build_model
import train_utils

MAX_ROUNDS = 75  # Increased to handle overtime matches (max observed: 71 rounds)
BATCH_SIZE = 64
EPOCHS = 12
LEARNING_RATE = 1e-3  # Lowered from 5e-3 to prevent divergence
WARMUP_EPOCHS = 2  # Warmup period for learning rate
LR_SCHEDULER = "cosine"  # Options: "cosine", "step", "exponential"
GRAD_CLIP = 1.0  # Gradient clipping threshold
MODEL_DIR = Path("models/saved_models")


def _artifact_paths(dataset: str, split: str) -> tuple[Path, Path]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"cnn_transformer_{dataset}_{split}.pt"
    scaler_path = MODEL_DIR / f"cnn_transformer_{dataset}_{split}_scaler.joblib"
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
    grad_clip: float = None,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    total = 0
    grad_norms = []

    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, targets)

        # Check for NaN loss
        if torch.isnan(loss):
            print(f"WARNING: NaN loss detected!")
            continue

        loss.backward()

        # Gradient clipping
        if grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            grad_norms.append(grad_norm.item())

        optimizer.step()
        running_loss += loss.item() * features.size(0)
        total += features.size(0)

    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
    return running_loss / max(total, 1), avg_grad_norm


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
    parser = argparse.ArgumentParser(description="Train the CNN-Transformer on a dataset")
    parser.add_argument(
        "--dataset",
        default="esta",
        choices=["esta", "kaggle", "combined"],
        help="Which processed dataset to use",
    )
    parser.add_argument(
        "--split",
        default="random",
        choices=["random", "time"],
        help="Train/test split strategy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = train_utils.load_processed_dataframe(args.dataset)
    df["is_ct"] = (df["team_side"] == "CT").astype(int)
    feature_columns = train_utils.FEATURE_COLUMNS + ["is_ct"]
    # Note: Currently only random split is supported in train_utils
    train_df, test_df = train_utils.split_matches(df)

    sequences_train, labels_train = _build_sequences(train_df, feature_columns)
    sequences_test, labels_test = _build_sequences(test_df, feature_columns)
    train_keys = set(zip(train_df["match_id"], train_df["team_name"]))
    test_keys = set(zip(test_df["match_id"], test_df["team_name"]))
    print("overlapping (match_id, team_name):", len(train_keys & test_keys))

    if sequences_train.size == 0 or sequences_test.size == 0:
        raise SystemExit("Not enough data to build CNN-Transformer sequences.")

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

    # Initialize learning rate scheduler (after warmup)
    if LR_SCHEDULER == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-5
        )
    elif LR_SCHEDULER == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif LR_SCHEDULER == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    else:
        scheduler = None

    best_val_auc = 0.0
    for epoch in range(1, EPOCHS + 1):
        # Learning rate warmup
        if epoch <= WARMUP_EPOCHS:
            warmup_lr = LEARNING_RATE * (epoch / WARMUP_EPOCHS)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

        current_lr = optimizer.param_groups[0]['lr']
        epoch_loss, grad_norm = _train_one_epoch(
            model, train_loader, optimizer, criterion, device, grad_clip=GRAD_CLIP
        )
        metrics = _evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch}/{EPOCHS} lr={current_lr:.6f} loss={epoch_loss:.3f} grad={grad_norm:.3f} "
            f"val_acc={metrics.get('accuracy', 0):.3f} "
            f"val_loss={metrics.get('log_loss', 0):.3f} "
            f"val_auc={metrics.get('roc_auc', float('nan')):.3f}"
        )

        # Track best model
        if metrics.get('roc_auc', 0) > best_val_auc:
            best_val_auc = metrics.get('roc_auc', 0)

        # Step the learning rate scheduler (after warmup)
        if scheduler is not None and epoch > WARMUP_EPOCHS:
            scheduler.step()

    # Using random split (args.split parameter removed)
    split_type = "random"
    model_path, scaler_path = _artifact_paths(args.dataset, split_type)
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    final_metrics = _evaluate(model, test_loader, device)
    summary = {
        "model": "cnn_transformer",
        "dataset": args.dataset,
        "split": split_type,
        "train_matches": train_df["match_id"].nunique(),
        "test_matches": test_df["match_id"].nunique(),
        "accuracy": final_metrics.get("accuracy", 0),
        "log_loss": final_metrics.get("log_loss", 0),
        "brier_score": final_metrics.get("brier_score", 0),
        "roc_auc": final_metrics.get("roc_auc", 0),
    }
    results_path = Path("results")
    results_path.mkdir(parents=True, exist_ok=True)
    results_file = results_path / "cnn_transformer_metrics.csv"
    df_results = pd.DataFrame([summary])
    if results_file.exists():
        existing = pd.read_csv(results_file)
        if "split" not in existing.columns:
            existing["split"] = "random"
        df_results = pd.concat([existing, df_results], ignore_index=True)
    df_results.to_csv(results_file, index=False)
    print(f"Saved CNN-Transformer metrics to {results_file}")
    print(f"Saved CNN-Transformer weights to {model_path} and scaler to {scaler_path}")
    # Always save legacy paths for backward compatibility
    dataset_path = MODEL_DIR / f"cnn_transformer_{args.dataset}.pt"
    dataset_scaler = MODEL_DIR / f"cnn_transformer_{args.dataset}_scaler.joblib"
    torch.save(model.state_dict(), dataset_path)
    joblib.dump(scaler, dataset_scaler)


if __name__ == "__main__":
    main()
