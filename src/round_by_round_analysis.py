#!/usr/bin/env python3
"""
Round-by-Round Win Probability Analysis
Analyzes how prediction accuracy evolves from round 1 to round 30
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_processed_data(dataset_name):
    """Load preprocessed data for a given dataset"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train_utils import load_processed_dataframe, split_matches, prepare_feature_matrix

    # Load and split data using the same logic as training scripts
    df = load_processed_dataframe(dataset_name)

    # Split by match_id
    train_df, test_df = split_matches(df)

    # Prepare features
    X_train, y_train = prepare_feature_matrix(train_df)
    X_test, y_test = prepare_feature_matrix(test_df)

    return X_train, X_test, y_train, y_test

def load_model(model_name, dataset_name):
    """Load a trained model"""
    import joblib
    import torch
    import torch.nn as nn

    model_dir = Path("models/saved_models")

    # Map analysis script model names to saved model file names
    model_file_mapping = {
        "logistic_regression": "logistic",
        "random_forest": "random_forest",
        "knn": "knn",
        "mlp": "mlp",
        "transformer": "transformer"
    }

    file_prefix = model_file_mapping.get(model_name, model_name)

    # Handle transformer models separately (PyTorch format)
    if model_name == "transformer":
        from train_transformer import TransformerClassifier, FEATURE_COLS

        model_path = model_dir / f"{file_prefix}_{dataset_name}.pt"
        scaler_path = model_dir / f"{file_prefix}_{dataset_name}_scaler.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")

        # Load scaler
        scaler = joblib.load(scaler_path)

        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_nn = TransformerClassifier(input_dim=len(FEATURE_COLS), d_model=128, nhead=4, num_layers=2)
        model_nn.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model_nn.to(device)
        model_nn.eval()

        # Create a wrapper that mimics sklearn interface
        class TransformerWrapper:
            def __init__(self, model, scaler, device, feature_cols):
                self.model = model
                self.scaler = scaler
                self.device = device
                self.feature_cols = feature_cols

            def predict_proba(self, X):
                # Keep only feature columns used during training
                X_features = X[self.feature_cols].copy()
                X_scaled = self.scaler.transform(X_features)
                X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    logits = self.model(X_tensor)
                    probs = torch.sigmoid(logits).cpu().numpy()

                # Return [P(class=0), P(class=1)]
                return np.column_stack([1 - probs, probs])

            def predict(self, X):
                probs = self.predict_proba(X)
                return (probs[:, 1] >= 0.5).astype(int)

        return TransformerWrapper(model_nn, scaler, device, FEATURE_COLS)

    else:
        # Handle sklearn models (joblib format)
        model_path = model_dir / f"{file_prefix}_{dataset_name}_pipeline.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = joblib.load(model_path)
        return model

def analyze_round_by_round(X_test, y_test, model, model_name, dataset_name):
    """
    Analyze prediction accuracy for each round (1-30)
    Returns DataFrame with per-round metrics
    """

    # Check if round_num column exists
    if 'round_num' not in X_test.columns:
        print(f"Warning: round_num not found in {dataset_name} dataset. Skipping round-by-round analysis.")
        return None

    results = []

    # Analyze each round from 1 to 30
    for round_num in range(1, 31):
        # Filter data for this round
        round_mask = X_test['round_num'] == round_num

        if round_mask.sum() == 0:
            continue  # Skip if no data for this round

        X_round = X_test[round_mask]
        y_round = y_test[round_mask]

        # Get predictions
        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_round)[:, 1]
            else:
                y_prob = model.predict(X_round)

            y_pred = (y_prob >= 0.5).astype(int)

            # Calculate metrics
            accuracy = accuracy_score(y_round, y_pred)

            # For calibration: among predictions >50%, how often did team actually win?
            high_conf_mask = y_prob > 0.5
            if high_conf_mask.sum() > 0:
                calibration_accuracy = y_round[high_conf_mask].mean()
            else:
                calibration_accuracy = np.nan

            # Calculate other metrics if enough samples
            if len(y_round) >= 10:
                try:
                    roc_auc = roc_auc_score(y_round, y_prob)
                except:
                    roc_auc = np.nan

                try:
                    logloss = log_loss(y_round, y_prob)
                except:
                    logloss = np.nan

                try:
                    brier = brier_score_loss(y_round, y_prob)
                except:
                    brier = np.nan
            else:
                roc_auc = logloss = brier = np.nan

            # Calculate confidence (average predicted probability)
            avg_confidence = y_prob.mean()

            results.append({
                'round_num': round_num,
                'n_samples': len(y_round),
                'accuracy': accuracy,
                'calibration_accuracy': calibration_accuracy,
                'roc_auc': roc_auc,
                'log_loss': logloss,
                'brier_score': brier,
                'avg_confidence': avg_confidence,
                'model': model_name,
                'dataset': dataset_name
            })

        except Exception as e:
            print(f"Error processing round {round_num} for {model_name} on {dataset_name}: {e}")
            continue

    if not results:
        return None

    return pd.DataFrame(results)

def plot_round_by_round_accuracy(df_results, output_dir):
    """Plot accuracy progression across rounds for all models"""

    datasets = df_results['dataset'].unique()

    for dataset in datasets:
        df_dataset = df_results[df_results['dataset'] == dataset]

        plt.figure(figsize=(14, 8))

        for model in df_dataset['model'].unique():
            df_model = df_dataset[df_dataset['model'] == model]
            plt.plot(df_model['round_num'], df_model['accuracy'],
                    marker='o', label=model, linewidth=2, markersize=4)

        plt.xlabel('Round Number', fontsize=12)
        plt.ylabel('Prediction Accuracy', fontsize=12)
        plt.title(f'Round-by-Round Prediction Accuracy - {dataset.upper()} Dataset', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0.4, 1.05)

        # Add reference line at 0.5 (random chance)
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Baseline')

        plt.tight_layout()
        plt.savefig(output_dir / f'round_by_round_accuracy_{dataset}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: round_by_round_accuracy_{dataset}.png")

def plot_calibration_by_round(df_results, output_dir):
    """Plot calibration (how often >50% predictions are correct) across rounds"""

    datasets = df_results['dataset'].unique()

    for dataset in datasets:
        df_dataset = df_results[df_results['dataset'] == dataset]

        plt.figure(figsize=(14, 8))

        for model in df_dataset['model'].unique():
            df_model = df_dataset[df_dataset['model'] == model]
            # Filter out NaN values
            df_model_clean = df_model.dropna(subset=['calibration_accuracy'])
            if len(df_model_clean) > 0:
                plt.plot(df_model_clean['round_num'], df_model_clean['calibration_accuracy'],
                        marker='s', label=model, linewidth=2, markersize=4)

        plt.xlabel('Round Number', fontsize=12)
        plt.ylabel('Win Rate When Predicted >50%', fontsize=12)
        plt.title(f'Prediction Calibration by Round - {dataset.upper()} Dataset', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0.4, 1.05)

        # Add reference line at 0.5
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Perfect Calibration at 50%')

        plt.tight_layout()
        plt.savefig(output_dir / f'calibration_by_round_{dataset}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: calibration_by_round_{dataset}.png")

def plot_confidence_evolution(df_results, output_dir):
    """Plot how model confidence evolves across rounds"""

    datasets = df_results['dataset'].unique()

    for dataset in datasets:
        df_dataset = df_results[df_results['dataset'] == dataset]

        plt.figure(figsize=(14, 8))

        for model in df_dataset['model'].unique():
            df_model = df_dataset[df_dataset['model'] == model]
            plt.plot(df_model['round_num'], df_model['avg_confidence'],
                    marker='^', label=model, linewidth=2, markersize=4)

        plt.xlabel('Round Number', fontsize=12)
        plt.ylabel('Average Predicted Probability', fontsize=12)
        plt.title(f'Model Confidence Evolution - {dataset.upper()} Dataset', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0.4, 0.9)

        plt.tight_layout()
        plt.savefig(output_dir / f'confidence_evolution_{dataset}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: confidence_evolution_{dataset}.png")

def plot_sample_distribution(df_results, output_dir):
    """Plot number of samples per round"""

    datasets = df_results['dataset'].unique()

    for dataset in datasets:
        df_dataset = df_results[df_results['dataset'] == dataset]

        # Get sample distribution from one model (they all have same distribution)
        df_samples = df_dataset[df_dataset['model'] == df_dataset['model'].iloc[0]]

        plt.figure(figsize=(14, 6))
        plt.bar(df_samples['round_num'], df_samples['n_samples'], color='steelblue', alpha=0.7)
        plt.xlabel('Round Number', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.title(f'Test Set Sample Distribution by Round - {dataset.upper()} Dataset', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_dir / f'sample_distribution_{dataset}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: sample_distribution_{dataset}.png")

def main():
    # Configuration
    datasets = ["esta", "kaggle", "combined"]
    models = ["logistic_regression", "random_forest", "knn", "mlp", "transformer"]

    output_dir = Path("results/round_by_round")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Analyze each dataset and model combination
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Analyzing {dataset.upper()} dataset...")
        print(f"{'='*60}")

        try:
            X_train, X_test, y_train, y_test = load_processed_data(dataset)
        except Exception as e:
            print(f"Error loading {dataset} data: {e}")
            continue

        for model_name in models:
            print(f"\nProcessing {model_name}...")

            try:
                model = load_model(model_name, dataset)
                df_round_results = analyze_round_by_round(X_test, y_test, model, model_name, dataset)

                if df_round_results is not None:
                    all_results.append(df_round_results)
                    print(f"  ✓ Completed {model_name} on {dataset}")
                else:
                    print(f"  ⚠ No round-by-round data for {model_name} on {dataset}")

            except FileNotFoundError as e:
                print(f"  ✗ Model not found: {e}")
            except Exception as e:
                print(f"  ✗ Error: {e}")

    # Combine all results
    if all_results:
        df_all_results = pd.concat(all_results, ignore_index=True)

        # Save results to CSV
        csv_path = output_dir / "round_by_round_metrics.csv"
        df_all_results.to_csv(csv_path, index=False)
        print(f"\n✓ Saved all results to {csv_path}")

        # Generate plots
        print("\nGenerating visualizations...")
        plot_round_by_round_accuracy(df_all_results, output_dir)
        plot_calibration_by_round(df_all_results, output_dir)
        plot_confidence_evolution(df_all_results, output_dir)
        plot_sample_distribution(df_all_results, output_dir)

        print("\n" + "="*60)
        print("Round-by-round analysis complete!")
        print(f"Results saved to: {output_dir}")
        print("="*60)
    else:
        print("\n⚠ No results generated. Check that models are trained and data is preprocessed.")

if __name__ == "__main__":
    main()
