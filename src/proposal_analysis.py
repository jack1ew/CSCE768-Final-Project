#!/usr/bin/env python3
"""
Proposal-Based Analysis: Round-by-Round Win Probability Modeling
Implements all analyses specified in the project proposal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")

# Paths
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models/saved_models")
RESULTS_DIR = Path("results/proposal_analysis")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Models as specified in proposal
MODELS = ['logistic', 'random_forest', 'knn', 'mlp', 'transformer']
MODEL_NAMES = {
    'logistic': 'Logistic Regression',
    'random_forest': 'Random Forest',
    'knn': 'k-Nearest Neighbors',
    'mlp': 'Multi-Layer Perceptron',
    'transformer': 'Transformer Encoder'
}

# Round checkpoints from proposal
CHECKPOINT_ROUNDS = [3, 5, 10, 15, 20, 25, 30]

# Datasets
DATASETS = ['esta', 'kaggle', 'combined']

def load_data(dataset_name):
    """Load regulation dataset"""
    csv_path = DATA_DIR / f"team_round_features_{dataset_name}_regulation.csv"
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    return df

def load_model(model_name, dataset_name):
    """Load trained model pipeline"""
    model_path = MODEL_DIR / f"{model_name}_{dataset_name}_pipeline.joblib"
    if not model_path.exists():
        print(f"  Warning: Model not found: {model_path}")
        return None
    return joblib.load(model_path)

def prepare_features(df):
    """Prepare feature matrix as specified in proposal"""
    # Features from proposal:
    # - Scoreline, combat aggregates, economy, objectives, survivorship

    feature_cols = [
        'round_num',
        'round_duration_seconds',
        'team_score_before',
        'opp_score_before',
        'score_diff',
        'team_eq_start',
        'team_eq_end',
        'team_eq_spend',
        'team_alive_end',
        'team_total_utility',
        'team_kills',
        'team_deaths',
        'team_damage',
        'team_round_result',
        'team_cumulative_kills',
        'team_cumulative_deaths',
        'team_cumulative_damage',
        'team_cumulative_eq_spend',
    ]

    X = df[feature_cols].copy()
    X['is_ct'] = (df['team_side'] == 'CT').astype(int)
    X = X.fillna(0).astype(float)

    y = df['team_is_match_winner'].astype(int)

    return X, y

def split_by_matches(df, test_size=0.2, random_state=42):
    """Split data by matches (no leakage)"""
    match_ids = df['match_id'].unique()
    np.random.seed(random_state)
    np.random.shuffle(match_ids)

    split_idx = int(len(match_ids) * (1 - test_size))
    train_ids = match_ids[:split_idx]
    test_ids = match_ids[split_idx:]

    train_df = df[df['match_id'].isin(train_ids)]
    test_df = df[df['match_id'].isin(test_ids)]

    return train_df, test_df

def evaluate_at_round(model, X_test, y_test, round_mask):
    """Evaluate model predictions at specific round"""
    X_round = X_test[round_mask]
    y_round = y_test[round_mask]

    if len(y_round) == 0:
        return None

    try:
        y_pred = model.predict(X_round)
        y_proba = model.predict_proba(X_round)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_round, y_pred),
            'log_loss': log_loss(y_round, y_proba),
            'brier_score': brier_score_loss(y_round, y_proba),
            'roc_auc': roc_auc_score(y_round, y_proba),
            'n_samples': len(y_round)
        }
        return metrics
    except Exception as e:
        print(f"    Error evaluating: {e}")
        return None

def compute_round_by_round_accuracy(dataset_name='kaggle'):
    """
    PRIMARY GOAL from proposal: Round-progress accuracy curves
    Compute accuracy at round checkpoints (3, 5, 10, 15, 20, 25, 30)
    """
    print(f"\n{'='*80}")
    print(f"ROUND-BY-ROUND ACCURACY CURVES: {dataset_name.upper()}")
    print(f"{'='*80}")

    # Load data
    df = load_data(dataset_name)
    train_df, test_df = split_by_matches(df)

    X_test, y_test = prepare_features(test_df)

    # Store results for all models
    results = {model: [] for model in MODELS}

    # Evaluate at each round from 1 to 30
    for round_num in range(1, 31):
        print(f"\nEvaluating at Round {round_num}...")
        round_mask = test_df['round_num'] == round_num

        for model_name in MODELS:
            model = load_model(model_name, dataset_name)
            if model is None:
                continue

            metrics = evaluate_at_round(model, X_test, y_test, round_mask)
            if metrics:
                results[model_name].append({
                    'round': round_num,
                    **metrics
                })

    # Convert to DataFrames
    results_dfs = {}
    for model_name, data in results.items():
        if data:
            results_dfs[model_name] = pd.DataFrame(data)

    return results_dfs

def plot_accuracy_curves(results_dict, dataset_name):
    """
    Plot accuracy vs round number for all models (PRIMARY OUTPUT)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Round-by-Round Performance: {dataset_name.upper()} Dataset',
                 fontsize=18, fontweight='bold')

    metrics_to_plot = [
        ('accuracy', 'Accuracy', axes[0, 0]),
        ('log_loss', 'Log Loss (lower is better)', axes[0, 1]),
        ('brier_score', 'Brier Score (lower is better)', axes[1, 0]),
        ('roc_auc', 'ROC-AUC', axes[1, 1])
    ]

    colors = plt.cm.Set2(np.linspace(0, 1, len(MODELS)))

    for metric_key, metric_label, ax in metrics_to_plot:
        for idx, (model_name, df) in enumerate(results_dict.items()):
            if df is not None and len(df) > 0:
                ax.plot(df['round'], df[metric_key],
                       marker='o', linewidth=2, markersize=4,
                       label=MODEL_NAMES[model_name], color=colors[idx],
                       alpha=0.8)

        # Highlight checkpoint rounds from proposal
        for checkpoint in CHECKPOINT_ROUNDS:
            ax.axvline(x=checkpoint, color='gray', linestyle='--', alpha=0.3)

        ax.set_xlabel('Round Number', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_label, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_label} Evolution', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, 30)

        # Add 50% baseline for accuracy
        if metric_key == 'accuracy':
            ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.5,
                      label='50% Baseline')

    plt.tight_layout()
    output_path = RESULTS_DIR / f'accuracy_curves_{dataset_name}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")

def create_checkpoint_comparison_table(results_dict, dataset_name):
    """
    Create table showing metrics at proposal checkpoints (3, 5, 10, 15, 20, 25, 30)
    """
    print(f"\n{'='*80}")
    print(f"CHECKPOINT COMPARISON: {dataset_name.upper()}")
    print(f"{'='*80}")

    table_data = []

    for model_name, df in results_dict.items():
        if df is None or len(df) == 0:
            continue

        for checkpoint in CHECKPOINT_ROUNDS:
            checkpoint_data = df[df['round'] == checkpoint]
            if len(checkpoint_data) > 0:
                row = checkpoint_data.iloc[0]
                table_data.append({
                    'Model': MODEL_NAMES[model_name],
                    'Round': checkpoint,
                    'Accuracy': f"{row['accuracy']:.4f}",
                    'Log Loss': f"{row['log_loss']:.4f}",
                    'Brier': f"{row['brier_score']:.4f}",
                    'ROC-AUC': f"{row['roc_auc']:.4f}",
                    'Samples': int(row['n_samples'])
                })

    comparison_df = pd.DataFrame(table_data)

    # Print table
    print(f"\n{'Model':<25} {'Round':<7} {'Accuracy':<10} {'Log Loss':<10} {'Brier':<10} {'ROC-AUC':<10}")
    print("-" * 80)
    for _, row in comparison_df.iterrows():
        print(f"{row['Model']:<25} {row['Round']:<7} {row['Accuracy']:<10} "
              f"{row['Log Loss']:<10} {row['Brier']:<10} {row['ROC-AUC']:<10}")

    # Save to CSV
    csv_path = RESULTS_DIR / f'checkpoint_comparison_{dataset_name}.csv'
    comparison_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    return comparison_df

def analyze_early_mid_late_performance(results_dict, dataset_name):
    """
    Analyze performance in early (1-10), mid (11-20), late (21-30) game phases
    """
    print(f"\n{'='*80}")
    print(f"EARLY/MID/LATE GAME ANALYSIS: {dataset_name.upper()}")
    print(f"{'='*80}")

    phases = {
        'Early (R1-10)': (1, 10),
        'Mid (R11-20)': (11, 20),
        'Late (R21-30)': (21, 30)
    }

    summary = []

    for model_name, df in results_dict.items():
        if df is None or len(df) == 0:
            continue

        for phase_name, (start, end) in phases.items():
            phase_data = df[(df['round'] >= start) & (df['round'] <= end)]
            if len(phase_data) > 0:
                summary.append({
                    'Model': MODEL_NAMES[model_name],
                    'Phase': phase_name,
                    'Avg Accuracy': phase_data['accuracy'].mean(),
                    'Avg Log Loss': phase_data['log_loss'].mean(),
                    'Avg Brier': phase_data['brier_score'].mean(),
                    'Improvement': phase_data['accuracy'].iloc[-1] - phase_data['accuracy'].iloc[0]
                })

    summary_df = pd.DataFrame(summary)

    print(f"\n{'Model':<25} {'Phase':<15} {'Avg Acc':<10} {'Improvement':<12}")
    print("-" * 70)
    for _, row in summary_df.iterrows():
        print(f"{row['Model']:<25} {row['Phase']:<15} {row['Avg Accuracy']:<10.4f} {row['Improvement']:<12.4f}")

    return summary_df

def plot_calibration_curves(dataset_name='kaggle'):
    """
    Calibration analysis (proposal requirement for well-calibrated probabilities)
    """
    print(f"\n{'='*80}")
    print(f"CALIBRATION ANALYSIS: {dataset_name.upper()}")
    print(f"{'='*80}")

    df = load_data(dataset_name)
    train_df, test_df = split_by_matches(df)
    X_test, y_test = prepare_features(test_df)

    # Evaluate at key checkpoints
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Calibration Curves at Key Checkpoints: {dataset_name.upper()}',
                 fontsize=18, fontweight='bold')

    axes = axes.flatten()

    for idx, checkpoint in enumerate([5, 10, 15, 20, 25, 30]):
        ax = axes[idx]
        round_mask = test_df['round_num'] == checkpoint
        X_round = X_test[round_mask]
        y_round = y_test[round_mask]

        for model_name in MODELS:
            model = load_model(model_name, dataset_name)
            if model is None:
                continue

            try:
                y_proba = model.predict_proba(X_round)[:, 1]

                # Compute calibration curve
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_round, y_proba, n_bins=10, strategy='uniform'
                )

                ax.plot(mean_predicted_value, fraction_of_positives,
                       marker='o', label=MODEL_NAMES[model_name], linewidth=2)
            except:
                continue

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)

        ax.set_xlabel('Mean Predicted Probability', fontsize=10, fontweight='bold')
        ax.set_ylabel('Fraction of Positives', fontsize=10, fontweight='bold')
        ax.set_title(f'Round {checkpoint}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()
    output_path = RESULTS_DIR / f'calibration_curves_{dataset_name}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")

def main():
    """
    Main analysis following proposal structure
    """
    print("="*80)
    print("PROPOSAL-BASED ANALYSIS")
    print("Round-by-Round Win Probability Modeling for Counter-Strike Matches")
    print("="*80)

    # Analyze each dataset
    for dataset in DATASETS:
        print(f"\n\n{'#'*80}")
        print(f"# DATASET: {dataset.upper()}")
        print(f"{'#'*80}")

        # PRIMARY GOAL: Round-by-round accuracy curves
        results = compute_round_by_round_accuracy(dataset)

        if results:
            # Plot accuracy curves (main deliverable)
            plot_accuracy_curves(results, dataset)

            # Checkpoint comparison table
            create_checkpoint_comparison_table(results, dataset)

            # Early/mid/late game analysis
            analyze_early_mid_late_performance(results, dataset)

            # Calibration analysis
            plot_calibration_curves(dataset)

    print("\n" + "="*80)
    print("PROPOSAL ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("\nGenerated outputs:")
    print("  • Round-by-round accuracy curves (PRIMARY DELIVERABLE)")
    print("  • Checkpoint comparison tables (rounds 3,5,10,15,20,25,30)")
    print("  • Early/mid/late game performance analysis")
    print("  • Calibration curves for probability quality assessment")

if __name__ == "__main__":
    main()
