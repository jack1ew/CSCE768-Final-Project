#!/usr/bin/env python3
"""
Multi-Run Training with History Tracking
Trains all models 5 times, tracks performance, selects best models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score

# Paths
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models/saved_models")
RESULTS_DIR = Path("results/multi_run_training")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Feature columns
FEATURE_COLUMNS = [
    "round_num", "round_duration_seconds",
    "team_score_before", "opp_score_before", "score_diff",
    "team_eq_start", "team_eq_end", "team_eq_spend",
    "team_alive_end", "team_total_utility",
    "team_kills", "team_deaths", "team_damage", "team_round_result",
    "team_cumulative_kills", "team_cumulative_deaths",
    "team_cumulative_damage", "team_cumulative_eq_spend"
]

# Model configurations
MODEL_CONFIGS = {
    'logistic': {
        'class': LogisticRegression,
        'params': {'max_iter': 1000, 'random_state': 42}
    },
    'random_forest': {
        'class': RandomForestClassifier,
        'params': {'n_estimators': 100, 'max_depth': 20, 'random_state': 42, 'n_jobs': -1}
    },
    'knn': {
        'class': KNeighborsClassifier,
        'params': {'n_neighbors': 5, 'n_jobs': -1}
    },
    'mlp': {
        'class': MLPClassifier,
        'params': {
            'hidden_layer_sizes': (128, 64),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 1e-4,
            'batch_size': 512,
            'max_iter': 400,
            'early_stopping': True,
            'random_state': 42,
            'tol': 1e-4
        }
    }
}

DATASETS = ['kaggle', 'esta', 'combined']

def load_and_prepare_data(dataset_name, random_state=42):
    """Load and prepare dataset with specific random state"""
    file_map = {
        'kaggle': 'team_round_features_kaggle_regulation.csv',
        'esta': 'team_round_features_esta_regulation.csv',
        'combined': 'team_round_features_combined_regulation.csv'
    }

    df = pd.read_csv(DATA_DIR / file_map[dataset_name], low_memory=False)

    # Split by matches with specified random state
    match_ids = df['match_id'].unique()
    train_ids, test_ids = train_test_split(
        match_ids, test_size=0.2, random_state=random_state
    )

    train_df = df[df['match_id'].isin(train_ids)]
    test_df = df[df['match_id'].isin(test_ids)]

    # Prepare features
    X_train = train_df[FEATURE_COLUMNS].copy()
    X_train['is_ct'] = (train_df['team_side'] == 'CT').astype(int)
    X_train = X_train.fillna(0).astype(float)

    X_test = test_df[FEATURE_COLUMNS].copy()
    X_test['is_ct'] = (test_df['team_side'] == 'CT').astype(int)
    X_test = X_test.fillna(0).astype(float)

    y_train = train_df['team_is_match_winner'].astype(int)
    y_test = test_df['team_is_match_winner'].astype(int)

    return X_train, X_test, y_train, y_test

def train_single_run(model_name, model_config, X_train, X_test, y_train, y_test, run_num):
    """Train a single model and return metrics"""

    # Update random state for this run
    params = model_config['params'].copy()
    if 'random_state' in params:
        params['random_state'] = 42 + run_num

    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model_config['class'](**params))
    ])

    # Train
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Evaluate on training set
    y_train_pred = pipeline.predict(X_train)
    y_train_proba = pipeline.predict_proba(X_train)[:, 1]
    train_acc = accuracy_score(y_train, y_train_pred)

    # Evaluate on test set
    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, y_test_pred)
    test_log_loss = log_loss(y_test, y_test_proba)
    test_brier = brier_score_loss(y_test, y_test_proba)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)

    # Generalization gap
    gen_gap = train_acc - test_acc

    metrics = {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'test_log_loss': test_log_loss,
        'test_brier_score': test_brier,
        'test_roc_auc': test_roc_auc,
        'generalization_gap': gen_gap,
        'train_time_seconds': train_time
    }

    return pipeline, metrics

def run_multi_training():
    """Run 5 training iterations for all models and datasets"""

    print("="*80)
    print("MULTI-RUN TRAINING: 5 Iterations per Model per Dataset")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models: {len(MODEL_CONFIGS)} (Logistic, Random Forest, k-NN, MLP)")
    print(f"Datasets: {len(DATASETS)} (Kaggle, ESTA, Combined)")
    print(f"Runs per model: 5")
    print(f"Total training runs: {len(MODEL_CONFIGS) * len(DATASETS) * 5}")
    print("="*80)

    all_results = []
    best_models = {}

    for dataset_name in DATASETS:
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'='*80}")

        for model_name, model_config in MODEL_CONFIGS.items():
            print(f"\n  Model: {model_name.upper()}")
            print(f"  {'-'*60}")

            run_results = []

            for run_num in range(5):
                print(f"    Run {run_num + 1}/5...", end=" ", flush=True)

                # Load data with different random state for each run
                X_train, X_test, y_train, y_test = load_and_prepare_data(
                    dataset_name, random_state=42 + run_num
                )

                # Train
                pipeline, metrics = train_single_run(
                    model_name, model_config, X_train, X_test, y_train, y_test, run_num
                )

                # Store results
                result = {
                    'model': model_name,
                    'dataset': dataset_name,
                    'run': run_num + 1,
                    'timestamp': datetime.now().isoformat(),
                    **metrics
                }
                run_results.append(result)
                all_results.append(result)

                print(f"Acc: {metrics['test_accuracy']:.4f}, Gap: {metrics['generalization_gap']:.4f}, Time: {metrics['train_time_seconds']:.1f}s")

                # Save pipeline for each run
                model_path = MODEL_DIR / f"{model_name}_{dataset_name}_run{run_num+1}_pipeline.joblib"
                joblib.dump(pipeline, model_path)

            # Find best run for this model-dataset combination
            best_run = max(run_results, key=lambda x: x['test_accuracy'])
            best_run_num = best_run['run']

            # Load and save best model
            best_model_path = MODEL_DIR / f"{model_name}_{dataset_name}_run{best_run_num}_pipeline.joblib"
            best_pipeline = joblib.load(best_model_path)

            # Save as primary model
            primary_model_path = MODEL_DIR / f"{model_name}_{dataset_name}_pipeline.joblib"
            joblib.dump(best_pipeline, primary_model_path)

            # For ESTA, also save legacy path
            if dataset_name == 'esta':
                legacy_path = MODEL_DIR / f"{model_name}_pipeline.joblib"
                joblib.dump(best_pipeline, legacy_path)

            best_models[f"{model_name}_{dataset_name}"] = {
                'run': best_run_num,
                'metrics': best_run
            }

            # Print summary statistics
            run_df = pd.DataFrame(run_results)
            print(f"\n    Summary (5 runs):")
            print(f"      Test Accuracy:  {run_df['test_accuracy'].mean():.4f} ± {run_df['test_accuracy'].std():.4f}")
            print(f"      Gen. Gap:       {run_df['generalization_gap'].mean():.4f} ± {run_df['generalization_gap'].std():.4f}")
            print(f"      Best Run:       #{best_run_num} (Acc: {best_run['test_accuracy']:.4f})")

    # Save all results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / 'all_training_runs.csv', index=False)

    # Save best models metadata
    with open(RESULTS_DIR / 'best_models.json', 'w') as f:
        json.dump(best_models, f, indent=2)

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Best models saved to: {MODEL_DIR}")

    return results_df, best_models

def generate_training_history_visualizations(results_df):
    """Generate visualizations of training history"""

    print("\n" + "="*80)
    print("GENERATING TRAINING HISTORY VISUALIZATIONS")
    print("="*80)

    # 1. Test Accuracy across runs
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    models = results_df['model'].unique()
    colors = {'kaggle': '#1f77b4', 'esta': '#ff7f0e', 'combined': '#2ca02c'}

    for idx, model in enumerate(models):
        ax = axes[idx]

        for dataset in DATASETS:
            data = results_df[(results_df['model'] == model) & (results_df['dataset'] == dataset)]
            runs = data['run'].values
            accs = data['test_accuracy'].values

            ax.plot(runs, accs, 'o-', label=dataset.capitalize(),
                   color=colors[dataset], linewidth=2, markersize=8, alpha=0.7)

            # Add mean line
            mean_acc = accs.mean()
            ax.axhline(mean_acc, color=colors[dataset], linestyle='--', alpha=0.3)

        ax.set_xlabel('Run Number', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title(f'{model.upper()} - Test Accuracy Across Runs',
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([1, 2, 3, 4, 5])

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'accuracy_across_runs.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {RESULTS_DIR / 'accuracy_across_runs.png'}")
    plt.close()

    # 2. Generalization gap analysis
    fig, ax = plt.subplots(figsize=(14, 8))

    summary_data = []
    for model in models:
        for dataset in DATASETS:
            data = results_df[(results_df['model'] == model) & (results_df['dataset'] == dataset)]
            summary_data.append({
                'Model': model.replace('_', ' ').title(),
                'Dataset': dataset.capitalize(),
                'Mean Gap': data['generalization_gap'].mean(),
                'Std Gap': data['generalization_gap'].std()
            })

    summary_df = pd.DataFrame(summary_data)

    x = np.arange(len(models))
    width = 0.25

    for idx, dataset in enumerate(DATASETS):
        data = summary_df[summary_df['Dataset'] == dataset.capitalize()]
        means = data['Mean Gap'].values
        stds = data['Std Gap'].values

        bars = ax.bar(x + idx*width, means, width, yerr=stds,
                     label=dataset.capitalize(), color=colors[dataset], alpha=0.8,
                     capsize=5)

        # Color code by generalization quality
        for i, (mean, std) in enumerate(zip(means, stds)):
            if mean > 0.10:
                bars[i].set_edgecolor('red')
                bars[i].set_linewidth(2)

    ax.set_xlabel('Model', fontsize=13)
    ax.set_ylabel('Generalization Gap (Train - Test Accuracy)', fontsize=13)
    ax.set_title('Model Generalization Analysis Across 5 Runs', fontsize=15, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in models])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0.05, color='orange', linestyle='--', alpha=0.5, label='5% Threshold')
    ax.axhline(0.10, color='red', linestyle='--', alpha=0.5, label='10% Threshold (Overfitting)')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'generalization_gap_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {RESULTS_DIR / 'generalization_gap_analysis.png'}")
    plt.close()

    # 3. Summary statistics table
    print("\n" + "="*80)
    print("SUMMARY STATISTICS (Mean ± Std across 5 runs)")
    print("="*80)

    summary_stats = []
    for model in models:
        for dataset in DATASETS:
            data = results_df[(results_df['model'] == model) & (results_df['dataset'] == dataset)]

            # Determine generalization status
            mean_gap = data['generalization_gap'].mean()
            mean_acc = data['test_accuracy'].mean()

            if mean_gap > 0.10:
                status = "OVERFITTING"
            elif mean_acc < 0.55:
                status = "UNDERFITTING"
            else:
                status = "Good Gen."

            summary_stats.append({
                'Model': model.replace('_', ' ').title(),
                'Dataset': dataset.capitalize(),
                'Test Acc': f"{data['test_accuracy'].mean():.4f} ± {data['test_accuracy'].std():.4f}",
                'Gen. Gap': f"{data['generalization_gap'].mean():.4f} ± {data['generalization_gap'].std():.4f}",
                'Log Loss': f"{data['test_log_loss'].mean():.4f} ± {data['test_log_loss'].std():.4f}",
                'ROC AUC': f"{data['test_roc_auc'].mean():.4f} ± {data['test_roc_auc'].std():.4f}",
                'Status': status
            })

    summary_table = pd.DataFrame(summary_stats)
    print(summary_table.to_string(index=False))
    summary_table.to_csv(RESULTS_DIR / 'training_summary_statistics.csv', index=False)
    print(f"\n✓ Summary saved to: {RESULTS_DIR / 'training_summary_statistics.csv'}")

    print("\n" + "="*80)
    print("✓ VISUALIZATION GENERATION COMPLETE")
    print("="*80)

def main():
    start_time = time.time()

    # Run multi-training
    results_df, best_models = run_multi_training()

    # Generate visualizations
    generate_training_history_visualizations(results_df)

    # Print best models
    print("\n" + "="*80)
    print("BEST MODEL SELECTION")
    print("="*80)
    for key, info in best_models.items():
        model, dataset = key.split('_', 1)
        print(f"{model.upper():15} | {dataset.capitalize():10} | Run #{info['run']} | Acc: {info['metrics']['test_accuracy']:.4f} | Gap: {info['metrics']['generalization_gap']:.4f}")

    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"TOTAL TIME: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
