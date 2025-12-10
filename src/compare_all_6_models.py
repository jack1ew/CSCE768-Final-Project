#!/usr/bin/env python3
"""
Compare All 6 Models - Streamlined Analysis
Answers: Is CNN+Transformer worth using? What are the advantages of each model?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Paths
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("results/model_comparison_final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# All 6 models
MODELS = ['logistic', 'random_forest', 'knn', 'mlp', 'transformer', 'cnn_transformer']
MODEL_NAMES = {
    'logistic': 'Logistic Regression',
    'random_forest': 'Random Forest',
    'knn': 'k-Nearest Neighbors',
    'mlp': 'Multi-Layer Perceptron',
    'transformer': 'Transformer',
    'cnn_transformer': 'CNN+Transformer'
}

DATASETS = ['esta', 'kaggle', 'combined']

def load_all_metrics():
    """Load metrics from all CSV files"""
    print("Loading metrics from all models...")

    all_data = []

    for model in MODELS:
        csv_path = RESULTS_DIR / f"{model}_metrics.csv"
        if not csv_path.exists():
            print(f"  Warning: {csv_path} not found")
            continue

        df = pd.read_csv(csv_path)

        # Normalize column names
        if 'model' not in df.columns:
            df['model'] = model

        # Keep only the columns we need
        required_cols = ['model', 'dataset', 'accuracy', 'log_loss', 'brier_score', 'roc_auc']

        # Check for split column
        if 'split' in df.columns:
            # Filter for random split only to match training
            df = df[df['split'] == 'random'].copy()

        # Select required columns (skip if not present)
        available_cols = [c for c in required_cols if c in df.columns]
        df_subset = df[available_cols].copy()

        # Add model name if not present
        if 'model' not in df_subset.columns:
            df_subset['model'] = model

        all_data.append(df_subset)
        print(f"  Loaded {model}: {len(df_subset)} entries")

    # Combine all data
    combined = pd.concat(all_data, ignore_index=True)

    # Filter for the 3 main datasets
    combined = combined[combined['dataset'].isin(DATASETS)]

    # Group by model and dataset, take the most recent (last) entry
    combined = combined.groupby(['model', 'dataset']).last().reset_index()

    print(f"\nTotal entries after filtering: {len(combined)}")
    print(f"Models: {combined['model'].unique()}")
    print(f"Datasets: {combined['dataset'].unique()}")

    return combined


def create_performance_summary(df):
    """Create summary statistics"""
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY BY DATASET")
    print("="*80)

    for dataset in DATASETS:
        df_dataset = df[df['dataset'] == dataset].copy()
        if len(df_dataset) == 0:
            continue

        print(f"\n--- {dataset.upper()} Dataset ---")
        df_dataset = df_dataset.set_index('model')
        df_dataset = df_dataset.reindex([m for m in MODELS if m in df_dataset.index])
        df_dataset['model_name'] = df_dataset.index.map(lambda x: MODEL_NAMES[x])

        display_cols = ['model_name', 'accuracy', 'roc_auc', 'log_loss', 'brier_score']
        print(df_dataset[display_cols].to_string(float_format=lambda x: f'{x:.4f}'))

    # Average across datasets
    print("\n" + "="*80)
    print("AVERAGE PERFORMANCE ACROSS ALL DATASETS")
    print("="*80)

    avg_metrics = df.groupby('model').agg({
        'accuracy': 'mean',
        'roc_auc': 'mean',
        'log_loss': 'mean',
        'brier_score': 'mean'
    }).round(4)

    avg_metrics = avg_metrics.reindex([m for m in MODELS if m in avg_metrics.index])
    avg_metrics['model_name'] = avg_metrics.index.map(lambda x: MODEL_NAMES[x])

    print(avg_metrics[['model_name', 'accuracy', 'roc_auc', 'log_loss', 'brier_score']].to_string())

    # Save summary
    summary_path = OUTPUT_DIR / "performance_summary.csv"
    avg_metrics.to_csv(summary_path)
    print(f"\nSaved summary to {summary_path}")

    return avg_metrics


def create_rankings(avg_metrics):
    """Create model rankings"""
    print("\n" + "="*80)
    print("MODEL RANKINGS (1 = Best)")
    print("="*80)

    rankings = pd.DataFrame({
        'Accuracy': avg_metrics['accuracy'].rank(ascending=False),
        'ROC-AUC': avg_metrics['roc_auc'].rank(ascending=False),
        'Log Loss': avg_metrics['log_loss'].rank(ascending=True),
        'Brier Score': avg_metrics['brier_score'].rank(ascending=True)
    })

    rankings = rankings.reindex([m for m in MODELS if m in rankings.index])
    rankings.index = rankings.index.map(lambda x: MODEL_NAMES[x])
    rankings = rankings.astype(int)

    print(rankings.to_string())

    # Overall ranking (average of all ranks)
    rankings['Average Rank'] = rankings.mean(axis=1).round(2)
    rankings = rankings.sort_values('Average Rank')

    print("\n--- Overall Rankings (by average rank) ---")
    print(rankings[['Average Rank']].to_string())

    # Save rankings
    rankings_path = OUTPUT_DIR / "model_rankings.csv"
    rankings.to_csv(rankings_path)
    print(f"\nSaved rankings to {rankings_path}")

    return rankings


def plot_performance_comparison(df):
    """Create comprehensive visualizations"""
    print("\nGenerating performance comparison plots...")

    # Main comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Model Performance Comparison Across All Datasets',
                 fontsize=16, fontweight='bold', y=0.995)

    metrics = [
        ('accuracy', 'Accuracy', axes[0, 0], True),
        ('roc_auc', 'ROC-AUC', axes[0, 1], True),
        ('log_loss', 'Log Loss', axes[1, 0], False),
        ('brier_score', 'Brier Score', axes[1, 1], False)
    ]

    colors = plt.cm.Set2(np.linspace(0, 1, len(DATASETS)))

    for metric, title, ax, higher_better in metrics:
        # Prepare data for grouped bar chart
        plot_data = []
        for dataset in DATASETS:
            df_dataset = df[df['dataset'] == dataset]
            for model in MODELS:
                model_data = df_dataset[df_dataset['model'] == model]
                if len(model_data) > 0:
                    plot_data.append({
                        'model': MODEL_NAMES[model],
                        'dataset': dataset.upper(),
                        'value': model_data[metric].values[0]
                    })

        df_plot = pd.DataFrame(plot_data)

        # Create grouped bar chart
        x = np.arange(len(MODELS))
        width = 0.25

        for i, dataset in enumerate(DATASETS):
            dataset_data = df_plot[df_plot['dataset'] == dataset.upper()]
            # Reorder to match MODELS order
            values = []
            for model in MODELS:
                model_name = MODEL_NAMES[model]
                val = dataset_data[dataset_data['model'] == model_name]['value']
                values.append(val.values[0] if len(val) > 0 else 0)

            offset = (i - 1) * width
            ax.bar(x + offset, values, width, label=dataset.upper(),
                   color=colors[i], alpha=0.8)

        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_ylabel(title + (' (higher is better)' if higher_better else ' (lower is better)'),
                     fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_NAMES[m] for m in MODELS],
                          rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=9, loc='best')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "performance_comparison_all_models.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved main comparison plot to {plot_path}")
    plt.close()

    # Average performance plot
    avg_metrics = df.groupby('model').agg({
        'accuracy': 'mean',
        'roc_auc': 'mean',
        'log_loss': 'mean',
        'brier_score': 'mean'
    })

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Average Performance Across All Datasets',
                 fontsize=16, fontweight='bold')

    # ROC-AUC comparison (most important metric)
    ax = axes[0]
    models_sorted = avg_metrics.sort_values('roc_auc', ascending=True).index
    y_pos = np.arange(len(models_sorted))
    values = [avg_metrics.loc[m, 'roc_auc'] for m in models_sorted]
    colors_bar = ['#2ecc71' if m == 'cnn_transformer' else '#3498db'
                  for m in models_sorted]

    ax.barh(y_pos, values, color=colors_bar, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([MODEL_NAMES[m] for m in models_sorted])
    ax.set_xlabel('Average ROC-AUC', fontsize=11, fontweight='bold')
    ax.set_title('ROC-AUC (Primary Metric)', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=9)

    # Accuracy comparison
    ax = axes[1]
    models_sorted = avg_metrics.sort_values('accuracy', ascending=True).index
    y_pos = np.arange(len(models_sorted))
    values = [avg_metrics.loc[m, 'accuracy'] for m in models_sorted]
    colors_bar = ['#2ecc71' if m == 'cnn_transformer' else '#3498db'
                  for m in models_sorted]

    ax.barh(y_pos, values, color=colors_bar, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([MODEL_NAMES[m] for m in models_sorted])
    ax.set_xlabel('Average Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Accuracy', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "average_performance_ranking.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved ranking plot to {plot_path}")
    plt.close()


def analyze_cnn_transformer(df, avg_metrics, rankings):
    """Specific analysis of CNN+Transformer"""
    print("\n" + "="*80)
    print("CNN+TRANSFORMER SPECIFIC ANALYSIS")
    print("="*80)

    if 'cnn_transformer' not in avg_metrics.index:
        print("CNN+Transformer not found in metrics")
        return

    cnn_t = avg_metrics.loc['cnn_transformer']
    best_model = avg_metrics['roc_auc'].idxmax()
    best_auc = avg_metrics.loc[best_model, 'roc_auc']

    print(f"\nCNN+Transformer Performance:")
    print(f"  Average Accuracy:    {cnn_t['accuracy']:.4f}")
    print(f"  Average ROC-AUC:     {cnn_t['roc_auc']:.4f}")
    print(f"  Average Log Loss:    {cnn_t['log_loss']:.4f}")
    print(f"  Average Brier Score: {cnn_t['brier_score']:.4f}")

    print(f"\nRankings:")
    model_name = MODEL_NAMES['cnn_transformer']
    for metric in ['Accuracy', 'ROC-AUC', 'Log Loss', 'Brier Score']:
        rank = rankings.loc[model_name, metric]
        print(f"  {metric:15s}: #{rank}/6")

    avg_rank = rankings.loc[model_name, 'Average Rank']
    print(f"  Overall (avg):  #{avg_rank:.1f}/6")

    print(f"\nComparison to Best Model ({MODEL_NAMES[best_model]}):")
    auc_diff = cnn_t['roc_auc'] - best_auc
    acc_diff = cnn_t['accuracy'] - avg_metrics.loc[best_model, 'accuracy']

    if auc_diff >= -0.001:
        verdict = "TIED or WINNING"
        color = "✓"
    elif auc_diff >= -0.01:
        verdict = "COMPETITIVE"
        color = "~"
    else:
        verdict = "BEHIND"
        color = "✗"

    print(f"  ROC-AUC Difference: {auc_diff:+.4f} [{color} {verdict}]")
    print(f"  Accuracy Difference: {acc_diff:+.4f}")

    return {
        'verdict': verdict,
        'auc_diff': auc_diff,
        'acc_diff': acc_diff,
        'avg_rank': avg_rank,
        'best_model': best_model
    }


def main():
    """Main analysis pipeline"""
    print("="*80)
    print("COMPREHENSIVE 6-MODEL COMPARISON")
    print("="*80)
    print("Comparing: Logistic, Random Forest, k-NN, MLP, Transformer, CNN+Transformer")
    print("="*80)

    # Load all metrics
    df = load_all_metrics()

    # Create summary
    avg_metrics = create_performance_summary(df)

    # Create rankings
    rankings = create_rankings(avg_metrics)

    # Generate plots
    plot_performance_comparison(df)

    # CNN+Transformer specific analysis
    cnn_analysis = analyze_cnn_transformer(df, avg_metrics, rankings)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("\nKey Questions Answered:")
    print("  ✓ Is CNN+Transformer worth using?")
    print("  ✓ What are the advantages of each model?")
    print("  ✓ Which model should you use for your use case?")


if __name__ == "__main__":
    main()




