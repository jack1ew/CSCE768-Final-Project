#!/usr/bin/env python3
"""
Feature Importance Analysis Across All Datasets
Compare which statistics matter most for predicting wins/losses
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

sns.set_style("whitegrid")

# Paths
MODEL_DIR = Path("models/saved_models")
RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR / "feature_importance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Feature names in readable format
FEATURE_NAMES = [
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
    'is_ct'
]

FEATURE_LABELS = {
    'round_num': 'Round Number',
    'round_duration_seconds': 'Round Duration',
    'team_score_before': 'Team Score',
    'opp_score_before': 'Opponent Score',
    'score_diff': 'Score Differential',
    'team_eq_start': 'Starting Equipment',
    'team_eq_end': 'Ending Equipment',
    'team_eq_spend': 'Equipment Spending',
    'team_alive_end': 'Players Alive',
    'team_total_utility': 'Total Utility',
    'team_kills': 'Round Kills',
    'team_deaths': 'Round Deaths',
    'team_damage': 'Round Damage',
    'team_round_result': 'Round Result',
    'team_cumulative_kills': 'Total Kills',
    'team_cumulative_deaths': 'Total Deaths',
    'team_cumulative_damage': 'Total Damage',
    'team_cumulative_eq_spend': 'Total Spending',
    'is_ct': 'CT Side'
}

def get_feature_importance(dataset_name):
    """Extract feature importance from Random Forest model"""
    model_path = MODEL_DIR / f"random_forest_{dataset_name}_pipeline.joblib"

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return None

    pipeline = joblib.load(model_path)
    model = pipeline.named_steps['model']

    importances = model.feature_importances_

    return pd.DataFrame({
        'feature': FEATURE_NAMES,
        'importance': importances,
        'dataset': dataset_name
    })

def create_comparison_plot():
    """Create side-by-side comparison of feature importance across datasets"""
    print("Extracting feature importance from all datasets...")

    # Collect importance data
    all_importance = []
    for dataset in ['esta', 'kaggle', 'combined']:
        df = get_feature_importance(dataset)
        if df is not None:
            all_importance.append(df)

    if not all_importance:
        print("ERROR: No feature importance data found!")
        return

    importance_df = pd.concat(all_importance, ignore_index=True)

    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 10), sharey=True)
    fig.suptitle('Feature Importance Comparison Across Datasets',
                 fontsize=18, fontweight='bold', y=0.98)

    datasets = ['esta', 'kaggle', 'combined']
    dataset_titles = ['ESTA Dataset', 'Kaggle Dataset', 'Combined Dataset']
    colors = ['#2E86AB', '#A23B72', '#F18F01']

    for idx, (dataset, title, color) in enumerate(zip(datasets, dataset_titles, colors)):
        ax = axes[idx]

        # Filter data for this dataset
        data = importance_df[importance_df['dataset'] == dataset].copy()

        # Sort by importance
        data = data.sort_values('importance', ascending=True)

        # Take top 15 features
        data = data.tail(15)

        # Create readable labels
        data['label'] = data['feature'].map(FEATURE_LABELS)

        # Create horizontal bar plot
        y_pos = np.arange(len(data))
        ax.barh(y_pos, data['importance'], color=color, alpha=0.8, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(data['label'], fontsize=10)
        ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, val in enumerate(data['importance']):
            ax.text(val + 0.005, i, f'{val:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'feature_importance_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    return importance_df

def create_top10_comparison():
    """Create focused comparison of top 10 features"""
    print("\nCreating top 10 feature comparison...")

    # Collect data
    all_data = []
    for dataset in ['esta', 'kaggle', 'combined']:
        df = get_feature_importance(dataset)
        if df is not None:
            # Get top 10
            df_top = df.nlargest(10, 'importance')
            all_data.append(df_top)

    if not all_data:
        return

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(16, 10))

    # Get union of top features across all datasets
    all_features = set()
    for df in all_data:
        all_features.update(df['feature'].tolist())

    # Sort features by average importance
    feature_avg = {}
    for feature in all_features:
        importances = []
        for df in all_data:
            imp = df[df['feature'] == feature]['importance']
            if len(imp) > 0:
                importances.append(imp.iloc[0])
            else:
                importances.append(0)
        feature_avg[feature] = np.mean(importances)

    sorted_features = sorted(feature_avg.keys(), key=lambda x: feature_avg[x], reverse=True)[:12]

    # Prepare data for grouped bars
    esta_vals = []
    kaggle_vals = []
    combined_vals = []

    for feature in sorted_features:
        # ESTA
        val = all_data[0][all_data[0]['feature'] == feature]['importance']
        esta_vals.append(val.iloc[0] if len(val) > 0 else 0)

        # Kaggle
        val = all_data[1][all_data[1]['feature'] == feature]['importance']
        kaggle_vals.append(val.iloc[0] if len(val) > 0 else 0)

        # Combined
        val = all_data[2][all_data[2]['feature'] == feature]['importance']
        combined_vals.append(val.iloc[0] if len(val) > 0 else 0)

    # Create grouped bars
    x = np.arange(len(sorted_features))
    width = 0.25

    ax.bar(x - width, esta_vals, width, label='ESTA', alpha=0.8, color='#2E86AB', edgecolor='black')
    ax.bar(x, kaggle_vals, width, label='Kaggle', alpha=0.8, color='#A23B72', edgecolor='black')
    ax.bar(x + width, combined_vals, width, label='Combined', alpha=0.8, color='#F18F01', edgecolor='black')

    ax.set_ylabel('Feature Importance', fontsize=14, fontweight='bold')
    ax.set_title('Top Features Comparison Across Datasets', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABELS[f] for f in sorted_features],
                        rotation=45, ha='right', fontsize=11)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'top_features_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def create_category_analysis():
    """Analyze importance by feature category"""
    print("\nCreating category analysis...")

    # Define feature categories
    categories = {
        'Economic': ['team_eq_start', 'team_eq_end', 'team_eq_spend', 'team_cumulative_eq_spend'],
        'Score': ['team_score_before', 'opp_score_before', 'score_diff'],
        'Performance': ['team_kills', 'team_deaths', 'team_damage', 'team_round_result'],
        'Cumulative': ['team_cumulative_kills', 'team_cumulative_deaths', 'team_cumulative_damage'],
        'Positional': ['team_alive_end', 'team_total_utility', 'is_ct'],
        'Temporal': ['round_num', 'round_duration_seconds']
    }

    # Collect data by category
    category_importance = {dataset: {cat: 0 for cat in categories.keys()}
                          for dataset in ['esta', 'kaggle', 'combined']}

    for dataset in ['esta', 'kaggle', 'combined']:
        df = get_feature_importance(dataset)
        if df is None:
            continue

        for category, features in categories.items():
            total_importance = df[df['feature'].isin(features)]['importance'].sum()
            category_importance[dataset][category] = total_importance

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(14, 8))

    datasets = ['esta', 'kaggle', 'combined']
    dataset_labels = ['ESTA', 'Kaggle', 'Combined']

    # Prepare data
    categories_list = list(categories.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories_list)))

    # Create stacked bars
    bottom = np.zeros(3)
    for idx, category in enumerate(categories_list):
        values = [category_importance[ds][category] for ds in datasets]
        ax.bar(dataset_labels, values, bottom=bottom, label=category,
               color=colors[idx], alpha=0.8, edgecolor='black')

        # Add value labels
        for i, val in enumerate(values):
            if val > 0.05:  # Only show if significant
                ax.text(i, bottom[i] + val/2, f'{val:.2f}',
                       ha='center', va='center', fontweight='bold', fontsize=10)

        bottom += values

    ax.set_ylabel('Total Feature Importance', fontsize=14, fontweight='bold')
    ax.set_title('Feature Importance by Category Across Datasets',
                 fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'category_importance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def print_summary_table():
    """Print summary table of top features"""
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE SUMMARY")
    print("="*80)

    for dataset in ['esta', 'kaggle', 'combined']:
        print(f"\n{dataset.upper()} Dataset - Top 10 Features:")
        print("-" * 60)

        df = get_feature_importance(dataset)
        if df is None:
            continue

        df_sorted = df.nlargest(10, 'importance')

        for idx, row in df_sorted.iterrows():
            feature_label = FEATURE_LABELS[row['feature']]
            print(f"  {feature_label:30} {row['importance']:6.4f} ({row['importance']*100:5.1f}%)")

def main():
    """Generate all feature importance visualizations"""
    print("="*80)
    print("FEATURE IMPORTANCE ANALYSIS - ALL DATASETS")
    print("="*80)

    # Create visualizations
    create_comparison_plot()
    create_top10_comparison()
    create_category_analysis()

    # Print summary
    print_summary_table()

    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  • feature_importance_comparison.png  (side-by-side comparison)")
    print("  • top_features_comparison.png        (grouped bar chart)")
    print("  • category_importance.png            (stacked by category)")

if __name__ == "__main__":
    main()
