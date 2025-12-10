#!/usr/bin/env python3
"""Generate comprehensive figures for CNN-Transformer analysis."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Paths
RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "cnn_transformer_figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load CNN-Transformer metrics
cnn_df = pd.read_csv(RESULTS_DIR / "cnn_transformer_metrics.csv")

# Load pure Transformer metrics for comparison
transformer_df = pd.read_csv(RESULTS_DIR / "transformer_metrics.csv")

# Clean and prepare data
cnn_df_clean = cnn_df[(cnn_df['accuracy'] > 0.6) & (cnn_df['accuracy'] < 0.99)].copy()

print(f"Loaded {len(cnn_df)} CNN-Transformer runs")
print(f"Clean runs: {len(cnn_df_clean)}")
print(f"Failed runs: {len(cnn_df[cnn_df['accuracy'] <= 0.51])}")
print(f"Overfit runs: {len(cnn_df[cnn_df['accuracy'] >= 0.99])}")

# ============================================================================
# Figure 1: Training Stability Analysis
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Pie chart of run outcomes
outcomes = pd.Series({
    'Stable': len(cnn_df_clean),
    'Overfit (100% acc)': len(cnn_df[cnn_df['accuracy'] >= 0.99]),
    'Failed (50% acc)': len(cnn_df[cnn_df['accuracy'] <= 0.51])
})
colors = ['#2ecc71', '#e74c3c', '#f39c12']
axes[0, 0].pie(outcomes, labels=outcomes.index, autopct='%1.1f%%', colors=colors, startangle=90)
axes[0, 0].set_title('CNN-Transformer Training Stability\n(25 total runs)', fontsize=13, fontweight='bold')

# Accuracy distribution histogram
axes[0, 1].hist(cnn_df['accuracy'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 1].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Random guess')
axes[0, 1].axvline(0.85, color='green', linestyle='--', linewidth=2, label='Stable region')
axes[0, 1].set_xlabel('Test Accuracy', fontweight='bold')
axes[0, 1].set_ylabel('Number of Runs', fontweight='bold')
axes[0, 1].set_title('Accuracy Distribution Across All Runs', fontsize=13, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Log loss distribution
axes[1, 0].hist(cnn_df_clean['log_loss'], bins=20, edgecolor='black', alpha=0.7, color='coral')
axes[1, 0].set_xlabel('Log Loss', fontweight='bold')
axes[1, 0].set_ylabel('Number of Runs', fontweight='bold')
axes[1, 0].set_title('Log Loss Distribution (Stable Runs Only)', fontsize=13, fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# ROC-AUC distribution
axes[1, 1].hist(cnn_df_clean['roc_auc'], bins=20, edgecolor='black', alpha=0.7, color='mediumpurple')
axes[1, 1].set_xlabel('ROC-AUC', fontweight='bold')
axes[1, 1].set_ylabel('Number of Runs', fontweight='bold')
axes[1, 1].set_title('ROC-AUC Distribution (Stable Runs Only)', fontsize=13, fontweight='bold')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'training_stability_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Generated training_stability_analysis.png")
plt.close()

# ============================================================================
# Figure 2: Dataset Performance Comparison
# ============================================================================
# Get best runs per dataset
best_runs = []
for dataset in ['esta', 'kaggle', 'combined']:
    dataset_runs = cnn_df_clean[cnn_df_clean['dataset'] == dataset]
    if len(dataset_runs) > 0:
        best_run = dataset_runs.loc[dataset_runs['roc_auc'].idxmax()]
        best_runs.append(best_run)

best_df = pd.DataFrame(best_runs)

# Also get pure Transformer best runs for comparison
transformer_best = []
for dataset in ['esta', 'kaggle', 'combined']:
    dataset_runs = transformer_df[transformer_df['dataset'] == dataset]
    if len(dataset_runs) > 0:
        best_run = dataset_runs.loc[dataset_runs['roc_auc'].idxmax()]
        best_run['model_type'] = 'Pure Transformer'
        transformer_best.append(best_run)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

datasets = best_df['dataset'].str.upper()
x = np.arange(len(datasets))
width = 0.35

# Accuracy comparison
cnn_acc = best_df['accuracy'].values
trans_acc = [transformer_best[i]['accuracy'] for i in range(len(datasets))]
axes[0, 0].bar(x - width/2, cnn_acc, width, label='CNN-Transformer', color='#3498db', edgecolor='black')
axes[0, 0].bar(x + width/2, trans_acc, width, label='Pure Transformer', color='#95a5a6', edgecolor='black')
axes[0, 0].set_xlabel('Dataset', fontweight='bold')
axes[0, 0].set_ylabel('Accuracy', fontweight='bold')
axes[0, 0].set_title('Accuracy Comparison: CNN-Transformer vs Pure Transformer', fontsize=13, fontweight='bold')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(datasets)
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3, axis='y')
axes[0, 0].set_ylim([0.7, 1.0])

# Log loss comparison
cnn_loss = best_df['log_loss'].values
trans_loss = [transformer_best[i]['log_loss'] for i in range(len(datasets))]
axes[0, 1].bar(x - width/2, cnn_loss, width, label='CNN-Transformer', color='#e74c3c', edgecolor='black')
axes[0, 1].bar(x + width/2, trans_loss, width, label='Pure Transformer', color='#95a5a6', edgecolor='black')
axes[0, 1].set_xlabel('Dataset', fontweight='bold')
axes[0, 1].set_ylabel('Log Loss (lower is better)', fontweight='bold')
axes[0, 1].set_title('Log Loss Comparison: CNN-Transformer vs Pure Transformer', fontsize=13, fontweight='bold')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(datasets)
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3, axis='y')

# Brier score comparison
cnn_brier = best_df['brier_score'].values
trans_brier = [transformer_best[i]['brier_score'] for i in range(len(datasets))]
axes[1, 0].bar(x - width/2, cnn_brier, width, label='CNN-Transformer', color='#9b59b6', edgecolor='black')
axes[1, 0].bar(x + width/2, trans_brier, width, label='Pure Transformer', color='#95a5a6', edgecolor='black')
axes[1, 0].set_xlabel('Dataset', fontweight='bold')
axes[1, 0].set_ylabel('Brier Score (lower is better)', fontweight='bold')
axes[1, 0].set_title('Brier Score Comparison: CNN-Transformer vs Pure Transformer', fontsize=13, fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(datasets)
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3, axis='y')

# ROC-AUC comparison
cnn_auc = best_df['roc_auc'].values
trans_auc = [transformer_best[i]['roc_auc'] for i in range(len(datasets))]
axes[1, 1].bar(x - width/2, cnn_auc, width, label='CNN-Transformer', color='#2ecc71', edgecolor='black')
axes[1, 1].bar(x + width/2, trans_auc, width, label='Pure Transformer', color='#95a5a6', edgecolor='black')
axes[1, 1].set_xlabel('Dataset', fontweight='bold')
axes[1, 1].set_ylabel('ROC-AUC', fontweight='bold')
axes[1, 1].set_title('ROC-AUC Comparison: CNN-Transformer vs Pure Transformer', fontsize=13, fontweight='bold')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(datasets)
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3, axis='y')
axes[1, 1].set_ylim([0.8, 1.0])

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'dataset_performance_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Generated dataset_performance_comparison.png")
plt.close()

# ============================================================================
# Figure 3: Accuracy Delta Heatmap
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate deltas
delta_data = []
for i, dataset in enumerate(datasets):
    delta_data.append([
        (cnn_acc[i] - trans_acc[i]) * 100,  # Accuracy delta (percentage points)
        (trans_loss[i] - cnn_loss[i]) * 100,  # Log loss delta (negative means CNN worse)
        (cnn_auc[i] - trans_auc[i]) * 100   # ROC-AUC delta
    ])

delta_df = pd.DataFrame(delta_data,
                        index=datasets,
                        columns=['Accuracy Δ (%)', 'Log Loss Improvement (%)', 'ROC-AUC Δ (%)'])

sns.heatmap(delta_df, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            cbar_kws={'label': 'Performance Delta\n(positive = CNN better)'}, ax=ax)
ax.set_title('CNN-Transformer Performance Advantage Over Pure Transformer', fontsize=14, fontweight='bold')
ax.set_xlabel('')
ax.set_ylabel('Dataset', fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'performance_delta_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Generated performance_delta_heatmap.png")
plt.close()

# ============================================================================
# Figure 4: Random vs Time Split Comparison
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Filter combined dataset runs by split type
combined_random = cnn_df_clean[(cnn_df_clean['dataset'] == 'combined') & (cnn_df_clean['split'] == 'random')]
combined_time = cnn_df_clean[(cnn_df_clean['dataset'] == 'combined') & (cnn_df_clean['split'] == 'time')]

if len(combined_random) > 0 and len(combined_time) > 0:
    # Box plot comparison
    split_data = pd.concat([
        combined_random.assign(split_type='Random'),
        combined_time.assign(split_type='Temporal')
    ])

    sns.boxplot(data=split_data, x='split_type', y='accuracy', ax=axes[0], palette=['lightblue', 'lightcoral'])
    axes[0].set_xlabel('Split Strategy', fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontweight='bold')
    axes[0].set_title('Accuracy: Random vs Temporal Split\n(Combined Dataset)', fontsize=13, fontweight='bold')
    axes[0].grid(alpha=0.3, axis='y')

    # Scatter plot: accuracy vs log loss
    axes[1].scatter(combined_random['log_loss'], combined_random['accuracy'],
                   s=100, alpha=0.7, label='Random Split', color='steelblue', edgecolor='black')
    axes[1].scatter(combined_time['log_loss'], combined_time['accuracy'],
                   s=100, alpha=0.7, label='Temporal Split', color='coral', edgecolor='black', marker='s')
    axes[1].set_xlabel('Log Loss', fontweight='bold')
    axes[1].set_ylabel('Accuracy', fontweight='bold')
    axes[1].set_title('Accuracy vs Log Loss: Split Strategy Comparison', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'split_strategy_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Generated split_strategy_comparison.png")
plt.close()

# ============================================================================
# Figure 5: Architecture Comparison Summary
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

# Summary metrics
models = ['CNN-Transformer', 'Pure Transformer']
esta_acc = [best_df[best_df['dataset'] == 'esta']['accuracy'].values[0],
            transformer_best[0]['accuracy']]
kaggle_acc = [best_df[best_df['dataset'] == 'kaggle']['accuracy'].values[0],
              transformer_best[1]['accuracy']]
combined_acc = [best_df[best_df['dataset'] == 'combined']['accuracy'].values[0],
                transformer_best[2]['accuracy']]

x = np.arange(len(models))
width = 0.25

bars1 = ax.bar(x - width, esta_acc, width, label='ESTA', color='#3498db', edgecolor='black')
bars2 = ax.bar(x, kaggle_acc, width, label='Kaggle', color='#e74c3c', edgecolor='black')
bars3 = ax.bar(x + width, combined_acc, width, label='Combined', color='#2ecc71', edgecolor='black')

ax.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
ax.set_title('Model Architecture Comparison Across Datasets', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(alpha=0.3, axis='y')
ax.set_ylim([0.8, 1.0])

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'architecture_comparison_summary.png', dpi=300, bbox_inches='tight')
print("✓ Generated architecture_comparison_summary.png")
plt.close()

# ============================================================================
# Figure 6: Failure Mode Analysis
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Categorize runs
failed = cnn_df[cnn_df['accuracy'] <= 0.51]
overfit = cnn_df[cnn_df['accuracy'] >= 0.99]
stable = cnn_df_clean

categories = pd.DataFrame({
    'Category': ['Failed\n(~50% acc)', 'Stable\n(60-99% acc)', 'Overfit\n(100% acc)'],
    'Count': [len(failed), len(stable), len(overfit)],
    'Color': ['#e74c3c', '#2ecc71', '#f39c12']
})

# Bar chart of counts
axes[0].bar(categories['Category'], categories['Count'], color=categories['Color'], edgecolor='black', width=0.6)
axes[0].set_ylabel('Number of Runs', fontweight='bold')
axes[0].set_title('Training Outcome Distribution', fontsize=13, fontweight='bold')
axes[0].grid(alpha=0.3, axis='y')

# Log loss distribution by category
if len(failed) > 0:
    axes[1].hist(failed['log_loss'], bins=10, alpha=0.5, label='Failed', color='#e74c3c', edgecolor='black')
axes[1].hist(stable['log_loss'], bins=15, alpha=0.5, label='Stable', color='#2ecc71', edgecolor='black')
if len(overfit) > 0:
    axes[1].hist(overfit['log_loss'], bins=10, alpha=0.5, label='Overfit', color='#f39c12', edgecolor='black')
axes[1].set_xlabel('Log Loss', fontweight='bold')
axes[1].set_ylabel('Frequency', fontweight='bold')
axes[1].set_title('Log Loss by Training Outcome', fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_xlim([0, 1.0])

# Brier score distribution
if len(failed) > 0:
    axes[2].hist(failed['brier_score'], bins=10, alpha=0.5, label='Failed', color='#e74c3c', edgecolor='black')
axes[2].hist(stable['brier_score'], bins=15, alpha=0.5, label='Stable', color='#2ecc71', edgecolor='black')
if len(overfit) > 0:
    axes[2].hist(overfit['brier_score'], bins=10, alpha=0.5, label='Overfit', color='#f39c12', edgecolor='black')
axes[2].set_xlabel('Brier Score', fontweight='bold')
axes[2].set_ylabel('Frequency', fontweight='bold')
axes[2].set_title('Brier Score by Training Outcome', fontsize=13, fontweight='bold')
axes[2].legend()
axes[2].grid(alpha=0.3)
axes[2].set_xlim([0, 0.3])

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'failure_mode_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Generated failure_mode_analysis.png")
plt.close()

print(f"\n✓ All figures saved to {FIGURES_DIR}/")
print("\nGenerated figures:")
for fig_file in sorted(FIGURES_DIR.glob("*.png")):
    print(f"  - {fig_file.name}")
