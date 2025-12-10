#!/usr/bin/env python3
"""
Generate Training Curves for Model Generalization Analysis
Shows overfitting, underfitting, or good generalization for all models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results/training_curves")
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

def load_and_prepare_data(dataset_name):
    """Load and prepare dataset"""
    file_map = {
        'kaggle': 'team_round_features_kaggle_regulation.csv',
        'esta': 'team_round_features_esta_regulation.csv',
        'combined': 'team_round_features_combined_regulation.csv'
    }

    df = pd.read_csv(DATA_DIR / file_map[dataset_name], low_memory=False)

    # Split by matches
    match_ids = df['match_id'].unique()
    train_ids, test_ids = train_test_split(match_ids, test_size=0.2, random_state=42)

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

def generate_mlp_training_curve(X_train, X_test, y_train, y_test, dataset_name):
    """Generate training curve for MLP (iterative model)"""
    print(f"\n{'='*60}")
    print(f"MLP Training Curve - {dataset_name.upper()}")
    print(f"{'='*60}")

    # Use smaller subset for faster training
    if len(X_train) > 100000:
        indices = np.random.choice(len(X_train), 100000, replace=False)
        X_train_sub = X_train.iloc[indices]
        y_train_sub = y_train.iloc[indices]
    else:
        X_train_sub = X_train
        y_train_sub = y_train

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sub)
    X_test_scaled = scaler.transform(X_test)

    # Train MLP with tracking
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        batch_size=512,
        max_iter=400,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=42,
        tol=1e-4,
        verbose=False
    )

    mlp.fit(X_train_scaled, y_train_sub)

    # Get training loss curve
    train_loss = mlp.loss_curve_
    val_loss = mlp.validation_scores_ if hasattr(mlp, 'validation_scores_') else None

    # Compute training and test accuracy at final epoch
    train_acc = accuracy_score(y_train_sub, mlp.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, mlp.predict(X_test_scaled))

    print(f"Final Train Accuracy: {train_acc:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Gap (Train - Test): {train_acc - test_acc:.4f}")

    if train_acc - test_acc > 0.05:
        print("⚠️  OVERFITTING: Train accuracy >> Test accuracy")
    elif test_acc < 0.55:
        print("⚠️  UNDERFITTING: Both accuracies are low")
    else:
        print("✓  GOOD GENERALIZATION")

    return {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'n_iters': len(train_loss)
    }

def generate_learning_curve(model, model_name, X_train, X_test, y_train, y_test, dataset_name):
    """Generate learning curve for non-iterative models"""
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} Learning Curve - {dataset_name.upper()}")
    print(f"{'='*60}")

    # Use smaller subset for faster computation
    if len(X_train) > 100000:
        indices = np.random.choice(len(X_train), 100000, replace=False)
        X_train_sub = X_train.iloc[indices]
        y_train_sub = y_train.iloc[indices]
    else:
        X_train_sub = X_train
        y_train_sub = y_train

    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    # Generate learning curve
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X_train_sub, y_train_sub,
        train_sizes=train_sizes,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    # Final model performance
    pipeline.fit(X_train_sub, y_train_sub)
    train_acc = accuracy_score(y_train_sub, pipeline.predict(X_train_sub))
    test_acc = accuracy_score(y_test, pipeline.predict(X_test))

    print(f"Final Train Accuracy: {train_acc:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Gap (Train - Test): {train_acc - test_acc:.4f}")

    if train_acc - test_acc > 0.10:
        print("⚠️  OVERFITTING: Train accuracy >> Test accuracy")
    elif test_acc < 0.55:
        print("⚠️  UNDERFITTING: Both accuracies are low")
    else:
        print("✓  GOOD GENERALIZATION")

    return {
        'train_sizes': train_sizes,
        'train_mean': train_mean,
        'train_std': train_std,
        'val_mean': val_mean,
        'val_std': val_std,
        'train_acc': train_acc,
        'test_acc': test_acc
    }

def plot_mlp_curves(results_dict, model_name='MLP'):
    """Plot MLP training curves for all datasets"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Training Loss
    ax = axes[0]
    colors = {'kaggle': '#1f77b4', 'esta': '#ff7f0e', 'combined': '#2ca02c'}

    for dataset, results in results_dict.items():
        epochs = range(1, results['n_iters'] + 1)
        ax.plot(epochs, results['train_loss'],
                label=f"{dataset.capitalize()} (Train)",
                color=colors[dataset], linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title(f'{model_name} - Training Loss Convergence', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Final Accuracy Comparison
    ax = axes[1]
    datasets = list(results_dict.keys())
    train_accs = [results_dict[d]['train_acc'] for d in datasets]
    test_accs = [results_dict[d]['test_acc'] for d in datasets]

    x = np.arange(len(datasets))
    width = 0.35

    bars1 = ax.bar(x - width/2, train_accs, width, label='Train Accuracy', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_accs, width, label='Test Accuracy', alpha=0.8)

    # Color code bars by generalization
    for i, (train_acc, test_acc) in enumerate(zip(train_accs, test_accs)):
        gap = train_acc - test_acc
        if gap > 0.05:  # Overfitting
            bars1[i].set_color('#d62728')
            bars2[i].set_color('#d62728')
            bars1[i].set_alpha(0.8)
            bars2[i].set_alpha(0.5)
        elif test_acc < 0.55:  # Underfitting
            bars1[i].set_color('#ff7f0e')
            bars2[i].set_color('#ff7f0e')
        else:  # Good generalization
            bars1[i].set_color('#2ca02c')
            bars2[i].set_color('#2ca02c')

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'{model_name} - Generalization Analysis', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in datasets])
    ax.legend(fontsize=10)
    ax.set_ylim([0.5, 1.0])
    ax.grid(True, alpha=0.3, axis='y')

    # Add gap annotations
    for i, (train_acc, test_acc) in enumerate(zip(train_accs, test_accs)):
        gap = train_acc - test_acc
        ax.text(i, max(train_acc, test_acc) + 0.02, f'Δ={gap:.3f}',
                ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{model_name.lower()}_training_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {RESULTS_DIR / f'{model_name.lower()}_training_curves.png'}")
    plt.close()

def plot_learning_curves(results_dict, model_name):
    """Plot learning curves for non-iterative models"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {'kaggle': '#1f77b4', 'esta': '#ff7f0e', 'combined': '#2ca02c'}

    for idx, (dataset, results) in enumerate(results_dict.items()):
        ax = axes[idx]

        # Plot training and validation curves
        ax.plot(results['train_sizes'], results['train_mean'],
                'o-', color=colors[dataset], linewidth=2, label='Train Score', alpha=0.8)
        ax.fill_between(results['train_sizes'],
                        results['train_mean'] - results['train_std'],
                        results['train_mean'] + results['train_std'],
                        alpha=0.15, color=colors[dataset])

        ax.plot(results['train_sizes'], results['val_mean'],
                's-', color='#d62728', linewidth=2, label='Val Score', alpha=0.8)
        ax.fill_between(results['train_sizes'],
                        results['val_mean'] - results['val_std'],
                        results['val_mean'] + results['val_std'],
                        alpha=0.15, color='#d62728')

        # Add final test accuracy as horizontal line
        ax.axhline(results['test_acc'], color='green', linestyle='--',
                  linewidth=1.5, label=f"Test Acc: {results['test_acc']:.3f}")

        # Determine generalization status
        gap = results['train_acc'] - results['test_acc']
        if gap > 0.10:
            status = "OVERFITTING"
            status_color = '#d62728'
        elif results['test_acc'] < 0.55:
            status = "UNDERFITTING"
            status_color = '#ff7f0e'
        else:
            status = "GOOD GEN."
            status_color = '#2ca02c'

        ax.set_xlabel('Training Set Size', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title(f'{dataset.capitalize()} - {status}',
                    fontsize=12, fontweight='bold', color=status_color)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.45, 1.0])

        # Add annotation
        ax.text(0.05, 0.95, f"Δ={gap:.3f}",
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.3))

    fig.suptitle(f'{model_name} - Learning Curves Across Datasets',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{model_name.lower().replace(" ", "_")}_training_curves.png',
                dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {RESULTS_DIR / f'{model_name.lower().replace(' ', '_')}_training_curves.png'}")
    plt.close()

def main():
    print("="*80)
    print("TRAINING CURVE GENERATION: Overfitting/Underfitting Analysis")
    print("="*80)

    datasets = ['kaggle', 'esta', 'combined']

    # Load all datasets
    data = {}
    for dataset_name in datasets:
        print(f"\nLoading {dataset_name} dataset...")
        data[dataset_name] = load_and_prepare_data(dataset_name)

    # ========== MLP (Iterative Model) ==========
    print("\n" + "="*80)
    print("MULTI-LAYER PERCEPTRON (MLP)")
    print("="*80)

    mlp_results = {}
    for dataset_name in datasets:
        X_train, X_test, y_train, y_test = data[dataset_name]
        mlp_results[dataset_name] = generate_mlp_training_curve(
            X_train, X_test, y_train, y_test, dataset_name
        )

    plot_mlp_curves(mlp_results, 'MLP')

    # ========== Logistic Regression ==========
    print("\n" + "="*80)
    print("LOGISTIC REGRESSION")
    print("="*80)

    lr_results = {}
    for dataset_name in datasets:
        X_train, X_test, y_train, y_test = data[dataset_name]
        model = LogisticRegression(max_iter=1000, random_state=42)
        lr_results[dataset_name] = generate_learning_curve(
            model, 'Logistic Regression', X_train, X_test, y_train, y_test, dataset_name
        )

    plot_learning_curves(lr_results, 'Logistic Regression')

    # ========== Random Forest ==========
    print("\n" + "="*80)
    print("RANDOM FOREST")
    print("="*80)

    rf_results = {}
    for dataset_name in datasets:
        X_train, X_test, y_train, y_test = data[dataset_name]
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf_results[dataset_name] = generate_learning_curve(
            model, 'Random Forest', X_train, X_test, y_train, y_test, dataset_name
        )

    plot_learning_curves(rf_results, 'Random Forest')

    # ========== k-NN ==========
    print("\n" + "="*80)
    print("k-NEAREST NEIGHBORS")
    print("="*80)

    knn_results = {}
    for dataset_name in datasets:
        X_train, X_test, y_train, y_test = data[dataset_name]
        model = KNeighborsClassifier(n_neighbors=5)
        knn_results[dataset_name] = generate_learning_curve(
            model, 'k-NN', X_train, X_test, y_train, y_test, dataset_name
        )

    plot_learning_curves(knn_results, 'k-NN')

    # ========== Summary Table ==========
    print("\n" + "="*80)
    print("SUMMARY: Generalization Analysis")
    print("="*80)

    summary_data = []
    for model_name, results_dict in [
        ('Logistic Regression', lr_results),
        ('Random Forest', rf_results),
        ('k-NN', knn_results),
        ('MLP', mlp_results)
    ]:
        for dataset_name in datasets:
            results = results_dict[dataset_name]
            train_acc = results['train_acc']
            test_acc = results['test_acc']
            gap = train_acc - test_acc

            if gap > 0.10:
                status = "Overfitting"
            elif test_acc < 0.55:
                status = "Underfitting"
            else:
                status = "Good Gen."

            summary_data.append({
                'Model': model_name,
                'Dataset': dataset_name.capitalize(),
                'Train Acc': f"{train_acc:.4f}",
                'Test Acc': f"{test_acc:.4f}",
                'Gap': f"{gap:.4f}",
                'Status': status
            })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # Save summary
    summary_df.to_csv(RESULTS_DIR / 'generalization_summary.csv', index=False)
    print(f"\n✓ Summary saved to: {RESULTS_DIR / 'generalization_summary.csv'}")

    print("\n" + "="*80)
    print("✓ TRAINING CURVE GENERATION COMPLETE")
    print(f"✓ Results saved to: {RESULTS_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()
