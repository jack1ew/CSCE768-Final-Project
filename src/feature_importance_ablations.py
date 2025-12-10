#!/usr/bin/env python3
"""
Feature Importance & Ablation Studies (Proposal Secondary Goals)
- Random Forest feature importances
- Ablations: score alone, score+economy, complete feature set
- Sanity checks for leakage
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")

# Paths
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models/saved_models")
RESULTS_DIR = Path("results/proposal_analysis")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Feature names
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

def load_data(dataset_name):
    """Load regulation dataset"""
    csv_path = DATA_DIR / f"team_round_features_{dataset_name}_regulation.csv"
    df = pd.read_csv(csv_path, low_memory=False)
    return df

def split_by_matches(df, test_size=0.2, random_state=42):
    """Split by matches"""
    match_ids = df['match_id'].unique()
    np.random.seed(random_state)
    np.random.shuffle(match_ids)

    split_idx = int(len(match_ids) * (1 - test_size))
    train_ids = match_ids[:split_idx]
    test_ids = match_ids[split_idx:]

    train_df = df[df['match_id'].isin(train_ids)]
    test_df = df[df['match_id'].isin(test_ids)]

    return train_df, test_df

def extract_random_forest_importance(dataset_name='kaggle'):
    """
    Extract and visualize Random Forest feature importances (proposal requirement)
    """
    print(f"\n{'='*80}")
    print(f"RANDOM FOREST FEATURE IMPORTANCE: {dataset_name.upper()}")
    print(f"{'='*80}")

    # Load trained Random Forest
    model_path = MODEL_DIR / f"random_forest_{dataset_name}_pipeline.joblib"
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return None

    pipeline = joblib.load(model_path)
    rf_model = pipeline.named_steps['model']

    # Get feature importances
    importances = rf_model.feature_importances_

    # Create DataFrame
    feature_names = list(FEATURE_LABELS.keys())
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
        'label': [FEATURE_LABELS[f] for f in feature_names]
    }).sort_values('importance', ascending=False)

    # Print top features
    print(f"\nTop 10 Most Important Features:")
    print(f"{'Feature':<30} {'Importance':<12} {'%':<8}")
    print("-" * 55)
    for idx, row in importance_df.head(10).iterrows():
        print(f"{row['label']:<30} {row['importance']:<12.6f} {row['importance']*100:<7.2f}%")

    # Visualize
    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(importance_df))
    ax.barh(y_pos, importance_df['importance'], alpha=0.8, color='steelblue', edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_df['label'], fontsize=10)
    ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax.set_title(f'Random Forest Feature Importances - {dataset_name.upper()}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, val in enumerate(importance_df['importance']):
        ax.text(val + 0.005, i, f'{val:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    output_path = RESULTS_DIR / f'rf_feature_importance_{dataset_name}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")

    return importance_df

def ablation_studies(dataset_name='kaggle'):
    """
    Ablation studies (proposal requirement):
    - Score alone
    - Score + economy
    - Complete feature set
    """
    print(f"\n{'='*80}")
    print(f"ABLATION STUDIES: {dataset_name.upper()}")
    print(f"{'='*80}")

    # Load data
    df = load_data(dataset_name)
    train_df, test_df = split_by_matches(df)

    # Define feature sets
    feature_sets = {
        'Score Only': ['score_diff', 'team_score_before', 'opp_score_before'],
        'Score + Economy': [
            'score_diff', 'team_score_before', 'opp_score_before',
            'team_eq_start', 'team_eq_end', 'team_eq_spend', 'team_cumulative_eq_spend'
        ],
        'Score + Combat': [
            'score_diff', 'team_score_before', 'opp_score_before',
            'team_kills', 'team_deaths', 'team_cumulative_kills', 'team_cumulative_deaths'
        ],
        'Complete Feature Set': [
            'round_num', 'round_duration_seconds',
            'team_score_before', 'opp_score_before', 'score_diff',
            'team_eq_start', 'team_eq_end', 'team_eq_spend',
            'team_alive_end', 'team_total_utility',
            'team_kills', 'team_deaths', 'team_damage', 'team_round_result',
            'team_cumulative_kills', 'team_cumulative_deaths',
            'team_cumulative_damage', 'team_cumulative_eq_spend'
        ]
    }

    # Results storage
    results = []

    # Test at different round checkpoints
    checkpoints = [5, 10, 15, 20, 30]

    for feature_set_name, feature_list in feature_sets.items():
        print(f"\n{feature_set_name}:")
        print(f"  Features: {len(feature_list)}")

        for checkpoint in checkpoints:
            # Filter to checkpoint
            train_checkpoint = train_df[train_df['round_num'] <= checkpoint]
            test_checkpoint = test_df[test_df['round_num'] == checkpoint]

            if len(test_checkpoint) == 0:
                continue

            # Prepare features
            X_train = train_checkpoint[feature_list].copy()
            X_train['is_ct'] = (train_checkpoint['team_side'] == 'CT').astype(int)
            X_train = X_train.fillna(0).astype(float)

            X_test = test_checkpoint[feature_list].copy()
            X_test['is_ct'] = (test_checkpoint['team_side'] == 'CT').astype(int)
            X_test = X_test.fillna(0).astype(float)

            y_train = train_checkpoint['team_is_match_winner'].astype(int)
            y_test = test_checkpoint['team_is_match_winner'].astype(int)

            # Train simple logistic regression
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(max_iter=1000, random_state=42))
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            logloss = log_loss(y_test, y_proba)

            results.append({
                'Feature Set': feature_set_name,
                'Round': checkpoint,
                'Accuracy': accuracy,
                'Log Loss': logloss,
                'Num Features': len(feature_list) + 1  # +1 for is_ct
            })

            print(f"  Round {checkpoint}: Accuracy = {accuracy:.4f}, Log Loss = {logloss:.4f}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Ablation Studies: {dataset_name.upper()}',
                 fontsize=16, fontweight='bold')

    # Plot 1: Accuracy by round for each feature set
    ax = axes[0]
    for feature_set_name in feature_sets.keys():
        subset = results_df[results_df['Feature Set'] == feature_set_name]
        ax.plot(subset['Round'], subset['Accuracy'],
               marker='o', linewidth=2, markersize=8, label=feature_set_name)

    ax.set_xlabel('Round Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Evolution by Feature Set', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label='50% Baseline')

    # Plot 2: Performance gain from feature additions
    ax = axes[1]
    final_round_data = results_df[results_df['Round'] == 30]
    feature_set_order = ['Score Only', 'Score + Economy', 'Score + Combat', 'Complete Feature Set']
    final_round_data = final_round_data.set_index('Feature Set').reindex(feature_set_order).reset_index()

    x_pos = np.arange(len(final_round_data))
    bars = ax.bar(x_pos, final_round_data['Accuracy'], alpha=0.8,
                  color=plt.cm.viridis(np.linspace(0.2, 0.9, len(final_round_data))),
                  edgecolor='black')

    ax.set_ylabel('Accuracy at Round 30', fontsize=12, fontweight='bold')
    ax.set_title('Feature Set Comparison (Final Round)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(final_round_data['Feature Set'], rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (acc, n_feat) in enumerate(zip(final_round_data['Accuracy'], final_round_data['Num Features'])):
        ax.text(i, acc + 0.01, f'{acc:.3f}\n({n_feat} features)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    output_path = RESULTS_DIR / f'ablation_studies_{dataset_name}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")

    # Save results table
    csv_path = RESULTS_DIR / f'ablation_results_{dataset_name}.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    return results_df

def sanity_check_for_leakage(dataset_name='kaggle'):
    """
    Sanity check for data leakage (proposal requirement)
    Early plateaus or near-perfect accuracy would indicate leakage
    """
    print(f"\n{'='*80}")
    print(f"SANITY CHECK FOR DATA LEAKAGE: {dataset_name.upper()}")
    print(f"{'='*80}")

    df = load_data(dataset_name)
    train_df, test_df = split_by_matches(df)

    # Check accuracy at round 1 (should be close to 50% if no leakage)
    round1_test = test_df[test_df['round_num'] == 1]

    # Use all features (same as model training), excluding 'is_ct' which is derived
    features = [f for f in FEATURE_LABELS.keys() if f != 'is_ct']

    X_test = round1_test[features].copy()
    X_test['is_ct'] = (round1_test['team_side'] == 'CT').astype(int)
    X_test = X_test.fillna(0).astype(float)

    y_test = round1_test['team_is_match_winner'].astype(int)

    # Load Random Forest model
    model_path = MODEL_DIR / f"random_forest_{dataset_name}_pipeline.joblib"
    if model_path.exists():
        model = joblib.load(model_path)

        # Evaluate on round 1
        y_pred = model.predict(X_test)
        round1_accuracy = accuracy_score(y_test, y_pred)

        print(f"\nRound 1 Accuracy: {round1_accuracy:.4f}")

        if round1_accuracy > 0.60:
            print("⚠️  WARNING: Round 1 accuracy > 60% - potential leakage detected")
            print("   Investigation recommended:")
            print("   - Check if pre-match features are leaking match outcome")
            print("   - Verify train/test split is by match, not by round")
        elif round1_accuracy < 0.45:
            print("⚠️  WARNING: Round 1 accuracy < 45% - possible label issues")
        else:
            print("✓  PASS: Round 1 accuracy is reasonable (45-60% range)")
            print("   No obvious signs of data leakage")

        # Check if accuracy plateaus early
        accuracies_by_round = []
        for round_num in range(1, 31):
            round_test = test_df[test_df['round_num'] == round_num]
            if len(round_test) > 0:
                X_round = round_test[features].copy()
                X_round['is_ct'] = (round_test['team_side'] == 'CT').astype(int)
                X_round = X_round.fillna(0).astype(float)
                y_round = round_test['team_is_match_winner'].astype(int)

                y_pred = model.predict(X_round)
                acc = accuracy_score(y_round, y_pred)
                accuracies_by_round.append({'round': round_num, 'accuracy': acc})

        acc_df = pd.DataFrame(accuracies_by_round)

        # Check for early plateau (accuracy stops improving after round 5)
        early_acc = acc_df[acc_df['round'] <= 5]['accuracy'].mean()
        late_acc = acc_df[acc_df['round'] >= 25]['accuracy'].mean()
        improvement = late_acc - early_acc

        print(f"\nAccuracy Progression:")
        print(f"  Early (Round 1-5):  {early_acc:.4f}")
        print(f"  Late (Round 25-30): {late_acc:.4f}")
        print(f"  Improvement:        {improvement:.4f} ({improvement*100:.1f} percentage points)")

        if improvement < 0.05:
            print("⚠️  WARNING: Accuracy plateaus very early - check for leakage")
        else:
            print("✓  PASS: Accuracy improves steadily over rounds")

def main():
    """Run all feature importance and ablation analyses"""
    print("="*80)
    print("FEATURE IMPORTANCE & ABLATION STUDIES (Proposal Secondary Goals)")
    print("="*80)

    for dataset in ['esta', 'kaggle', 'combined']:
        print(f"\n\n{'#'*80}")
        print(f"# DATASET: {dataset.upper()}")
        print(f"{'#'*80}")

        # Feature importance (Random Forest)
        extract_random_forest_importance(dataset)

        # Ablation studies
        ablation_studies(dataset)

        # Sanity check
        sanity_check_for_leakage(dataset)

    print("\n" + "="*80)
    print("FEATURE IMPORTANCE & ABLATION ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
