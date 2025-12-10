#!/usr/bin/env python3
"""
Extended Analysis: Deep dive into CS:GO win prediction patterns
Answers specific research questions about:
1. First round impact
2. Comeback potential by scoreline
3. Feature importance
4. Overtime patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.inspection import permutation_importance
import joblib

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Paths
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
EXTENDED_DIR = RESULTS_DIR / "extended_analysis"
EXTENDED_DIR.mkdir(parents=True, exist_ok=True)

def analyze_pistol_round_impact(dataset_name="kaggle"):
    """Analyze the impact of winning pistol rounds (rounds 1 and 16) on match outcomes.
    
    RQ2: How impactful is winning the first round of each half on overall match outcomes?
    """
    print(f"\n{'='*60}")
    print(f"PISTOL ROUND IMPACT ANALYSIS - {dataset_name.upper()}")
    print(f"{'='*60}")

    csv_path = PROCESSED_DIR / f"team_round_features_{dataset_name}.csv"
    df = pd.read_csv(csv_path, low_memory=False)

    results = {}
    
    # Analyze both pistol rounds (round 1 = first half, round 16 = second half)
    pistol_rounds = {1: "First Half Pistol (Round 1)", 16: "Second Half Pistol (Round 16)"}
    
    for round_num, round_name in pistol_rounds.items():
        print(f"\n--- {round_name} ---")
        pistol_data = df[df['round_num'] == round_num].copy()
        
        if len(pistol_data) == 0:
            print(f"No data for round {round_num}")
            continue
        
        match_analysis = []
        for match_id in pistol_data['match_id'].unique():
            match_data = df[df['match_id'] == match_id]
            round_data = match_data[match_data['round_num'] == round_num]
            if len(round_data) == 0:
                continue
            
            pistol_winner = round_data[round_data['team_round_result'] == 1]
            if len(pistol_winner) == 0:
                continue
            
            winning_team = pistol_winner.iloc[0]['team_name']
            team_match_data = match_data[match_data['team_name'] == winning_team]
            if len(team_match_data) == 0:
                continue
            
            match_winner = team_match_data.iloc[0]['team_is_match_winner']
            match_analysis.append({'match_id': match_id, 'pistol_round': round_num,
                                   'won_pistol': 1, 'won_match': match_winner})
        
        if len(match_analysis) > 0:
            results_df = pd.DataFrame(match_analysis)
            total_matches = len(results_df)
            pistol_winners_won = results_df['won_match'].sum()
            win_rate = pistol_winners_won / total_matches
            
            results[round_num] = {'name': round_name, 'total_matches': total_matches,
                                  'pistol_winners_won_match': int(pistol_winners_won),
                                  'win_rate': win_rate}
            print(f"Matches analyzed: {total_matches:,}")
            print(f"Pistol winners who won match: {pistol_winners_won:,.0f}")
            print(f"Win rate: {win_rate:.1%}")
            print(f"Advantage over 50%: +{(win_rate - 0.5) * 100:.1f} pp")
    
    # Combined analysis: winning BOTH pistol rounds
    if 1 in results and 16 in results:
        print(f"\n--- Combined: Both Pistol Rounds ---")
        both_pistols = []
        matches_with_both = df[df['round_num'].isin([1, 16])].groupby('match_id').filter(
            lambda x: len(x['round_num'].unique()) == 2)['match_id'].unique()
        
        for match_id in matches_with_both:
            match_data = df[df['match_id'] == match_id]
            r1_winner = match_data[(match_data['round_num'] == 1) & (match_data['team_round_result'] == 1)]
            r16_winner = match_data[(match_data['round_num'] == 16) & (match_data['team_round_result'] == 1)]
            if len(r1_winner) == 0 or len(r16_winner) == 0:
                continue
            r1_team = r1_winner.iloc[0]['team_name']
            r16_team = r16_winner.iloc[0]['team_name']
            won_both = r1_team == r16_team
            team_data = match_data[match_data['team_name'] == r1_team]
            if len(team_data) == 0:
                continue
            r1_team_won_match = team_data.iloc[0]['team_is_match_winner']
            both_pistols.append({'match_id': match_id, 'won_both_pistols': won_both,
                                 'won_match': r1_team_won_match if won_both else None})
        
        both_df = pd.DataFrame(both_pistols)
        won_both_matches = both_df[both_df['won_both_pistols'] == True]
        if len(won_both_matches) > 0:
            both_win_rate = won_both_matches['won_match'].mean()
            results['both'] = {'name': 'Won Both Pistol Rounds', 
                               'total_matches': len(won_both_matches), 'win_rate': both_win_rate}
            print(f"Teams winning both pistols: {len(won_both_matches):,}")
            print(f"Match win rate when winning both: {both_win_rate:.1%}")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f'Pistol Round Impact on Match Outcomes - {dataset_name.upper()}', fontsize=14, fontweight='bold')
    
    categories, win_rates, counts, colors_list = [], [], [], []
    if 1 in results:
        categories.append('Round 1\n(First Half)')
        win_rates.append(results[1]['win_rate'] * 100)
        counts.append(results[1]['total_matches'])
        colors_list.append('#2E86AB')
    if 16 in results:
        categories.append('Round 16\n(Second Half)')
        win_rates.append(results[16]['win_rate'] * 100)
        counts.append(results[16]['total_matches'])
        colors_list.append('#A23B72')
    if 'both' in results:
        categories.append('Both Pistols')
        win_rates.append(results['both']['win_rate'] * 100)
        counts.append(results['both']['total_matches'])
        colors_list.append('#F18F01')
    
    x_pos = np.arange(len(categories))
    bars = ax.bar(x_pos, win_rates, color=colors_list, alpha=0.8, edgecolor='black')
    ax.axhline(50, color='gray', linestyle='--', alpha=0.7, label='50% (Random)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Match Win Rate (%)', fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, rate, count in zip(bars, win_rates, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{rate:.1f}%\n(n={count:,})', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(EXTENDED_DIR / f'pistol_round_impact_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: pistol_round_impact_{dataset_name}.png")
    
    # Save results to CSV
    results_for_csv = [{'pistol_round': k, 'name': v['name'], 'total_matches': v['total_matches'], 
                        'win_rate': v['win_rate']} for k, v in results.items()]
    pd.DataFrame(results_for_csv).to_csv(EXTENDED_DIR / f'pistol_round_results_{dataset_name}.csv', index=False)
    print(f"Saved: pistol_round_results_{dataset_name}.csv")
    
    return results



def analyze_comeback_potential(dataset_name="kaggle"):
    """Analyze comeback potential from different scorelines"""
    print(f"\n{'='*60}")
    print(f"COMEBACK POTENTIAL ANALYSIS - {dataset_name.upper()}")
    print(f"{'='*60}")

    csv_path = PROCESSED_DIR / f"team_round_features_{dataset_name}.csv"
    df = pd.read_csv(csv_path, low_memory=False)

    # Focus on deficits at round 15 (halfway point)
    round15 = df[df['round_num'] == 15].copy()

    comeback_scenarios = []

    for _, row in round15.iterrows():
        score_diff = row['score_diff']  # Positive means team is ahead
        match_winner = row['team_is_match_winner']

        # Categorize deficit
        if score_diff >= 5:
            deficit_category = "Leading by 5+"
        elif score_diff >= 3:
            deficit_category = "Leading by 3-4"
        elif score_diff >= 1:
            deficit_category = "Leading by 1-2"
        elif score_diff == 0:
            deficit_category = "Tied"
        elif score_diff >= -2:
            deficit_category = "Behind by 1-2"
        elif score_diff >= -4:
            deficit_category = "Behind by 3-4"
        else:
            deficit_category = "Behind by 5+"

        comeback_scenarios.append({
            'deficit_category': deficit_category,
            'score_diff': score_diff,
            'won_match': match_winner
        })

    comeback_df = pd.DataFrame(comeback_scenarios)

    # Calculate comeback rates
    print("\nComeback Rates by Scoreline at Round 15:")
    print("-" * 60)

    category_order = [
        "Behind by 5+",
        "Behind by 3-4",
        "Behind by 1-2",
        "Tied",
        "Leading by 1-2",
        "Leading by 3-4",
        "Leading by 5+"
    ]

    for category in category_order:
        cat_data = comeback_df[comeback_df['deficit_category'] == category]
        if len(cat_data) > 0:
            win_rate = cat_data['won_match'].mean()
            count = len(cat_data)
            print(f"{category:20} | Win rate: {win_rate:6.1%} | Sample size: {count:5,}")

    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))

    summary = comeback_df.groupby('deficit_category')['won_match'].agg(['mean', 'count']).reset_index()
    summary = summary.set_index('deficit_category').reindex(category_order)

    colors = ['#d62728', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#1f77b4', '#aec7e8']
    ax.barh(range(len(summary)), summary['mean'] * 100, color=colors)
    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels(summary.index)
    ax.set_xlabel('Win Rate (%)', fontsize=12)
    ax.set_title(f'Match Win Rate by Scoreline at Round 15 - {dataset_name.upper()}',
                 fontsize=14, fontweight='bold')
    ax.axvline(50, color='black', linestyle='--', alpha=0.5, label='50% (random)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (val, count) in enumerate(zip(summary['mean'] * 100, summary['count'])):
        ax.text(val + 2, i, f'{val:.1f}% (n={count:,})',
                va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(EXTENDED_DIR / f'comeback_potential_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: comeback_potential_{dataset_name}.png")

def analyze_feature_importance(dataset_name="kaggle"):
    """Analyze which features are most important for predictions"""
    print(f"\n{'='*60}")
    print(f"FEATURE IMPORTANCE ANALYSIS - {dataset_name.upper()}")
    print(f"{'='*60}")

    # Load trained model
    model_path = Path("models/saved_models") / f"random_forest_{dataset_name}_pipeline.joblib"

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    pipeline = joblib.load(model_path)
    model = pipeline.named_steps['model']

    # Get feature names
    feature_names = [
        'round_num', 'round_duration_seconds', 'team_score_before',
        'opp_score_before', 'score_diff', 'team_eq_start', 'team_eq_end',
        'team_eq_spend', 'team_alive_end', 'team_total_utility', 'team_kills',
        'team_deaths', 'team_damage', 'team_round_result',
        'team_cumulative_kills', 'team_cumulative_deaths',
        'team_cumulative_damage', 'team_cumulative_eq_spend', 'is_ct'
    ]

    # Get feature importances from random forest
    importances = model.feature_importances_

    # Sort by importance
    indices = np.argsort(importances)[::-1]

    print("\nTop 10 Most Important Features:")
    print("-" * 60)
    for i, idx in enumerate(indices[:10], 1):
        print(f"{i:2}. {feature_names[idx]:30} {importances[idx]:.4f}")

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 8))

    top_n = 15
    top_indices = indices[:top_n]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = importances[top_indices]

    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_importances, color='steelblue', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Most Important Features - {dataset_name.upper()}',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, val in enumerate(top_importances):
        ax.text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(EXTENDED_DIR / f'feature_importance_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: feature_importance_{dataset_name}.png")

def analyze_overtime_rounds(dataset_name="kaggle"):
    """Analyze prediction accuracy in overtime rounds (30+)"""
    print(f"\n{'='*60}")
    print(f"OVERTIME ANALYSIS - {dataset_name.upper()}")
    print(f"{'='*60}")

    csv_path = PROCESSED_DIR / f"team_round_features_{dataset_name}.csv"
    df = pd.read_csv(csv_path, low_memory=False)

    # Check how many matches go to overtime
    matches_by_length = df.groupby('match_id')['round_num'].max().value_counts().sort_index()

    print(f"\nMatch Length Distribution:")
    print("-" * 60)

    total_matches = len(df['match_id'].unique())
    matches_30_or_less = matches_by_length[matches_by_length.index <= 30].sum()
    matches_overtime = matches_by_length[matches_by_length.index > 30].sum()

    print(f"Total matches: {total_matches:,}")
    print(f"Matches ending in regulation (≤30 rounds): {matches_30_or_less:,} ({matches_30_or_less/total_matches:.1%})")
    print(f"Matches going to overtime (>30 rounds): {matches_overtime:,} ({matches_overtime/total_matches:.1%})")

    if matches_overtime > 0:
        print(f"\nOvertime rounds found in dataset!")
        print(f"Maximum rounds in any match: {matches_by_length.index.max()}")

        # Get overtime round data
        overtime_data = df[df['round_num'] > 30]
        if len(overtime_data) > 0:
            print(f"Total overtime round samples: {len(overtime_data):,}")
            print(f"\nNote: Overtime analysis requires retraining models on overtime data")
            print(f"      Current models were trained on rounds 1-30 only")
    else:
        print(f"\nNo overtime rounds found in {dataset_name} dataset.")
        print(f"All matches ended in regulation time (≤30 rounds)")

def main():
    """Run all extended analyses"""
    print("="*60)
    print("CS:GO WIN PREDICTION - EXTENDED ANALYSIS")
    print("="*60)

    # Run analyses on Kaggle dataset (largest sample size)
    dataset = "kaggle"

    analyze_pistol_round_impact(dataset)
    analyze_comeback_potential(dataset)
    analyze_feature_importance(dataset)
    analyze_overtime_rounds(dataset)

    print(f"\n{'='*60}")
    print("EXTENDED ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    print(f"Results saved to: {EXTENDED_DIR}")
    print("\nKey files generated:")
    print(f"  • pistol_round_impact_{dataset}.png")
    print(f"  • pistol_round_results_{dataset}.csv")
    print(f"  • comeback_potential_{dataset}.png")
    print(f"  • feature_importance_{dataset}.png")

if __name__ == "__main__":
    main()

