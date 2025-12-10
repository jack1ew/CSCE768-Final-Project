#!/usr/bin/env python3
"""
Pistol Round Analysis - RQ2
Analyzes the impact of winning pistol rounds (rounds 1 and 16) on overall match outcomes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data_processing' / 'data' / 'processed'
RESULTS_DIR = Path(__file__).parent / 'results' / 'pistol_analysis'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data(dataset_name):
    """Load processed dataset."""
    file_path = DATA_DIR / f'team_round_features_{dataset_name}_regulation.csv'
    print(f"Loading {file_path}")
    df = pd.read_csv(file_path)
    return df

def analyze_pistol_rounds(df, dataset_name):
    """Analyze impact of winning pistol rounds on match outcomes."""

    # Get pistol rounds (round 1 and round 16)
    pistol_rounds = df[df['round_num'].isin([1, 16])].copy()

    # For each match, determine which team won which pistol round
    match_pistol_results = []

    for match_id in pistol_rounds['match_id'].unique():
        match_data = pistol_rounds[pistol_rounds['match_id'] == match_id]

        # Get all teams in this match
        teams = match_data['team_name'].unique()

        for team in teams:
            team_data = match_data[match_data['team_name'] == team]

            # Check if this team won the match
            match_winner = team_data['team_is_match_winner'].iloc[0]

            # Get round 1 result
            r1_data = team_data[team_data['round_num'] == 1]
            won_r1 = r1_data['team_round_result'].iloc[0] if len(r1_data) > 0 else None

            # Get round 16 result (second half pistol)
            r16_data = team_data[team_data['round_num'] == 16]
            won_r16 = r16_data['team_round_result'].iloc[0] if len(r16_data) > 0 else None

            if won_r1 is not None and won_r16 is not None:
                match_pistol_results.append({
                    'match_id': match_id,
                    'team_name': team,
                    'won_r1': bool(won_r1),
                    'won_r16': bool(won_r16),
                    'won_both_pistols': bool(won_r1 and won_r16),
                    'won_one_pistol': bool(won_r1 != won_r16),
                    'won_no_pistols': bool(not won_r1 and not won_r16),
                    'match_winner': bool(match_winner)
                })

    pistol_df = pd.DataFrame(match_pistol_results)

    return pistol_df

def calculate_win_rates(pistol_df):
    """Calculate win rates based on pistol round outcomes."""

    stats = {
        'both_pistols': {
            'total': pistol_df['won_both_pistols'].sum(),
            'match_wins': pistol_df[pistol_df['won_both_pistols']]['match_winner'].sum(),
        },
        'one_pistol': {
            'total': pistol_df['won_one_pistol'].sum(),
            'match_wins': pistol_df[pistol_df['won_one_pistol']]['match_winner'].sum(),
        },
        'no_pistols': {
            'total': pistol_df['won_no_pistols'].sum(),
            'match_wins': pistol_df[pistol_df['won_no_pistols']]['match_winner'].sum(),
        },
        'r1_only': {
            'total': pistol_df[pistol_df['won_r1']].shape[0],
            'match_wins': pistol_df[pistol_df['won_r1']]['match_winner'].sum(),
        },
        'r16_only': {
            'total': pistol_df[pistol_df['won_r16']].shape[0],
            'match_wins': pistol_df[pistol_df['won_r16']]['match_winner'].sum(),
        }
    }

    # Calculate win rates
    for key in stats:
        if stats[key]['total'] > 0:
            stats[key]['win_rate'] = stats[key]['match_wins'] / stats[key]['total']
        else:
            stats[key]['win_rate'] = 0.0

    return stats

def plot_win_rates(stats_dict, output_dir):
    """Plot win rates based on pistol round outcomes."""

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # For each dataset
    datasets = ['combined', 'kaggle', 'esta']
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    for idx, dataset in enumerate(datasets):
        if dataset not in stats_dict:
            continue

        stats = stats_dict[dataset]

        # Plot 1: Win rate by number of pistol rounds won
        ax = axes[0]
        categories = ['Won Both\nPistols', 'Won One\nPistol', 'Won No\nPistols']
        win_rates = [
            stats['both_pistols']['win_rate'] * 100,
            stats['one_pistol']['win_rate'] * 100,
            stats['no_pistols']['win_rate'] * 100
        ]
        counts = [
            stats['both_pistols']['total'],
            stats['one_pistol']['total'],
            stats['no_pistols']['total']
        ]

        x = np.arange(len(categories))
        width = 0.25
        offset = (idx - 1) * width

        bars = ax.bar(x + offset, win_rates, width, label=dataset.upper(),
                     color=colors[idx], alpha=0.8)

        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%\n(n={count})',
                   ha='center', va='bottom', fontsize=8)

    axes[0].set_ylabel('Match Win Rate (%)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Pistol Rounds Won', fontsize=12, fontweight='bold')
    axes[0].set_title('Match Win Rate by Pistol Round Outcomes', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim([0, 100])

    # Plot 2: Individual pistol round impact
    for idx, dataset in enumerate(datasets):
        if dataset not in stats_dict:
            continue

        stats = stats_dict[dataset]
        ax = axes[1]

        categories = ['Won R1', 'Won R16']
        win_rates = [
            stats['r1_only']['win_rate'] * 100,
            stats['r16_only']['win_rate'] * 100
        ]
        counts = [
            stats['r1_only']['total'],
            stats['r16_only']['total']
        ]

        x = np.arange(len(categories))
        width = 0.25
        offset = (idx - 1) * width

        bars = ax.bar(x + offset, win_rates, width, label=dataset.upper(),
                     color=colors[idx], alpha=0.8)

        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%\n(n={count})',
                   ha='center', va='bottom', fontsize=8)

    axes[1].set_ylabel('Match Win Rate (%)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Pistol Round', fontsize=12, fontweight='bold')
    axes[1].set_title('Match Win Rate by Individual Pistol Round', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(categories)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim([0, 100])

    # Plot 3: Summary bar chart comparing datasets
    ax = axes[2]
    categories = ['Both Pistols', 'One Pistol', 'No Pistols']

    for idx, dataset in enumerate(datasets):
        if dataset not in stats_dict:
            continue

        stats = stats_dict[dataset]
        win_rates = [
            stats['both_pistols']['win_rate'] * 100,
            stats['one_pistol']['win_rate'] * 100,
            stats['no_pistols']['win_rate'] * 100
        ]

        x = np.arange(len(categories))
        ax.plot(x, win_rates, marker='o', linewidth=2, markersize=8,
               label=dataset.upper(), color=colors[idx])

    ax.set_ylabel('Match Win Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Pistol Round Outcome', fontsize=12, fontweight='bold')
    ax.set_title('Win Rate Trends Across Datasets', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(output_dir / 'pistol_round_win_rates.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'pistol_round_win_rates.png'}")
    plt.close()

def create_summary_table(stats_dict, output_dir):
    """Create summary table of pistol round statistics."""

    rows = []
    for dataset in ['combined', 'kaggle', 'esta']:
        if dataset not in stats_dict:
            continue

        stats = stats_dict[dataset]

        rows.append({
            'Dataset': dataset.upper(),
            'Outcome': 'Won Both Pistols',
            'Sample Size': stats['both_pistols']['total'],
            'Match Wins': stats['both_pistols']['match_wins'],
            'Win Rate (%)': f"{stats['both_pistols']['win_rate']*100:.1f}"
        })

        rows.append({
            'Dataset': dataset.upper(),
            'Outcome': 'Won One Pistol',
            'Sample Size': stats['one_pistol']['total'],
            'Match Wins': stats['one_pistol']['match_wins'],
            'Win Rate (%)': f"{stats['one_pistol']['win_rate']*100:.1f}"
        })

        rows.append({
            'Dataset': dataset.upper(),
            'Outcome': 'Won No Pistols',
            'Sample Size': stats['no_pistols']['total'],
            'Match Wins': stats['no_pistols']['match_wins'],
            'Win Rate (%)': f"{stats['no_pistols']['win_rate']*100:.1f}"
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'pistol_round_statistics.csv', index=False)
    print(f"Saved: {output_dir / 'pistol_round_statistics.csv'}")

    return df

def main():
    """Main analysis function."""

    print("="*80)
    print("PISTOL ROUND ANALYSIS - RQ2")
    print("="*80)

    stats_dict = {}
    pistol_dfs = {}

    # Analyze each dataset
    for dataset in ['combined', 'kaggle', 'esta']:
        print(f"\n{'='*80}")
        print(f"Analyzing {dataset.upper()} dataset...")
        print(f"{'='*80}")

        try:
            df = load_data(dataset)
            print(f"Loaded {len(df)} rows")

            pistol_df = analyze_pistol_rounds(df, dataset)
            pistol_dfs[dataset] = pistol_df

            print(f"Analyzed {len(pistol_df)} team-match combinations")

            stats = calculate_win_rates(pistol_df)
            stats_dict[dataset] = stats

            # Print statistics
            print(f"\n{dataset.upper()} Results:")
            print(f"  Won Both Pistols: {stats['both_pistols']['match_wins']}/{stats['both_pistols']['total']} = {stats['both_pistols']['win_rate']*100:.1f}%")
            print(f"  Won One Pistol:   {stats['one_pistol']['match_wins']}/{stats['one_pistol']['total']} = {stats['one_pistol']['win_rate']*100:.1f}%")
            print(f"  Won No Pistols:   {stats['no_pistols']['match_wins']}/{stats['no_pistols']['total']} = {stats['no_pistols']['win_rate']*100:.1f}%")
            print(f"  Won R1:           {stats['r1_only']['match_wins']}/{stats['r1_only']['total']} = {stats['r1_only']['win_rate']*100:.1f}%")
            print(f"  Won R16:          {stats['r16_only']['match_wins']}/{stats['r16_only']['total']} = {stats['r16_only']['win_rate']*100:.1f}%")

        except Exception as e:
            print(f"Error analyzing {dataset}: {e}")
            import traceback
            traceback.print_exc()

    # Create visualizations
    if stats_dict:
        print(f"\n{'='*80}")
        print("Creating visualizations...")
        print(f"{'='*80}")
        plot_win_rates(stats_dict, RESULTS_DIR)

        print(f"\n{'='*80}")
        print("Creating summary table...")
        print(f"{'='*80}")
        summary_df = create_summary_table(stats_dict, RESULTS_DIR)
        print("\nSummary Table:")
        print(summary_df.to_string(index=False))

    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
