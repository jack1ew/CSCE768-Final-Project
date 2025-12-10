#!/usr/bin/env python3
"""
Economic Momentum Analysis
Analyze how equipment/money creates momentum through:
- Equipment savings after wins
- Forced eco rounds after losses
- Economic breathing room
- Momentum building patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")

# Paths
DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results/economic_analysis")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Economic thresholds (typical CS:GO economy)
ECO_THRESHOLD = 10000  # Below this = eco round
FORCE_BUY_THRESHOLD = 15000  # 15k-20k = force buy
FULL_BUY_THRESHOLD = 20000  # Above 20k = full buy capability

def load_regulation_data(dataset_name="kaggle"):
    """Load regulation rounds only"""
    csv_path = DATA_DIR / f"team_round_features_{dataset_name}_regulation.csv"
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    return df

def analyze_equipment_after_result():
    """Analyze equipment value in next round after win/loss"""
    print("\n" + "="*80)
    print("ANALYZING EQUIPMENT CHANGES AFTER WIN/LOSS")
    print("="*80)

    df = load_regulation_data("kaggle")

    # Add next round equipment as feature
    df = df.sort_values(['match_id', 'round_num'])
    df['next_round_eq_start'] = df.groupby('match_id')['team_eq_start'].shift(-1)
    df['eq_change'] = df['next_round_eq_start'] - df['team_eq_end']

    # Filter out last rounds of matches
    df = df[df['next_round_eq_start'].notna()].copy()

    # Separate by round result
    wins = df[df['team_round_result'] == 1].copy()
    losses = df[df['team_round_result'] == 0].copy()

    print(f"\nRounds analyzed:")
    print(f"  Wins:   {len(wins):,}")
    print(f"  Losses: {len(losses):,}")

    print(f"\nAVERAGE STARTING EQUIPMENT (Next Round):")
    print(f"  After Win:  ${wins['next_round_eq_start'].mean():,.0f}")
    print(f"  After Loss: ${losses['next_round_eq_start'].mean():,.0f}")
    print(f"  Difference: ${wins['next_round_eq_start'].mean() - losses['next_round_eq_start'].mean():,.0f}")

    # Categorize economy states
    def categorize_economy(eq_value):
        if eq_value < ECO_THRESHOLD:
            return "Eco"
        elif eq_value < FORCE_BUY_THRESHOLD:
            return "Force Buy"
        elif eq_value < FULL_BUY_THRESHOLD:
            return "Semi-Buy"
        else:
            return "Full Buy"

    wins['next_economy'] = wins['next_round_eq_start'].apply(categorize_economy)
    losses['next_economy'] = losses['next_round_eq_start'].apply(categorize_economy)

    print(f"\nECONOMY STATE IN NEXT ROUND:")
    print(f"\nAfter WIN:")
    for state, count in wins['next_economy'].value_counts().items():
        pct = count / len(wins) * 100
        print(f"  {state:12} {count:7,} ({pct:5.1f}%)")

    print(f"\nAfter LOSS:")
    for state, count in losses['next_economy'].value_counts().items():
        pct = count / len(losses) * 100
        print(f"  {state:12} {count:7,} ({pct:5.1f}%)")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Economic Momentum: Equipment Changes After Round Results',
                 fontsize=18, fontweight='bold')

    # Plot 1: Distribution of next round equipment
    ax = axes[0, 0]
    ax.hist([wins['next_round_eq_start'], losses['next_round_eq_start']],
            bins=50, alpha=0.7, label=['After Win', 'After Loss'],
            color=['green', 'red'], edgecolor='black')
    ax.axvline(x=ECO_THRESHOLD, color='gray', linestyle='--', alpha=0.5, label='Eco Threshold')
    ax.axvline(x=FULL_BUY_THRESHOLD, color='blue', linestyle='--', alpha=0.5, label='Full Buy Threshold')
    ax.set_xlabel('Starting Equipment Value (Next Round)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Equipment Distribution in Next Round', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Economy state breakdown
    ax = axes[0, 1]
    economy_states = ['Eco', 'Force Buy', 'Semi-Buy', 'Full Buy']
    win_counts = [wins[wins['next_economy'] == state].shape[0] / len(wins) * 100
                  for state in economy_states]
    loss_counts = [losses[losses['next_economy'] == state].shape[0] / len(losses) * 100
                   for state in economy_states]

    x = np.arange(len(economy_states))
    width = 0.35
    ax.bar(x - width/2, win_counts, width, label='After Win', alpha=0.8, color='green', edgecolor='black')
    ax.bar(x + width/2, loss_counts, width, label='After Loss', alpha=0.8, color='red', edgecolor='black')

    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Economy State Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(economy_states)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (win_val, loss_val) in enumerate(zip(win_counts, loss_counts)):
        ax.text(i - width/2, win_val + 1, f'{win_val:.1f}%', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, loss_val + 1, f'{loss_val:.1f}%', ha='center', va='bottom', fontsize=9)

    # Plot 3: Equipment savings (end value vs start value in next round)
    ax = axes[1, 0]
    wins_sample = wins.sample(n=min(10000, len(wins)), random_state=42)
    ax.scatter(wins_sample['team_eq_end'], wins_sample['next_round_eq_start'],
               alpha=0.3, s=10, color='green', label='After Win')
    ax.plot([0, 30000], [0, 30000], 'k--', alpha=0.3, label='No Change Line')
    ax.set_xlabel('Ending Equipment Value', fontsize=12, fontweight='bold')
    ax.set_ylabel('Starting Equipment Value (Next Round)', fontsize=12, fontweight='bold')
    ax.set_title('Equipment Savings After Win', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Forced eco rounds after consecutive losses
    ax = axes[1, 1]
    # Calculate consecutive losses
    df_sorted = df.sort_values(['match_id', 'round_num'])
    df_sorted['prev_result'] = df_sorted.groupby('match_id')['team_round_result'].shift(1)
    df_sorted['prev2_result'] = df_sorted.groupby('match_id')['team_round_result'].shift(2)

    consecutive_1_loss = df_sorted[(df_sorted['prev_result'] == 0) &
                                    (df_sorted['team_eq_start'].notna())]
    consecutive_2_loss = df_sorted[(df_sorted['prev_result'] == 0) &
                                    (df_sorted['prev2_result'] == 0) &
                                    (df_sorted['team_eq_start'].notna())]

    categories = ['1 Loss', '2+ Losses']
    eco_pct = [
        (consecutive_1_loss['team_eq_start'] < ECO_THRESHOLD).sum() / len(consecutive_1_loss) * 100,
        (consecutive_2_loss['team_eq_start'] < ECO_THRESHOLD).sum() / len(consecutive_2_loss) * 100
    ]

    ax.bar(categories, eco_pct, alpha=0.8, color=['orange', 'darkred'], edgecolor='black')
    ax.set_ylabel('% Forced to Eco', fontsize=12, fontweight='bold')
    ax.set_title('Forced Eco Rounds After Consecutive Losses', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, val in enumerate(eco_pct):
        ax.text(i, val + 1, f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    output_path = RESULTS_DIR / 'economic_momentum_overview.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")

    return wins, losses

def analyze_momentum_building():
    """Analyze how economic advantages build momentum"""
    print("\n" + "="*80)
    print("ANALYZING MOMENTUM BUILDING PATTERNS")
    print("="*80)

    df = load_regulation_data("kaggle")
    df = df.sort_values(['match_id', 'round_num'])

    # Calculate win streaks
    df['win_streak'] = 0
    for match_id in df['match_id'].unique():
        match_df = df[df['match_id'] == match_id].copy()
        streak = 0
        streaks = []
        for result in match_df['team_round_result']:
            if result == 1:
                streak += 1
            else:
                streak = 0
            streaks.append(streak)
        df.loc[df['match_id'] == match_id, 'win_streak'] = streaks

    # Analyze equipment advantage by win streak
    streak_analysis = []
    for streak in range(0, 6):
        streak_data = df[df['win_streak'] == streak]
        if len(streak_data) > 0:
            streak_analysis.append({
                'streak': streak,
                'count': len(streak_data),
                'avg_eq_start': streak_data['team_eq_start'].mean(),
                'avg_score_diff': streak_data['score_diff'].mean(),
                'win_rate': streak_data['team_round_result'].mean()
            })

    streak_df = pd.DataFrame(streak_analysis)

    print("\nMOMENTUM BY WIN STREAK:")
    print(f"{'Streak':<10} {'Rounds':<10} {'Avg Eq':<12} {'Avg Score Diff':<15} {'Win Rate':<10}")
    print("-" * 70)
    for _, row in streak_df.iterrows():
        print(f"{int(row['streak']):<10} {int(row['count']):<10,} "
              f"${row['avg_eq_start']:<11,.0f} {row['avg_score_diff']:<15.2f} "
              f"{row['win_rate']*100:<9.1f}%")

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Economic Momentum Building Patterns', fontsize=18, fontweight='bold')

    # Plot 1: Equipment by win streak
    ax = axes[0]
    ax.plot(streak_df['streak'], streak_df['avg_eq_start'], marker='o', linewidth=2,
            markersize=8, color='green')
    ax.set_xlabel('Win Streak Length', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Starting Equipment ($)', fontsize=12, fontweight='bold')
    ax.set_title('Equipment Value vs Win Streak', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 2: Win rate by win streak
    ax = axes[1]
    ax.plot(streak_df['streak'], streak_df['win_rate'] * 100, marker='o', linewidth=2,
            markersize=8, color='blue')
    ax.set_xlabel('Win Streak Length', fontsize=12, fontweight='bold')
    ax.set_ylabel('Win Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Win Rate vs Win Streak (Momentum)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Plot 3: Score differential by win streak
    ax = axes[2]
    ax.plot(streak_df['streak'], streak_df['avg_score_diff'], marker='o', linewidth=2,
            markersize=8, color='purple')
    ax.set_xlabel('Win Streak Length', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Score Differential', fontsize=12, fontweight='bold')
    ax.set_title('Score Lead vs Win Streak', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    output_path = RESULTS_DIR / 'momentum_building_patterns.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")

def analyze_economic_breathing_room():
    """Analyze how equipment cushion affects win probability"""
    print("\n" + "="*80)
    print("ANALYZING ECONOMIC BREATHING ROOM")
    print("="*80)

    df = load_regulation_data("kaggle")

    # Calculate equipment advantage over opponent
    # First, get opponent equipment by swapping teams
    df = df.sort_values(['match_id', 'round_num'])

    # For each round, calculate equipment differential
    df['eq_advantage'] = df['team_eq_start'] - df.groupby(['match_id', 'round_num'])['team_eq_start'].transform(
        lambda x: x.iloc[1] if len(x) > 1 and x.name == x.iloc[0] else (x.iloc[0] if len(x) > 1 else 0)
    )

    # Bin equipment advantages
    bins = [-np.inf, -10000, -5000, -2000, 2000, 5000, 10000, np.inf]
    labels = ['Large Disadvantage\n(<-$10k)', 'Moderate Disadvantage\n(-$10k to -$5k)',
              'Small Disadvantage\n(-$5k to -$2k)', 'Even\n(±$2k)',
              'Small Advantage\n($2k to $5k)', 'Moderate Advantage\n($5k to $10k)',
              'Large Advantage\n(>$10k)']

    df['eq_advantage_category'] = pd.cut(df['eq_advantage'], bins=bins, labels=labels)

    # Calculate win rate by equipment advantage
    advantage_analysis = df.groupby('eq_advantage_category', observed=True).agg({
        'team_round_result': ['mean', 'count']
    }).reset_index()

    advantage_analysis.columns = ['category', 'win_rate', 'count']

    print("\nWIN RATE BY EQUIPMENT ADVANTAGE:")
    print(f"{'Category':<30} {'Win Rate':<12} {'Rounds':<10}")
    print("-" * 60)
    for _, row in advantage_analysis.iterrows():
        print(f"{row['category']:<30} {row['win_rate']*100:<11.1f}% {int(row['count']):<10,}")

    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = ['darkred', 'red', 'orange', 'gray', 'lightgreen', 'green', 'darkgreen']

    bars = ax.bar(range(len(advantage_analysis)), advantage_analysis['win_rate'] * 100,
                   alpha=0.8, color=colors, edgecolor='black')

    ax.set_ylabel('Win Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Economic Breathing Room: Win Rate by Equipment Advantage',
                 fontsize=16, fontweight='bold')
    ax.set_xticks(range(len(advantage_analysis)))
    ax.set_xticklabels(advantage_analysis['category'], rotation=15, ha='right', fontsize=10)
    ax.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='50% (Even)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (rate, count) in enumerate(zip(advantage_analysis['win_rate'] * 100,
                                           advantage_analysis['count'])):
        ax.text(i, rate + 2, f'{rate:.1f}%\n({count:,} rounds)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    output_path = RESULTS_DIR / 'economic_breathing_room.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")

def main():
    """Main economic momentum analysis"""
    print("="*80)
    print("ECONOMIC MOMENTUM ANALYSIS")
    print("="*80)

    wins, losses = analyze_equipment_after_result()
    analyze_momentum_building()
    analyze_economic_breathing_room()

    print("\n" + "="*80)
    print("ECONOMIC MOMENTUM ANALYSIS COMPLETE!")
    print("="*80)
    print(f"Results saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()

