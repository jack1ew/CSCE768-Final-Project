# Pistol Round Impact Analysis (RQ2)

## Research Question
**RQ2:** How impactful is winning the first round of each half (pistol rounds) on overall match outcomes?

## Methodology
We analyzed all matches across three datasets (ESTA, Kaggle, Combined) to determine which teams won pistol rounds (rounds 1 and 16) and correlated these outcomes with final match winners. Each match contributed 2 team-level observations (one per team).

## Key Findings

### Overall Win Rates by Pistol Round Outcomes

| Dataset  | Outcome           | Sample Size | Match Wins | Win Rate |
|----------|-------------------|-------------|------------|----------|
| COMBINED | Won Both Pistols  | 16,793      | 10,981     | **65.4%** |
| COMBINED | Won One Pistol    | 32,940      | 16,470     | **50.0%** |
| COMBINED | Won No Pistols    | 16,791      | 5,811      | **34.6%** |
| KAGGLE   | Won Both Pistols  | 16,001      | 10,437     | **65.2%** |
| KAGGLE   | Won One Pistol    | 31,418      | 15,709     | **50.0%** |
| KAGGLE   | Won No Pistols    | 16,000      | 5,563      | **34.8%** |
| ESTA     | Won Both Pistols  | 792         | 544        | **68.7%** |
| ESTA     | Won One Pistol    | 1,522       | 761        | **50.0%** |
| ESTA     | Won No Pistols    | 791         | 248        | **31.4%** |

### Individual Pistol Round Impact

| Dataset  | Pistol Round | Sample Size | Match Wins | Win Rate |
|----------|--------------|-------------|------------|----------|
| COMBINED | Round 1      | 33,263      | 19,213     | **57.8%** |
| COMBINED | Round 16     | 33,263      | 19,219     | **57.8%** |
| KAGGLE   | Round 1      | 31,710      | 18,277     | **57.6%** |
| KAGGLE   | Round 16     | 31,710      | 18,306     | **57.7%** |
| ESTA     | Round 1      | 1,553       | 936        | **60.3%** |
| ESTA     | Round 16     | 1,553       | 913        | **58.8%** |

## Strategic Insights

### 1. Dramatic Win Rate Swing
- **30-38 percentage point difference** between winning both pistols (65-69%) vs losing both (31-35%)
- This represents one of the strongest predictors of match outcomes in CS:GO
- Validates the competitive community's emphasis on pistol round practice

### 2. Perfect Balance with Split Pistols
- Teams winning exactly one pistol round achieve **exactly 50%** win rates across all datasets
- This suggests that pistol round advantages roughly cancel out when split between teams
- Economic momentum from one pistol can be countered by momentum from the other

### 3. Individual Pistol Round Value
- Each individual pistol round won correlates with **57-60% match win rate**
- Both R1 (first half) and R16 (second half) show similar impact
- Slight advantage to R1 in ESTA dataset (60.3% vs 58.8%) but difference is small

### 4. Consistency Across Datasets
- Effect sizes are remarkably consistent across:
  - ESTA: 792-1,553 samples (professional matches, high telemetry)
  - Kaggle: 16,000-31,710 samples (large-scale, economy-focused)
  - Combined: 16,791-33,263 samples (union of both)
- This validates that the pistol round effect is robust and not dataset-specific

## Economic Explanation

The large impact of pistol rounds stems from CS:GO's economic cascade system:

1. **Winning Pistol → Equipment Savings**: Teams winning pistol rounds often survive with equipment (armor, upgraded pistols), reducing buy costs for Round 2/17

2. **Round 2/17 Advantage**: Pistol winners can afford full rifle buys while losers are economically forced into "eco rounds" with pistols only

3. **Multi-Round Momentum**: Winning rounds 2-4 (first half) or 17-19 (second half) builds substantial economic leads that compound over 5+ rounds

4. **Comeback Difficulty**: Teams losing both pistols start both halves economically disadvantaged, making comebacks significantly harder

## Practical Implications

### For Competitive Teams
- Justify extensive pistol-only practice time (5-10% of practice regimen)
- Develop specialized pistol round strategies and setups
- Consider pistol specialists in team composition

### For Esports Analytics
- Pistol round outcomes should be prominent features in live win probability models
- Early-game predictions (Rounds 1-5, 16-20) should heavily weight pistol results

### For Betting Markets
- 30+ percentage point swings justify significant odds adjustments after pistol rounds
- In-game betting markets should rapidly update after R1 and R16 outcomes

### For Broadcast Production
- Highlight pistol round importance in pre-match commentary
- Use "pistol round win rate" as a team performance metric
- Create graphics showing pistol round impact on live win probability

## Visualization

See `pistol_round_win_rates.png` for visual representation of these findings across all datasets.

## Conclusion

Pistol rounds represent **critical momentum points** in CS:GO matches, with teams winning both pistols achieving 65-69% match win rates and teams losing both winning only 31-35%. The 30-38 percentage point swing validates the strategic importance placed on pistol rounds in competitive play and provides quantitative evidence for resource allocation toward pistol-specific practice and preparation.
