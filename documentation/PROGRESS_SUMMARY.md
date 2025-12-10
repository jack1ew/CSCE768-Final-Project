# CS:GO Win Probability Analysis - Progress Summary
**Jackie Wang | Date: November 16, 2025**

---

## Overtime Analysis & Regulation Data Separation

### Key Findings:
- **ESTA Dataset**: 108 overtime matches (6.93%), reaching up to 71 rounds (7 overtime periods)
- **Kaggle Dataset**: 0 overtime matches (all ended in regulation ≤30 rounds)
- **Combined Dataset**: 108 overtime matches (0.32% of total)

### Data Separation Completed:
| Dataset | Regulation Rounds | Overtime Rounds | Regulation Matches | Overtime Matches |
|---------|------------------|-----------------|-------------------|------------------|
| ESTA | 80,396 (97.86%) | 1,756 (2.14%) | 1,558 | 108 |
| Kaggle | 1,614,962 (100%) | 0 (0%) | 31,710 | 0 |
| Combined | 1,695,358 (99.90%) | 1,756 (0.10%) | 33,268 | 108 |

**Files Created:**
- `team_round_features_esta_regulation.csv` (17.05 MB)
- `team_round_features_kaggle_regulation.csv` (226.32 MB)
- `team_round_features_combined_regulation.csv` (244.52 MB)
- `team_round_features_esta_overtime.csv` (0.38 MB)
- `team_round_features_combined_overtime.csv` (0.41 MB)

---

## Model Retraining (In Progress)

**Status**: Retraining all models on regulation rounds only (1-30)

**Models Being Retrained** (15 total):
1. Logistic Regression × 3 datasets
2. Random Forest × 3 datasets
3. k-NN × 3 datasets
4. MLP (Neural Network) × 3 datasets
5. Transformer × 3 datasets

**Why Retraining?**:
- Overtime rounds have different economic rules (reset to fixed amount)
- Overtime should be analyzed separately as "mini-games"
- Main models should focus on regulation match prediction
- Improves prediction accuracy by removing confounding overtime data

---

## Economic Analysis (In Progress)

### Economic Momentum Analysis:
Investigating how equipment/money creates cascading advantages:
- **Equipment after wins vs losses**: How much $ teams start with next round
- **Forced eco rounds**: Frequency of <$10k starts after losses
- **Full buy capability**: Ability to purchase rifles/armor/utility
- **Momentum building**: Win streaks and economic advantages
- **Breathing room**: How equipment cushion allows mistakes

### Comeback Analysis with Economy:
Analyzing comeback potential considering both score and economy:
- **Score + Equipment heatmaps**: Win rates by deficit AND equipment state
- **Economic deficit impact**: How being broke while behind affects comebacks
- **Critical factors**: Equipment differences in successful vs failed comebacks
- **Comeback thresholds**: At what point (score + economy) are comebacks unlikely?

**Expected Insights**:
- Equipment differential compounds score differential effects
- Teams with full buy while behind have better comeback odds
- Forced eco rounds while trailing make comebacks nearly impossible
- Economic management is critical when behind in score

---

## Analyses Already Completed (Previous Work)

### Round-by-Round Accuracy Evolution:
- **Round 1**: 57.6% accuracy (barely better than random)
- **Round 10**: 68.8% accuracy (+11.2pp improvement)
- **Round 15**: 72.0% accuracy (+3.2pp improvement)
- **Round 30**: 74.5% accuracy (+2.5pp improvement)
- **Conclusion**: Most improvement in first 15 rounds (diminishing returns)

### First Round Impact:
- Winning Round 1: 57.6% match win rate
- Advantage: 7.6% above random chance (50%)
- **Not deterministic** - economic snowball + team quality signal

### Comeback Potential (Round 15):
| Score Situation | Win Rate | Interpretation |
|----------------|----------|----------------|
| Behind 5+ | 17.6% | Nearly insurmountable |
| Behind 3-4 | 30.1% | Difficult but possible |
| Behind 1-2 | 39.8% | Still competitive |
| Tied (7-7) | 50.0% | Even match |
| Leading 1-2 | 60.2% | Slight advantage |
| Leading 3-4 | 69.9% | Strong position |
| Leading 5+ | 82.4% | Dominant, likely win |

### Feature Importance (Across All Datasets):

**ESTA Dataset** (Rich telemetry):
1. Score Differential: 20.4%
2. Round Damage: 7.4%
3. Round Duration: 7.0%
- More balanced importance distribution

**Kaggle Dataset** (Economy-focused):
1. Score Differential: 25.1%
2. Ending Equipment: 24.5%
3. Starting Equipment: 24.4%
- Economic features dominate (49% combined)

**Combined Dataset**:
1. Score Differential: 28.3%
2. Ending Equipment: 21.0%
3. Starting Equipment: 20.5%
- Economic features remain critical (43% combined)

### Model Performance:

**Best Overall**: Transformer
- ESTA: 97.3% accuracy
- Kaggle: 84.6% accuracy
- Captures sequential dependencies and economic patterns

**Traditional Models** (Logistic, RF, k-NN, MLP):
- 65-75% accuracy range
- More interpretable
- Faster training and inference

---

## Remaining Work

### 1. Overtime-Specific Analysis
- Treat each overtime period as separate "mini-game"
- Analyze overtime prediction separately from regulation
- Economic patterns in overtime (reset to fixed amount)
- Overtime win prediction models

### 2. 3-Class Prediction Model
- Current: Binary (Win/Loss)
- New: 3-class (Win/Loss/Overtime)
- Predict probability of going to overtime at round 30
- Measure accuracy of tie predictions

### 3. Academic LaTeX Report
- Follow format of HW3 example report
- Proper sections: Abstract, Introduction, Methodology, Results, Conclusion
- Professional tables with booktabs
- Figures with captions
- Mathematical notation where appropriate

### 4. Comprehensive PDF Report Generation
- Combine all analyses into single academic document
- Include all visualizations
- Proper citations and references
- Executive summary with key findings

---

## Data Sources

1. **ESTA Dataset**: 82,152 samples, 1,558 matches
   - Rich telemetry (kills, damage, utility, positions)
   - Professional matches
   - 6.93% overtime rate

2. **Kaggle Dataset**: 1,614,962 samples, 31,710 matches
   - Economy-focused features
   - Large sample size
   - No overtime matches

3. **Combined Dataset**: 1,697,114 samples, 33,268 matches
   - Largest dataset
   - Balanced performance

---

## Scripts Created/Modified

**New Scripts:**
- `analyze_overtime_distribution.py` - Overtime analysis and visualization
- `separate_regulation_overtime.py` - Data filtering and separation
- `retrain_all_models_regulation.py` - Comprehensive retraining orchestrator
- `economic_momentum_analysis.py` - Economic patterns analysis
- `comeback_analysis_with_economy.py` - Score + economy comeback analysis

**Modified Scripts:**
- `train_utils.py` - Updated to use regulation datasets
- `generate_comprehensive_report.py` - Enhanced with feature importance

---

## Key Insights So Far

1. **Early game is NOT deterministic** - Matches remain competitive even after Round 1
2. **Economic decisions have cascading effects** - Wins save equipment, losses force ecos
3. **5-round deficit is critical threshold** - Only ~18% comeback rate
4. **Equipment matters as much as score** - Economic state compounds advantages
5. **Overtime is fundamentally different** - Separate analysis needed
6. **Sequence models excel** - Transformer outperforms traditional approaches
7. **Dataset characteristics matter** - ESTA's telemetry enables better predictions

---

## Technical Implementation

**Machine Learning Models:**
- Logistic Regression (baseline)
- Random Forest (feature importance)
- k-Nearest Neighbors (instance-based)
- Multi-Layer Perceptron (neural network)
- Transformer Encoder (sequential patterns)

**Features** (19 total):
- **Temporal**: round_num, round_duration_seconds
- **Score**: team_score_before, opp_score_before, score_diff
- **Economic**: team_eq_start, team_eq_end, team_eq_spend, team_cumulative_eq_spend
- **Performance**: team_kills, team_deaths, team_damage, team_round_result
- **Cumulative**: team_cumulative_kills, team_cumulative_deaths, team_cumulative_damage
- **Positional**: team_alive_end, team_total_utility, is_ct

**Target Variable**: team_is_match_winner (binary, will become 3-class)

---

## Next Session Priorities

1. ✅ Check status of background retraining (15 models)
2. ✅ Review economic momentum analysis results
3. ✅ Review comeback + economy analysis results
4. 🔲 Create overtime-specific analysis
5. 🔲 Implement 3-class prediction
6. 🔲 Build LaTeX report template
7. 🔲 Generate final academic PDF

---

**Last Updated**: November 16, 2025 11:32 PM
**Status**: Major restructuring in progress (overtime separation + retraining + economic analysis)
