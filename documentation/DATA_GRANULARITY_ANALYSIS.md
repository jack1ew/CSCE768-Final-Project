# Data Granularity Analysis - What's Available

## Your Question:
**"Does this have round-by-round player statistics like kills/assists/deaths by round 7? And the scoreline?"**

---

## Quick Answer:

### ✅ YES - You Have:
1. **Round-by-round scoreline** (who's winning at round 7, 10, etc.)
2. **Round-by-round equipment values** (economy for each team)
3. **Round-by-round winners** (which team won each specific round)

### ❌ NO - You Don't Have:
1. **Round-by-round PLAYER statistics** (individual kills/deaths/assists per round)
2. **Cumulative player kills through round 7**

### ⚠️ PARTIAL - You Can Derive:
1. **Team-level cumulative stats** (aggregate player stats to team level)
2. **Scoreline through round X** (count round winners)

---

## Detailed Breakdown

### 1. Economy.csv - Round-by-Round Team Data ✅

**What it has:**
- Equipment value for Team 1 at each round (rounds 1-30)
- Equipment value for Team 2 at each round (rounds 1-30)
- Winner of each round (rounds 1-30)

**Column structure:**
```
'1_t1', '2_t1', '3_t1', ..., '30_t1'    → Team 1 equipment at rounds 1-30
'1_t2', '2_t2', '3_t2', ..., '30_t2'    → Team 2 equipment at rounds 1-30
'1_winner', '2_winner', ..., '30_winner' → Round winners (1 or 2)
```

**Example:**
```
Match: G2 vs Natus Vincere on Nuke
Round 1: G2 equipment = $4,350, Na'Vi wins (winner = 2)
Round 2: G2 equipment = $1,100, Na'Vi wins (winner = 2)
Round 7: G2 equipment = $24,600, Na'Vi wins (winner = 2)
```

**What you CAN do:**
- ✅ Get scoreline at round 7: Count how many rounds each team won through round 7
- ✅ Track equipment values through the match
- ✅ Identify eco rounds, force buys, full buys
- ✅ See momentum shifts (round win streaks)

---

### 2. Players.csv - Match-Level Player Stats ⚠️

**What it has:**
- **TOTAL** kills/deaths/assists for entire match
- **NOT** broken down by individual rounds

**Columns:**
```
kills, deaths, assists, hs (headshots)  → Total for entire match
m1_kills, m1_deaths, m1_assists         → Map 1 totals
m2_kills, m2_deaths, m2_assists         → Map 2 totals
...
```

**Example:**
```
Player: Brehze (Evil Geniuses)
Total kills: 57 (for entire match)
Total deaths: 61 (for entire match)
Total assists: 14 (for entire match)
```

**What you CANNOT do:**
- ❌ Get player kills at round 7 specifically
- ❌ Get cumulative player kills through round 7
- ❌ Track individual player performance round-by-round

**What you CAN do:**
- ✅ Aggregate to team-level for entire match
- ✅ Use as pre-match "player strength" features
- ✅ Compute team K/D ratios for full match

---

### 3. Results.csv - Final Match Outcomes ✅

**What it has:**
- Final round scores
- Rounds won as CT/T for each team
- Map winner

**Columns:**
```
result_1, result_2  → Final round count (e.g., 16-14)
ct_1, t_1          → Rounds won as CT/T for team 1
ct_2, t_2          → Rounds won as CT/T for team 2
map_winner         → Which team won
```

**Example:**
```
Team 1: 16 rounds, Team 2: 14 rounds
Team 1 won 10 as CT, 6 as T
Team 2 won 5 as CT, 9 as T
```

---

## For Your CSCE768 Project

### ✅ Your Proposal Requirements vs. Available Data

From your proposal, you need:
> "Given Team A's cumulative stats at round r (e.g., kills, deaths, economy/equipment proxies, scoreline)"

| Feature | Available? | Source | Notes |
|---------|-----------|--------|-------|
| **Scoreline through round r** | ✅ YES | economy.csv | Count round winners 1 through r |
| **Equipment values** | ✅ YES | economy.csv | Direct columns |
| **Round winners** | ✅ YES | economy.csv | Direct columns |
| **Team kills/deaths (cumulative)** | ⚠️ PARTIAL | Derive from players.csv | Match-level only, must allocate |
| **Individual player round stats** | ❌ NO | N/A | Not in this dataset |

---

## What You Can Build

### ✅ Definitely Possible:

**1. Round-by-round scoreline:**
```python
# Count round winners through round 7
round_7_score_team1 = sum([1 for r in range(1, 8) if winner[f'{r}_winner'] == 1])
round_7_score_team2 = sum([1 for r in range(1, 8) if winner[f'{r}_winner'] == 2])
# Result: "After round 7: Team 1 leads 4-3"
```

**2. Equipment-based features:**
```python
# Average equipment through round 7
avg_equipment_t1 = mean([row[f'{r}_t1'] for r in range(1, 8)])

# Equipment differential
equip_diff_round_7 = row['7_t1'] - row['7_t2']

# Eco round detection
eco_rounds = [r for r in range(1, 8) if row[f'{r}_t1'] < 10000]
```

**3. Momentum features:**
```python
# Win streak going into round 7
recent_wins = [row[f'{r}_winner'] for r in range(5, 8)]  # Last 3 rounds

# Economy swing detection
economy_swing = row['7_t1'] - row['6_t1']
```

### ⚠️ Requires Approximation:

**Team-level kill/death stats:**
Since you only have match-level player stats, you could:

**Option 1: Use match-level as proxy**
```python
# Aggregate all players on team to get team K/D
team_kills = players[players['team'] == 'G2']['kills'].sum()
team_deaths = players[players['team'] == 'G2']['deaths'].sum()

# Use as static feature (doesn't change by round)
```

**Option 2: Linear interpolation (rough estimate)**
```python
# Assume kills accumulate roughly linearly
total_match_kills = 57
rounds_in_match = 30
estimated_kills_by_round_7 = (7/30) * total_match_kills ≈ 13.3 kills

# ⚠️ This is a ROUGH estimate!
```

### ❌ Not Possible:

- Exact player kills at round 7
- Player-level cumulative stats through round r
- Round-by-round K/D tracking

---

## Impact on Your Project

### Good News: Your Core Models Are Still Viable! ✅

Looking at your proposal, your primary features are:

1. **Scoreline through round r** → ✅ Available (compute from round winners)
2. **Equipment values** → ✅ Available directly (economy.csv)
3. **Combat aggregates (kills, deaths)** → ⚠️ Match-level only (can use as static)

**Your project can still work!** Here's how:

### Recommended Feature Set:

**Strong Features (directly available):**
- ✅ Cumulative round score (rounds won by each team through round r)
- ✅ Equipment value at round r for both teams
- ✅ Equipment differential
- ✅ Round win momentum (recent round winners)
- ✅ Eco round indicators
- ✅ Side (CT/T) performance from results.csv

**Weak/Static Features (match-level only):**
- ⚠️ Match-level team K/D (doesn't update by round, but can use as baseline)
- ⚠️ Match-level player ratings (from players.csv)

### Modified Feature Engineering:

Instead of:
```
❌ "Cumulative kills through round 7" (not available)
```

Use:
```
✅ "Scoreline through round 7" (3-4, team is down 1)
✅ "Equipment value at round 7" ($24,600)
✅ "Equipment differential at round 7" (+$5,000 advantage)
✅ "Recent momentum" (won last 2 of 3 rounds)
✅ "Eco round pattern" (eco'd on rounds 2, 3)
✅ "Match-level team strength" (static K/D ratio: 1.15)
```

---

## Recommendation for Your Project

### Option 1: Use What You Have (Recommended)

**Proceed with these features:**
1. Scoreline through round r (primary signal!)
2. Equipment values at round r
3. Round win patterns/momentum
4. Match-level player aggregates (as baseline strength)
5. CT/T side information

**Expected performance:**
- Scoreline is the STRONGEST predictor
- Equipment adds economic context
- Should achieve your success criteria (>50% accuracy early, improving by round)

### Option 2: Find Different Dataset

If you absolutely need round-by-round player stats, consider:

**Alternative: CSGO Demo Parsing**
- Parse .dem files directly
- Get frame-by-frame player positions, kills, deaths
- Much more complex, requires demo parsing tools (awpy, demoinfocs-golang)
- Example: ESTA dataset (https://github.com/pnxenopoulos/esta)

**Trade-off:**
- ✅ Get exact round-by-round player stats
- ❌ Much smaller dataset (~1,500 demos vs. 25,000 matches)
- ❌ Requires complex preprocessing
- ❌ Much more compute-intensive

---

## Final Assessment

### For Your CSCE768 Project:

**Verdict: The current dataset IS sufficient!** ✅

**Why:**
1. **Scoreline is the key feature** - and you have it!
2. Your proposal emphasizes "team-level cumulative statistics" - equipment + scoreline cover this
3. Academic papers you cited (Broms, Björklund, Rubin) also use similar granularity
4. Your evaluation focuses on "accuracy curves by round" - this works with scoreline + equipment
5. 600k samples is plenty for training all 5 models

**What to emphasize:**
- Round-by-round win probability based on scoreline + economy
- Economic strategy analysis (your Transformer can learn this!)
- Momentum and comeback probability
- CT/T side performance

**What to de-emphasize:**
- Individual player contributions by round (mention as future work)
- Exact combat stats by round (use match-level as proxy)

---

## Code Example: Building Your Features

```python
import pandas as pd
import numpy as np

def create_round_features(economy_row, round_num):
    """
    Create features for predicting match winner at round_num.

    economy_row: one row from economy.csv
    round_num: current round (1-30)
    """
    features = {}

    # 1. Scoreline through round_num
    team1_rounds = sum([1 for r in range(1, round_num+1)
                       if economy_row[f'{r}_winner'] == 1])
    team2_rounds = sum([1 for r in range(1, round_num+1)
                       if economy_row[f'{r}_winner'] == 2])

    features['team1_score'] = team1_rounds
    features['team2_score'] = team2_rounds
    features['score_diff'] = team1_rounds - team2_rounds

    # 2. Equipment values
    features['team1_equipment'] = economy_row[f'{round_num}_t1']
    features['team2_equipment'] = economy_row[f'{round_num}_t2']
    features['equipment_diff'] = (economy_row[f'{round_num}_t1'] -
                                  economy_row[f'{round_num}_t2'])

    # 3. Momentum (last 3 rounds)
    if round_num >= 3:
        recent_rounds = [economy_row[f'{r}_winner']
                        for r in range(round_num-2, round_num+1)]
        features['team1_recent_wins'] = sum([1 for w in recent_rounds if w == 1])

    # 4. Eco round indicators
    features['team1_eco'] = 1 if features['team1_equipment'] < 10000 else 0
    features['team2_eco'] = 1 if features['team2_equipment'] < 10000 else 0

    return features

# Example usage
economy = pd.read_csv('samples/economy.csv')
match = economy.iloc[0]

# Get features at round 7
features_r7 = create_round_features(match, round_num=7)
print(f"After round 7:")
print(f"  Score: {features_r7['team1_score']}-{features_r7['team2_score']}")
print(f"  Equipment: ${features_r7['team1_equipment']:,} vs ${features_r7['team2_equipment']:,}")
print(f"  Team 1 eco round: {features_r7['team1_eco']}")
```

---

## Summary

**Q: Do you have round-by-round player kills/deaths/assists?**
**A: No, only match-level player stats.**

**Q: Do you have round-by-round scoreline?**
**A: YES! You can compute it from round winners.**

**Q: Can you still do your project?**
**A: ABSOLUTELY YES! Focus on scoreline + equipment features. These are strong predictors and match your proposal's "team-level cumulative statistics" requirement.**

**Bottom line:** Your dataset supports your project goals. Proceed with confidence! 🚀
