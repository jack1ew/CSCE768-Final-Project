# Kaggle vs. ESTA: Which Dataset for Your Project?

## Quick Answer

**For your CSCE768 project: Use ESTA dataset**

**Why:** Your proposal emphasizes player-level features (kills, deaths, ADR), and ESTA has round-by-round player stats while Kaggle doesn't.

---

## Side-by-Side Comparison

| Feature | Kaggle Dataset | ESTA Dataset |
|---------|----------------|--------------|
| **Matches** | 5,210 unique matches | ~1,558 matches |
| **Round-by-round player stats** | ❌ NO | ✅ YES |
| **Player kills/deaths by round** | ❌ NO (match-level only) | ✅ YES |
| **Equipment values by round** | ✅ YES (team-level) | ✅ YES (player-level) |
| **Round winners** | ✅ YES | ✅ YES |
| **Sample size (team-level)** | ~312,600 | ~46,740 |
| **Sample size (player-level)** | ❌ Not possible | ~467,400 |
| **Storage** | 11 MB | ~5-10 GB |
| **Ready to use** | ✅ Downloaded | ⚠️ Need to download |
| **Time period** | 2015-2020 | 2021-2022 |
| **Data quality** | Aggregated stats | Frame-by-frame parsed |

---

## Your Proposal Requirements

From your proposal, you want these features:

### ✅ Required Features:

1. **"Scoreline through round r (rounds won)"**
   - Kaggle: ✅ YES
   - ESTA: ✅ YES

2. **"Combat aggregates: total kills, total deaths, kill–death differential"**
   - Kaggle: ⚠️ Match-level only
   - ESTA: ✅ Round-by-round cumulative

3. **"Economy/equipment proxies: cumulative equipment value spent/held"**
   - Kaggle: ✅ YES (team-level)
   - ESTA: ✅ YES (player-level)

4. **"Headshot count/headshot %"**
   - Kaggle: ⚠️ Match-level only
   - ESTA: ✅ Round-by-round

5. **"Objectives: bomb plants/defuses"**
   - Kaggle: ❌ NO
   - ESTA: ✅ YES

6. **"Survivorship: average surviving players"**
   - Kaggle: ❌ NO
   - ESTA: ✅ YES (can compute)

### Verdict: **ESTA matches your proposal better** ✅

---

## Sample Size Comparison

### Your Proposal States:
> "~10,000 matches... provides ≈600,000 labeled team-round examples"

### Actual Sample Sizes:

**Kaggle (5,210 matches):**
- Team-level: 5,210 × 30 rounds × 2 teams = **~312,600 samples**
- ❌ Cannot do player-level (no round-by-round player data)

**ESTA (1,558 matches):**
- Team-level: 1,558 × 30 rounds × 2 teams = **~46,740 samples**
- Player-level: 1,558 × 30 rounds × 10 players = **~467,400 samples**
- ✅ Can do both team-level and player-level

### Training Sufficiency:

| Model | Needs ~samples | Kaggle OK? | ESTA OK? |
|-------|----------------|------------|----------|
| Logistic Regression | ~10k | ✅ YES (312k) | ✅ YES (467k) |
| Random Forest | ~50k | ✅ YES (312k) | ✅ YES (467k) |
| k-NN | ~10k | ✅ YES (312k) | ✅ YES (467k) |
| MLP | ~100k+ | ✅ YES (312k) | ✅ YES (467k) |
| Transformer | ~100k+ | ✅ YES (312k) | ✅ YES (467k) |

**Both datasets have sufficient samples for all 5 models!** ✅

---

## Pros & Cons

### Kaggle Dataset

**Pros:**
- ✅ Already downloaded (ready now!)
- ✅ Larger dataset (5,210 matches)
- ✅ Simpler to work with
- ✅ Fast to process (smaller files)
- ✅ Team-level features sufficient for baseline

**Cons:**
- ❌ No round-by-round player statistics
- ❌ Missing features from your proposal (bomb plants, survivorship)
- ❌ Cannot build player-level models
- ❌ Match-level player stats only (not round-cumulative)
- ❌ Less rich for feature engineering

**Best for:**
- Quick baseline models
- Team-level win probability
- If you're short on time
- Simpler project scope

---

### ESTA Dataset

**Pros:**
- ✅ Round-by-round player statistics (exactly what you asked for!)
- ✅ All features from your proposal available
- ✅ Player-level modeling possible
- ✅ More granular data = better models
- ✅ Bomb plants, defuses, survivorship all available
- ✅ Pre-parsed and clean
- ✅ Can compute cumulative stats (kills through round 7)

**Cons:**
- ❌ Smaller dataset (1,558 matches)
- ❌ Need to download (~5-10 GB)
- ❌ More complex data structure
- ❌ Larger storage requirements
- ❌ Slower to process

**Best for:**
- Complete feature set from proposal
- Player-level analysis
- Richer feature engineering
- More sophisticated models
- Better matches your proposal

---

## For Your CSCE768 Project

### Your Proposal Emphasizes:

**Quote from your proposal:**
> "Combat aggregates: total kills, total deaths, kill–death differential; headshot count/headshot % (if available)... Economy/equipment proxies... Objectives: bomb plants/defuses... Survivorship: average surviving players"

**These features require round-by-round player data!**

### Academic Precedent:

Your cited papers (Broms, Björklund, Rubin) use:
- Player-level features ✅
- Round-by-round statistics ✅
- Cumulative combat stats ✅

**They likely used demo-parsed data (like ESTA), not aggregated data (like Kaggle).**

---

## My Recommendation

### **Use ESTA Dataset** ✅

**Reasons:**

1. **Matches your proposal** - You specifically list player-level features that require round-by-round data
2. **More complete** - Has all features you proposed (bomb plants, survivorship, etc.)
3. **Better models** - Richer features = better predictions
4. **Sufficient samples** - 467k player-round samples is plenty for all 5 models
5. **Academic rigor** - Aligns with cited papers' methodology
6. **Stronger results** - Player-level features will improve accuracy

### Sample Size Is NOT an Issue:

- Your proposal estimated 600k samples
- ESTA gives you 467k samples (78% of target)
- This is MORE than enough for all models
- Quality > Quantity (ESTA has richer features)

### Timeline:

**Week 1:** Download ESTA, explore data structure
**Week 2:** Build preprocessing pipeline (cumulative stats)
**Week 3:** Train baseline models (Logistic, RF, kNN)
**Week 4:** Train neural models (MLP, Transformer)
**Week 5:** Evaluation, comparison, write-up

**Totally feasible!** ✅

---

## Alternative: Hybrid Approach

If you're concerned about ESTA complexity:

**Option: Start with Kaggle, Switch to ESTA**

1. **Week 1:** Use Kaggle data to build pipeline
   - Implement round-by-round framework
   - Test with team-level features
   - Verify models train correctly

2. **Week 2:** Download and integrate ESTA
   - Add player-level features
   - Retrain models
   - Compare team-only vs. team+player

**Benefits:**
- ✅ Risk mitigation (have working baseline)
- ✅ Great ablation study (with/without player features)
- ✅ Shows incremental value of player data

**Downside:**
- ⚠️ More work (two datasets)
- ⚠️ Less time for each

---

## Decision Framework

**Choose ESTA if:**
- ✅ You want to fully implement your proposal
- ✅ You have 4+ weeks
- ✅ You're comfortable with data preprocessing
- ✅ You want stronger models
- ✅ You want player-level insights

**Choose Kaggle if:**
- ✅ You have <3 weeks
- ✅ You want simplicity
- ✅ You're okay modifying proposal scope
- ✅ Team-level modeling is sufficient
- ✅ You prioritize speed over completeness

**Choose Hybrid if:**
- ✅ You want both datasets' benefits
- ✅ You have time for two implementations
- ✅ You want ablation study (great for paper!)
- ✅ Risk-averse approach

---

## Bottom Line

**For your specific proposal: ESTA is the better choice** ✅

Your proposal explicitly requests:
- "Combat aggregates: total kills, total deaths, kill–death differential"
- "Headshot count/headshot %"
- "Objectives: bomb plants/defuses"
- "Survivorship: average surviving players"

**Kaggle CANNOT provide these at round-by-round granularity.**
**ESTA CAN provide all of these.** ✅

### The 467k samples from ESTA is sufficient:
- Logistic Regression: Excellent
- Random Forest: Excellent
- k-NN: Excellent
- MLP: Very good
- Transformer: Good (with proper regularization)

**Recommendation: Download ESTA and use it as your primary dataset.**

---

## Next Steps

If you choose ESTA:

1. **Install awpy:**
```bash
source .venv/bin/activate
pip install awpy
```

2. **Visit ESTA repo:**
```
https://github.com/pnxenopoulos/esta
```

3. **Download dataset** (follow their instructions)

4. **Explore structure:**
```python
import json
with open('esta_match.json') as f:
    match = json.load(f)
print(match['rounds'][0])  # See round 1 data
```

Want me to help you download ESTA now?
