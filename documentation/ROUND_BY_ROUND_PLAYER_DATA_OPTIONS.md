# How to Get Round-by-Round Player Statistics

## What You Need
**Round-by-round, player-by-player kill/death/assist data** (e.g., "Player X has 3 kills, 2 deaths, 1 assist after round 7")

This requires parsing **CS:GO demo files (.dem)** which contain frame-by-frame game data.

---

## Option 1: ESTA Dataset (Recommended - Ready to Use)

### Overview
The **Esports Trajectories and Actions (ESTA)** dataset provides parsed CS:GO demo data with detailed round-by-round statistics.

**What it has:**
- ✅ Round-by-round player kills, deaths, assists
- ✅ Player positions and movements
- ✅ Weapon information
- ✅ Bomb plant/defuse events
- ✅ Pre-parsed and ready to use

**Dataset size:**
- 1,558 professional demos
- 878 online tournament demos (Jan 2021 - May 2022)
- 680 LAN tournament demos (Jul 2021 - May 2022)

### How to Get It:

**1. Visit the ESTA GitHub repository:**
```
https://github.com/pnxenopoulos/esta
```

**2. Download options:**

**Option A: Use their Python package (easiest)**
```bash
pip install awpy
```

Then download data:
```python
from awpy.data import AWPY_VERSIONS

# Download ESTA dataset
# Instructions at: https://github.com/pnxenopoulos/esta
```

**Option B: Direct download from their links**
- Check the GitHub repo for Google Drive/Kaggle links
- Files are in JSON/Parquet format

**3. Data structure:**
```json
{
  "match_id": "...",
  "map": "de_dust2",
  "rounds": [
    {
      "round_num": 1,
      "players": [
        {
          "player_name": "s1mple",
          "team": "Natus Vincere",
          "kills": 2,
          "deaths": 0,
          "assists": 1,
          "damage": 250,
          "equipment_value": 4750
        },
        // ... more players
      ],
      "winner": "CT",
      "round_end_reason": "BombDefused"
    },
    // ... more rounds
  ]
}
```

### Pros:
- ✅ Pre-parsed and clean
- ✅ Ready for ML use
- ✅ Round-by-round player stats included
- ✅ High-quality professional matches
- ✅ Academic paper backing (trustworthy)

### Cons:
- ❌ Smaller dataset (~1,558 matches vs. 25,000)
- ❌ Only 2021-2022 (more recent, but shorter time span)
- ❌ Larger file sizes (more detailed data)

### For Your Project:
**This is probably your best option!** It has exactly what you need with less work.

---

## Option 2: Parse Demo Files Yourself

### Overview
Download raw `.dem` files and parse them yourself using demo parsing tools.

### Tools Available:

#### A. **awpy** (Python - Recommended)
```bash
pip install awpy
```

**Usage:**
```python
from awpy.parser import DemoParser

# Parse a demo file
parser = DemoParser(demofile="match.dem", parse_rate=128)
data = parser.parse()

# Extract round-by-round stats
for round_num, round_data in enumerate(data['rounds']):
    print(f"Round {round_num + 1}:")
    for player in round_data['kills']:
        print(f"  {player['attacker_name']} killed {player['victim_name']}")
```

**What you get:**
- Round-by-round events (kills, deaths, bomb plants, etc.)
- Player positions every 128 ticks (~1 second)
- Equipment purchases
- Damage dealt

#### B. **demoinfocs-golang** (Go language)
```bash
go get -u github.com/markus-wa/demoinfocs-golang/v3
```

Faster than Python, but requires Go programming.

#### C. **csgo-parser** (Node.js)
```bash
npm install csgo-parser
```

JavaScript-based parser.

### Where to Get Demo Files:

**1. HLTV.org**
- Download professional match demos
- URL: https://www.hltv.org/matches
- Look for "GOTV Demo" download links
- Free, but manual download

**2. Valve Matchmaking**
- Your own competitive matches
- Download from in-game or Steam
- Personal matches only

**3. Demo download scripts**
```python
# Example: Scrape HLTV for demo links
import requests
from bs4 import BeautifulSoup

def get_demo_links(match_id):
    url = f"https://www.hltv.org/matches/{match_id}/"
    # ... scraping logic
    # Returns demo download URL
```

### Pros:
- ✅ Maximum flexibility
- ✅ Access to ANY match with demos
- ✅ Can get very large dataset
- ✅ Complete control over parsing

### Cons:
- ❌ Very time-consuming (parsing is slow)
- ❌ Large storage requirements (demos are big: 50-200 MB each)
- ❌ Complex preprocessing pipeline
- ❌ Requires significant dev time (1-2 weeks setup)

### For Your Project:
**Only do this if:** You need more than 1,558 matches and have time to build parsing infrastructure.

---

## Option 3: Hybrid Approach (Recommended for Your Project)

### Strategy:
Use both datasets to maximize your analysis!

**For training (bulk of data):**
- Use current Kaggle dataset (25k matches)
- Features: scoreline, equipment values, momentum
- Already downloaded and ready!

**For validation/enrichment:**
- Download ESTA dataset (~1,558 matches)
- Add player-level features to a subset
- Compare model performance with/without player stats

### Implementation:

**Phase 1: Baseline Models (Week 1-2)**
```python
# Use Kaggle data (already have it!)
# Features: scoreline, equipment, momentum
# Train all 5 models (Logistic, RF, kNN, MLP, Transformer)
```

**Phase 2: Enhanced Models (Week 3-4)**
```python
# Download ESTA dataset
# Add player features: cumulative K/D by round
# Retrain models on enriched features
# Compare: does player data improve predictions?
```

### Benefits:
- ✅ Start immediately with existing data
- ✅ Don't block on getting new data
- ✅ Add player data as enhancement, not requirement
- ✅ Great for ablation study ("with vs. without player stats")
- ✅ More robust project (two datasets = better validation)

---

## Comparison Table

| Dataset | Matches | Round Player Stats | Ready to Use | Storage | Time to Setup |
|---------|---------|-------------------|--------------|---------|---------------|
| **Kaggle (current)** | 25,000 | ❌ No | ✅ Downloaded | 100 MB | ✅ Done |
| **ESTA** | 1,558 | ✅ Yes | ⚠️ Needs download | 5-10 GB | ~2 hours |
| **Parse demos yourself** | Unlimited | ✅ Yes | ❌ Need to build | 50+ GB | ~1-2 weeks |

---

## Detailed: How to Get ESTA Dataset

### Step 1: Install awpy
```bash
source .venv/bin/activate
pip install awpy
```

### Step 2: Download ESTA data

Check the ESTA repository for download links:
```bash
# Visit: https://github.com/pnxenopoulos/esta

# They provide links to:
# - Google Drive
# - Kaggle dataset
# - Direct download
```

### Step 3: Load and explore
```python
import pandas as pd
import json

# ESTA data is typically in JSON or Parquet format
# Example structure:
with open('esta_match.json') as f:
    match = json.load(f)

# Extract round 7 stats
round_7 = match['rounds'][6]  # 0-indexed
for player in round_7['players']:
    print(f"{player['name']}: {player['kills']}K {player['deaths']}D")
```

### Step 4: Create feature dataset
```python
def extract_round_features(match_data, round_num):
    """Extract cumulative player stats through round_num"""
    features = []

    for round_idx in range(round_num):
        round_data = match_data['rounds'][round_idx]

        for player in round_data['players']:
            # Cumulative kills through this round
            cumulative_kills = sum([
                p['kills'] for r in match_data['rounds'][:round_idx+1]
                for p in r['players'] if p['steamID'] == player['steamID']
            ])

            features.append({
                'player': player['name'],
                'round': round_idx + 1,
                'cumulative_kills': cumulative_kills,
                # ... more features
            })

    return features
```

---

## Recommendation for Your CSCE768 Project

### Timeline-Conscious Approach:

**If you have 4+ weeks:**
- ✅ Download ESTA dataset
- ✅ Build features with player stats
- ✅ Compare models with/without player data
- ✅ More complete analysis

**If you have 2-3 weeks:**
- ✅ Use Kaggle data (already have it!)
- ✅ Focus on scoreline + equipment features
- ⚠️ Download ESTA as "future work" if time permits
- ✅ Still achieves all proposal goals

**If you have < 2 weeks:**
- ✅ Stick with Kaggle data
- ❌ Don't add complexity
- ✅ Focus on model comparison and evaluation

---

## My Recommendation

**Best approach for your project: Hybrid**

### Week 1-2: Use Kaggle Data
1. Build preprocessing pipeline
2. Implement feature engineering (scoreline, equipment)
3. Train baseline models (Logistic, RF, kNN)
4. **Deliverable:** Working models with accuracy curves

### Week 2-3: Add Neural Models
5. Implement MLP
6. Implement Transformer with sequence modeling
7. **Deliverable:** All 5 models trained and evaluated

### Week 3-4: Enhancement (if time)
8. Download ESTA dataset
9. Add player features to subset of matches
10. Ablation study: with vs. without player stats
11. **Deliverable:** Enhanced analysis showing player stat impact

### Benefits:
- ✅ Not blocked on getting new data
- ✅ Achieves all proposal requirements with Kaggle data
- ✅ ESTA becomes a "value-add" not a requirement
- ✅ Great story: "We tested with and without player data"
- ✅ More robust conclusions

---

## Quick Start: Download ESTA Now

If you want to download ESTA dataset now while you work on preprocessing:

**Check these resources:**

1. **ESTA GitHub:** https://github.com/pnxenopoulos/esta
2. **ESTA Paper:** https://arxiv.org/abs/2303.10506
3. **awpy Documentation:** https://awpy.readthedocs.io/

**Expected commands:**
```bash
# Install awpy
pip install awpy

# Download ESTA (check repo for exact command)
# They may provide a download script or links
```

---

## Storage Requirements

### With ESTA:
- Kaggle data: ~100 MB
- ESTA data: ~5-10 GB (parsed)
- Raw demos (if parsing): ~50-100 GB
- **Total: ~10-15 GB** (with ESTA)

You have 50 GB available, so you're fine! ✅

---

## Example: What You Can Do With ESTA

```python
# Load ESTA data
match = load_esta_match('match_id_123.json')

# Extract player stats through round 7
for player in match['teams']['team1']['players']:
    kills_r7 = sum([
        r['player_stats'][player['steamID']]['kills']
        for r in match['rounds'][:7]
    ])
    deaths_r7 = sum([
        r['player_stats'][player['steamID']]['deaths']
        for r in match['rounds'][:7]
    ])

    print(f"{player['name']}: {kills_r7}K {deaths_r7}D through round 7")

# Result:
# s1mple: 12K 3D through round 7
# electronic: 8K 5D through round 7
# ...
```

**This is EXACTLY what you're asking for!** ✅

---

## Summary

**To get round-by-round player statistics:**

1. **Easiest:** Download ESTA dataset (~2 hours setup)
2. **Most flexible:** Parse demos yourself (~1-2 weeks setup)
3. **Pragmatic:** Use Kaggle now, add ESTA later (hybrid)

**My recommendation:** Start with Kaggle (you have it!), download ESTA in parallel, integrate if time permits.

**This gives you:**
- ✅ Immediate progress on project
- ✅ Fallback if ESTA is complex
- ✅ Enhanced analysis if ESTA works out
- ✅ Robust validation across datasets

Want me to help you download ESTA dataset now?
