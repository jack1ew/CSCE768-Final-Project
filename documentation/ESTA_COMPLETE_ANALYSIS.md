# ESTA Dataset Complete Analysis

## Your Questions Answered

### Q1: Does ESTA have team-level stats (economy, scoreline)?

**YES! ✅ ESTA has BOTH player-level AND team-level data**

Based on the ESTA documentation:

#### Team-Level Features Available:
- ✅ **Team equipment value** (`teamEqVal`)
- ✅ **Team side** (T/CT)
- ✅ **Alive player count** (per team)
- ✅ **Total utility count** (grenades, etc.)
- ✅ **Round scores** (implicit from round winners)
- ✅ **Round winners** (which team won each round)

#### Player-Level Features Available:
- ✅ **Position** (x, y, z coordinates)
- ✅ **Health, armor**
- ✅ **Equipment value** (per player)
- ✅ **Cash** (per player)
- ✅ **Inventory** (weapons, ammo)
- ✅ **Kills, deaths, assists**
- ✅ **Damage dealt**
- ✅ **Headshot status**

#### Actions Tracked:
- ✅ Kills (weapon, distance, headshot, wallbang)
- ✅ Damage events
- ✅ Bomb plants/defuses
- ✅ Grenade throws
- ✅ Weapon fires
- ✅ Flash events

**Verdict: ESTA has EVERYTHING you need!** ✅

---

### Q2: CS:GO or CS2 matches?

**ESTA contains CS:GO matches ONLY (not CS2)** 🎮

**Time Period:** January 2021 - May 2022
**Game Version:** Counter-Strike: Global Offensive (CS:GO)
**Reason:** CS2 wasn't released until September 2023

**Dataset breakdown:**
- 878 online tournament demos (Jan 2021 - May 2022)
- 680 LAN tournament demos (Jul 2021 - May 2022)
- **Total: 1,558 CS:GO professional matches**

---

### Q3: Which is better for your project - CS:GO or CS2?

**CS:GO data is BETTER for your academic project** ✅

**Reasons:**

1. **More data available**
   - CS:GO: 7+ years of data (2015-2023)
   - CS2: Only 1.5 years of data (Sep 2023-now)

2. **Established research precedent**
   - Your cited papers use CS:GO data
   - Academic community has validated CS:GO datasets
   - Easier to compare with prior work

3. **Game stability**
   - CS:GO rules/economy stable for years
   - CS2 still receiving balance patches
   - CS:GO = consistent training data

4. **Your proposal cites CS:GO**
   - Your proposal: "~10,000 CS:GO matches from Kaggle/HLTV"
   - You reference CS:GO's MR15 format
   - Reviewers expect CS:GO data

---

### Q4: Will CS:GO models work on CS2 matches?

**MOSTLY YES, but with caveats** ⚠️

#### What's the Same (Model Still Works):
- ✅ **Core mechanics:** 5v5, bomb plant/defuse, round-based
- ✅ **Economy system:** Similar (buy rounds, eco rounds, force buys)
- ✅ **Win conditions:** Eliminate enemies or complete objective
- ✅ **Round structure:** Teams switch sides at half
- ✅ **Equipment tiers:** Pistols, SMGs, rifles, AWP still exist

#### What Changed (Model May Need Adjustment):
- ⚠️ **MR12 vs MR15:** CS2 uses 12-round halves (24 max) vs CS:GO's 15 (30 max)
- ⚠️ **Economy tweaks:** Some buy costs changed, loss bonus adjusted
- ⚠️ **Weapon balance:** AWP, M4A1-S costs changed
- ⚠️ **Map updates:** Some maps redesigned (e.g., Overpass, Inferno)

#### Transfer Learning Assessment:

**High transferability features:**
- ✅ Scoreline patterns (team ahead/behind)
- ✅ Equipment value differentials
- ✅ Momentum (recent round wins)
- ✅ Player K/D ratios
- ✅ Side (CT/T) advantages

**Low transferability features:**
- ⚠️ Specific equipment values (thresholds changed)
- ⚠️ Round 15/16 behaviors (different in MR12)
- ⚠️ Exact weapon costs

#### Expected Performance:

**Training on CS:GO → Testing on CS2:**
- **Accuracy drop:** ~5-10% (estimated)
- **Calibration:** May need recalibration
- **Core patterns:** Still valid (scoreline, momentum)

**Example:**
```
CS:GO model trained to predict:
"Team down 3-10 at round 13 has 15% win probability"

CS2 equivalent:
"Team down 2-7 at round 9 has ~18% win probability"
(Different because fewer rounds total in MR12)
```

---

## Recommendation for Your Project

### **Use CS:GO data (ESTA or Kaggle)** ✅

**Why:**

1. **Matches your proposal** - You cited CS:GO explicitly
2. **More data available** - Larger datasets
3. **Academic standard** - Your cited papers use CS:GO
4. **Stable game state** - No ongoing balance changes
5. **Sufficient for graduation** - Models CS:GO ≠ Models CS2 is okay for academic work

### Your Proposal Already Acknowledges This:

**From your proposal:**
> "We note CS2's MR12 (24 rounds) for future extension"

**This shows you're aware of CS2 but focusing on CS:GO!** ✅

---

## If You Want CS2 Compatibility

### Option 1: Train on CS:GO, Mention CS2 in Future Work

**In your paper:**
- "We train on CS:GO data following established literature"
- "Future work: Adapt models to CS2's MR12 format"
- "Transfer learning from CS:GO → CS2 is an open research question"

**Pros:**
- ✅ Aligns with proposal
- ✅ Standard academic approach
- ✅ More data available
- ✅ Easier comparison with prior work

---

### Option 2: Fine-tune for CS2 (Advanced)

**Approach:**
1. Train base model on CS:GO data (ESTA - 1,558 matches)
2. Collect small CS2 dataset (100-200 matches)
3. Fine-tune model on CS2 data
4. Compare CS:GO-only vs. CS:GO+CS2 fine-tuned

**Pros:**
- ✅ Models work on latest game
- ✅ Great ablation study
- ✅ More impactful for esports industry

**Cons:**
- ❌ Need to collect/parse CS2 demos (extra work)
- ❌ CS2 data harder to find
- ❌ More complex project scope

---

### Option 3: Normalize for MR12/MR15 Differences

**Feature engineering approach:**

Instead of absolute round numbers:
```python
# Bad (game-specific)
features['round_number'] = 13

# Good (normalized)
features['round_progress'] = 13 / 30  # 0.43 (43% through match)
features['rounds_remaining'] = 30 - 13  # 17
```

Instead of absolute scores:
```python
# Bad (game-specific)
features['team_score'] = 10

# Good (normalized)
features['score_percentage'] = 10 / 30  # 0.33 (33% of rounds won)
features['score_diff_normalized'] = (10 - 3) / 30  # 0.23
```

**This makes your model more transferable to CS2!** ✅

---

## Data Structure in ESTA

Based on documentation, here's what you can extract:

### Round-Level Data Structure:

```python
{
  "match_id": "...",
  "map": "de_dust2",
  "rounds": [
    {
      "round_num": 1,
      "start_tick": 1000,
      "end_tick": 15000,

      # TEAM-LEVEL (What you asked about!)
      "t_side": {
        "team_name": "Natus Vincere",
        "team_eq_val": 4350,  # Total equipment value
        "alive_players": 5,
        "total_utility": 10
      },
      "ct_side": {
        "team_name": "G2",
        "team_eq_val": 4750,
        "alive_players": 5,
        "total_utility": 8
      },

      "round_winner": "CT",  # Who won this round
      "round_end_reason": "Elimination",  # Or "BombDefused", "BombExploded"

      # PLAYER-LEVEL
      "players": [
        {
          "player_name": "s1mple",
          "steam_id": "...",
          "team": "Natus Vincere",
          "side": "T",

          # Stats for this round
          "kills": 2,
          "deaths": 0,
          "assists": 1,
          "damage": 250,
          "headshot_kills": 1,

          # Equipment
          "equipment_value": 4750,
          "cash": 800,
          "weapons": ["ak47", "glock", "flashbang", "smoke"],

          # End-of-round state
          "survived": true,
          "health": 85,
          "armor": 50
        },
        // ... 9 more players
      ],

      # EVENTS during this round
      "kills": [
        {
          "tick": 12000,
          "attacker": "s1mple",
          "victim": "NiKo",
          "weapon": "AK-47",
          "headshot": true,
          "distance": 25.5
        },
        // ... more kills
      ],

      "bomb_events": [
        {
          "tick": 14000,
          "player": "Perfecto",
          "event_type": "plant",
          "site": "A"
        }
      ],

      "damages": [...],
      "grenades": [...],
      "weapon_fires": [...]
    },
    // ... rounds 2-30
  ],

  # MATCH-LEVEL
  "match_winner": "CT",
  "final_score": {
    "t_rounds": 13,
    "ct_rounds": 16
  }
}
```

---

## What You CAN Compute from ESTA:

### ✅ Team-Level Features (Your Question!):

1. **Scoreline through round r:**
```python
rounds_won_t = sum([1 for round in rounds[:r] if round['round_winner'] == 'T'])
rounds_won_ct = sum([1 for round in rounds[:r] if round['round_winner'] == 'CT'])
scoreline = f"{rounds_won_t}-{rounds_won_ct}"  # e.g., "4-3" after round 7
```

2. **Team equipment value at round r:**
```python
team_eq_val = rounds[r]['t_side']['team_eq_val']  # e.g., $24,600
```

3. **Equipment differential:**
```python
eq_diff = rounds[r]['t_side']['team_eq_val'] - rounds[r]['ct_side']['team_eq_val']
```

4. **Eco round detection:**
```python
is_eco = rounds[r]['t_side']['team_eq_val'] < 10000
```

### ✅ Player-Level Features:

5. **Cumulative player kills through round r:**
```python
player_kills = sum([
    len([k for k in rounds[i]['kills'] if k['attacker'] == 's1mple'])
    for i in range(r)
])
```

6. **Team K/D ratio through round r:**
```python
team_kills = sum([len(rounds[i]['kills']) for i in range(r) if player.team == 'Na\'Vi'])
team_deaths = sum([len(rounds[i]['kills']) for i in range(r) if victim.team == 'Na\'Vi'])
kd_ratio = team_kills / team_deaths
```

7. **Player ADR through round r:**
```python
total_damage = sum([d['damage'] for i in range(r) for d in rounds[i]['damages'] if d['attacker'] == 's1mple'])
adr = total_damage / r
```

---

## Summary

### Your Questions:

**Q: Does ESTA have team-level stats like economy and scoreline?**
**A: YES! ✅** Team equipment values, round winners, alive counts all available.

**Q: CS:GO or CS2?**
**A: CS:GO only** (ESTA is from 2021-2022, before CS2 existed)

**Q: Which is better for your project?**
**A: CS:GO** - matches your proposal, more data, academic standard

**Q: Will CS:GO models work on CS2?**
**A: Mostly yes** (~5-10% accuracy drop expected), core patterns transfer

---

## Final Recommendation

**Use ESTA dataset with CS:GO data** ✅

**Why:**
1. ✅ Has BOTH team-level (economy, scoreline) AND player-level stats
2. ✅ CS:GO aligns with your proposal
3. ✅ More data available than CS2
4. ✅ Standard for academic work
5. ✅ ~467k samples sufficient for all 5 models
6. ✅ Can mention CS2 transfer learning as "future work"

**You get everything you need:**
- Round-by-round scoreline
- Team equipment values
- Player kills/deaths/assists by round
- Bomb plants/defuses
- Complete feature set from your proposal

**Next step:** Download ESTA and start preprocessing!

Want me to help you download it now?
