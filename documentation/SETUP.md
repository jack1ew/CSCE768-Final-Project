# CSCE768 Final Project - Setup Instructions

**Project:** Round-by-Round Win Probability Modeling for Counter-Strike Matches
**Team Members:** Devon Goshorn, Jackie Wang
**Dataset:** ESTA (Esports Trajectories and Actions)

---

## Quick Start (5 Minutes)

```bash
# 1. Clone the repository
git clone <repository-url>
cd CSCE768-Final-Project

# 2. Run setup script
./setup_esta.sh

# 3. Download ESTA dataset (see below)

# 4. Verify installation
python scripts/verify_esta.py

# 5. Start developing!
```

---

## Important Notes for Team Members

### ⚠️ Data is NOT in GitHub Repository

**Why?**
- ESTA dataset is ~5-10 GB (too large for GitHub)
- GitHub has 100 MB file size limit
- Each team member downloads data locally

**What this means:**
- ✅ Source code (models, scripts) → Committed to GitHub
- ❌ Data files → Stored locally only
- ✅ Model architectures → Committed to GitHub
- ❌ Trained model weights → Stored locally only

---

## Directory Structure

```
CSCE768-Final-Project/
│
├── data/                       # ❌ NOT in GitHub (.gitignore)
│   ├── esta/
│   │   ├── raw/               # Download ESTA files here
│   │   └── parsed/            # Parsed match data
│   └── processed/             # Preprocessed training features
│
├── models/                     # ⚠️ Partial in GitHub
│   ├── architectures/         # ✅ Model code (in GitHub)
│   ├── checkpoints/           # ❌ Model weights (NOT in GitHub)
│   └── saved_models/          # ❌ Trained models (NOT in GitHub)
│
├── scripts/                    # ✅ All scripts in GitHub
│   ├── verify_esta.py
│   ├── parse_esta.py
│   ├── preprocess_data.py
│   └── train_*.py
│
├── notebooks/                  # ✅ Jupyter notebooks in GitHub
│   └── exploratory_analysis.ipynb
│
├── results/                    # ⚠️ Summaries only in GitHub
│   └── accuracy_curves.png    # ✅ Small result files
│
├── logs/                       # ❌ NOT in GitHub
│
├── .gitignore                  # ✅ Tells Git what to ignore
├── SETUP.md                    # ✅ This file
├── setup_esta.sh               # ✅ Setup script
├── README.md                   # ✅ Project overview
└── requirements.txt            # ✅ Python dependencies
```

---

## Detailed Setup Instructions

### Prerequisites

- **Python 3.8+** (preferably 3.10+)
- **10-15 GB free disk space** (for ESTA dataset)
- **Good internet connection** (for downloading 5-10 GB)
- **Git** installed

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd CSCE768-Final-Project
```

### Step 2: Run Setup Script

```bash
./setup_esta.sh
```

This script will:
- Create directory structure
- Set up Python virtual environment
- Install required packages (awpy, pandas, torch, etc.)
- Create verification script

### Step 3: Download ESTA Dataset

**ESTA dataset must be downloaded manually:**

1. **Visit ESTA GitHub:**
   ```
   https://github.com/pnxenopoulos/esta
   ```

2. **Find download links** (check their README)
   - They provide Google Drive or Kaggle links
   - Dataset is ~5-10 GB compressed

3. **Download options:**

   **Option A: Pre-parsed JSON** (Recommended)
   - Download parsed JSON files
   - Extract to: `data/esta/raw/`
   - Faster to work with

   **Option B: Raw demo files**
   - Download .dem files
   - Extract to: `data/esta/raw/`
   - Need to parse them (slower)

4. **Extract files:**
   ```bash
   # If you downloaded a ZIP/TAR
   cd data/esta/raw/
   unzip esta_dataset.zip
   # OR
   tar -xzf esta_dataset.tar.gz
   ```

### Step 4: Verify Installation

```bash
source .venv/bin/activate
python scripts/verify_esta.py
```

**Expected output:**
```
======================================================================
ESTA Dataset Installation Verification
======================================================================

1. Checking directory structure...
   ✓ data/esta/raw
   ✓ data/esta/parsed
   ✓ data/processed
   ✓ models/checkpoints
   ✓ results

2. Checking for ESTA data...
   ✓ Found 1558 .json files

3. Checking Python packages...
   ✓ awpy
   ✓ pandas
   ✓ numpy
   ✓ sklearn
   ✓ torch

======================================================================
✓ Installation verified successfully!
======================================================================
```

### Step 5: Parse Data (if needed)

If you downloaded raw .dem files:

```bash
source .venv/bin/activate
python scripts/parse_esta.py
```

This converts .dem files to JSON (takes 1-2 hours for 1,558 demos).

### Step 6: Preprocess Data

```bash
python scripts/preprocess_data.py
```

This creates training-ready features from parsed demos.

---

## Working with the Repository

### Before You Start Coding

```bash
# 1. Pull latest changes
git pull origin main

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Install any new dependencies
pip install -r requirements.txt
```

### Adding Your Code

```bash
# 1. Create a new branch for your feature
git checkout -b feature/logistic-regression

# 2. Write your code
# models/architectures/logistic_regression.py
# scripts/train_logistic.py

# 3. Test your code
python scripts/train_logistic.py

# 4. Add and commit (only code, not data!)
git add models/architectures/logistic_regression.py
git add scripts/train_logistic.py
git commit -m "Add logistic regression model"

# 5. Push to GitHub
git push origin feature/logistic-regression

# 6. Create Pull Request on GitHub
```

### What to Commit vs. Not Commit

**✅ DO commit:**
- Python scripts (`.py` files)
- Model architecture code
- Jupyter notebooks (`.ipynb`)
- Small config files
- Documentation (`.md` files)
- Requirements (`requirements.txt`)
- Small result summaries (CSV < 1 MB)
- Plots/figures (PNG, PDF)

**❌ DON'T commit:**
- Data files (`data/` directory)
- Trained model weights (`.pth`, `.h5`, `.pkl`)
- Large CSV files (> 1 MB)
- Log files
- Virtual environment (`.venv/`)
- Temporary files (`.tmp`, `__pycache__/`)

**The `.gitignore` file handles this automatically!** ✅

---

## Python Environment Management

### Activating Environment

**Every time you work on the project:**

```bash
source .venv/bin/activate
```

**You'll see:**
```bash
(.venv) your-username@computer:~/CSCE768-Final-Project$
```

### Installing New Packages

```bash
# Install a package
pip install some-package

# Update requirements.txt
pip freeze > requirements.txt

# Commit requirements.txt
git add requirements.txt
git commit -m "Add some-package dependency"
```

### Deactivating Environment

```bash
deactivate
```

---

## Troubleshooting

### "Data directory not found"

**Problem:** You see errors about missing `data/esta/raw/`

**Solution:**
1. Download ESTA dataset (see Step 3 above)
2. Extract files to `data/esta/raw/`
3. Run `python scripts/verify_esta.py`

---

### "awpy not installed"

**Problem:** Import errors for `awpy` or other packages

**Solution:**
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

---

### "Git says data files are too large"

**Problem:** Git error when trying to push data files

**Solution:** You're trying to commit data files (don't do this!)
```bash
# Remove data files from staging
git rm --cached data/esta/raw/*.json

# Check .gitignore is working
git status  # Should NOT show data files
```

---

### "My teammate's code doesn't work"

**Problem:** Code works on their machine but not yours

**Solution:**
1. Make sure you're on the same branch
2. Pull latest changes: `git pull`
3. Update dependencies: `pip install -r requirements.txt`
4. Make sure you have the ESTA dataset downloaded
5. Check Python version matches (3.10+)

---

### "Setup script fails"

**Problem:** `./setup_esta.sh` errors

**Solution:**
```bash
# Make sure it's executable
chmod +x setup_esta.sh

# Run with bash explicitly
bash setup_esta.sh
```

---

## Dataset Information

### ESTA Dataset

- **Full Name:** Esports Trajectories and Actions
- **Matches:** 1,558 professional CS:GO matches
- **Time Period:** January 2021 - May 2022
- **Size:** ~5-10 GB (parsed)
- **Format:** JSON files
- **GitHub:** https://github.com/pnxenopoulos/esta

### What's Included:

**Match-level:**
- Match metadata (teams, tournament, date)
- Map played
- Final score

**Round-level:**
- Round winners
- Team equipment values
- Team sides (T/CT)
- Alive player counts

**Player-level (per round):**
- Kills, deaths, assists
- Damage dealt
- Equipment values
- Positions (x, y, z)
- Health, armor
- Weapons, utility

**Events:**
- Kill events (weapon, distance, headshot)
- Bomb plants/defuses
- Grenade throws
- Flash events

---

## Expected Sample Sizes

From 1,558 matches:

- **Team-level samples:** 1,558 × 30 rounds × 2 teams = **~93,480 samples**
- **Player-level samples:** 1,558 × 30 rounds × 10 players = **~467,400 samples**

Both are sufficient for training all 5 models in the proposal! ✅

---

## Model Training Workflow

### Standard workflow for each model:

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Preprocess data (if not done)
python scripts/preprocess_data.py

# 3. Train model
python scripts/train_logistic.py  # or train_rf.py, train_mlp.py, etc.

# 4. Evaluate model
python scripts/evaluate_model.py --model logistic

# 5. Generate plots
python scripts/plot_accuracy_curves.py

# 6. Commit code (NOT model weights!)
git add scripts/train_logistic.py
git add results/logistic_accuracy.png
git commit -m "Add logistic regression training"
git push
```

---

## Collaboration Tips

### Before Starting a New Feature:

1. Pull latest code: `git pull`
2. Create a branch: `git checkout -b feature/my-feature`
3. Verify data: `python scripts/verify_esta.py`

### When Committing:

1. Only commit code, not data
2. Write clear commit messages
3. Test your code before pushing
4. Don't commit `.pyc` files or `__pycache__/`

### Communication:

- **Who's working on what?** Coordinate model assignments
  - Devon: Logistic + Random Forest + kNN?
  - Jackie: MLP + Transformer?
- **Share results:** Commit plots/figures to `results/`
- **Document changes:** Update README if you add new scripts

---

## Quick Reference

### Key Commands

```bash
# Setup (once)
./setup_esta.sh

# Every session
source .venv/bin/activate

# Verify data
python scripts/verify_esta.py

# Pull updates
git pull

# Commit changes
git add <files>
git commit -m "message"
git push

# Check what's being committed
git status
```

### Key Files

- `SETUP.md` - This file (setup instructions)
- `README.md` - Project overview
- `.gitignore` - What NOT to commit
- `requirements.txt` - Python dependencies
- `setup_esta.sh` - Automated setup script

---

## Need Help?

1. **Check troubleshooting section** above
2. **Verify installation:** `python scripts/verify_esta.py`
3. **Check ESTA docs:** https://github.com/pnxenopoulos/esta
4. **Ask your teammate:** Devon or Jackie
5. **Check project proposal:** CSCE768_Final_Project_Proposal.pdf

---

## Summary for Quick Reference

**First-time setup:**
```bash
git clone <repo>
cd CSCE768-Final-Project
./setup_esta.sh
# Download ESTA dataset to data/esta/raw/
python scripts/verify_esta.py
```

**Every time you work:**
```bash
git pull
source .venv/bin/activate
# Do your work
git add <your-code-files>
git commit -m "description"
git push
```

**Remember:**
- ✅ Commit code
- ❌ Don't commit data
- ✅ Download ESTA locally
- ❌ Data not in GitHub

---

**Questions?** See ESTA_COMPLETE_ANALYSIS.md or KAGGLE_VS_ESTA_COMPARISON.md for dataset details.
