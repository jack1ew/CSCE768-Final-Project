# Jackie's Round-by-Round Win Probability Analysis

This document explains how to reproduce the extended round-by-round analysis for the CS:GO win probability project.

## Overview

This analysis extends the baseline work (main.pdf) by answering the key question:

> **How does win prediction accuracy evolve from round 1 to round 30?**

## Key Features

1. **Round-by-Round Accuracy**: Tracks prediction accuracy for each round (1-30)
2. **Calibration Analysis**: Evaluates whether predicted probabilities match actual outcomes
3. **Confidence Evolution**: Shows how model confidence changes as matches progress
4. **All Models & Datasets**: Analyzes all 5 models (Logistic, RF, k-NN, MLP, Transformer) across all 3 datasets (ESTA, Kaggle, Combined)
5. **Automated Report Generation**: LaTeX report with all figures embedded

## Quick Start

### Option 1: Automated (Recommended)

Run the complete pipeline with a single command:

```bash
./run_full_analysis.sh
```

This will:
1. Download and preprocess all datasets (CS:GO only, no CS2)
2. Train all 5 models on all 3 datasets
3. Run round-by-round analysis
4. Generate all figures
5. Compile `jackiew_report.pdf`

**Estimated time**: 2-4 hours depending on hardware (Transformer training is slow without GPU)

### Option 2: Manual Step-by-Step

If you want more control or the automated script fails:

#### 1. Setup Environment

```bash
./setup_esta.sh
source .venv/bin/activate
```

#### 2. Download Datasets

**ESTA Dataset** (if not already present):
```bash
# The setup script should have downloaded this
# If not, manually download from: https://github.com/skybox-sa/esta-dataset
```

**Kaggle Dataset**:
```bash
# Download manually from: https://www.kaggle.com/datasets/mateusdmachado/csgo-professional-matches
# Extract to: data/kaggle/
```

#### 3. Preprocess Data

```bash
# Process ESTA (extracts round-by-round features)
python scripts/preprocess_data.py

# Process Kaggle (filters for CS:GO only, pre-CS2)
python scripts/preprocess_kaggle.py

# Build combined dataset
python scripts/build_combined_dataset.py
```

**Important**: The Kaggle preprocessing script filters out CS2 matches. Only MR15 format (max 30 rounds) CS:GO matches are included.

#### 4. Train All Models

```bash
# Train all 5 models on all 3 datasets (15 total models)
python scripts/train_all_models.py

# Or train individually:
python scripts/train_logistic.py --dataset esta
python scripts/train_rf.py --dataset esta
python scripts/train_knn.py --dataset esta
python scripts/train_mlp.py --dataset esta
python scripts/train_transformer.py --dataset esta

# Repeat for kaggle and combined datasets
```

**Note**: The Transformer is slow without GPU (~3 min per dataset on CPU).

#### 5. Run Round-by-Round Analysis

```bash
python scripts/round_by_round_analysis.py
```

This generates:
- `results/round_by_round/round_by_round_metrics.csv` - Raw data
- `results/round_by_round/*.png` - Figures for each dataset

#### 6. Generate Report

```bash
cd report
pdflatex jackiew_report.tex
pdflatex jackiew_report.tex  # Run twice for references
```

**Requirements**:
- macOS: `brew install --cask mactex`
- Ubuntu: `sudo apt-get install texlive-full`

Output: `report/jackiew_report.pdf`

## Project Structure

```
CSCE768-Final-Project/
├── data/
│   ├── esta/               # ESTA dataset (not in git)
│   ├── kaggle/             # Kaggle dataset (not in git)
│   └── processed/          # Preprocessed parquet files
├── models/
│   ├── architectures/      # Model class definitions
│   └── saved_models/       # Trained model checkpoints (not in git)
├── results/
│   ├── figures/            # Baseline evaluation figures
│   ├── round_by_round/     # Round-by-round analysis figures
│   └── *.csv               # Metrics files
├── report/
│   ├── main.pdf            # Baseline report (j-vaught)
│   ├── jackiew_report.tex  # Extended analysis source
│   └── jackiew_report.pdf  # Extended analysis output
├── scripts/
│   ├── preprocess_*.py     # Data preprocessing
│   ├── train_*.py          # Model training scripts
│   ├── train_all_models.py # Master training script
│   ├── round_by_round_analysis.py  # Round-by-round analysis
│   └── generate_*.py       # Figure generation
└── run_full_analysis.sh    # Master orchestration script
```

## Key Files

| File | Purpose |
|------|---------|
| `scripts/round_by_round_analysis.py` | Main analysis script - computes per-round metrics |
| `scripts/train_all_models.py` | Trains all 5 models on all 3 datasets |
| `report/jackiew_report.tex` | LaTeX source for extended report |
| `run_full_analysis.sh` | One-command full pipeline |

## Expected Results

### ESTA Dataset
- **Early rounds (1-5)**: 55-70% accuracy (near random)
- **Mid rounds (6-15)**: 70-85% accuracy
- **Late rounds (16-30)**: 90-95%+ accuracy (Transformer)

### Kaggle Dataset
- Generally 5-10% lower accuracy than ESTA due to limited features
- Still shows clear upward trend

### Combined Dataset
- Performance between ESTA and Kaggle
- Benefits from Kaggle's volume and ESTA's feature richness

## Troubleshooting

### Issue: "Model not found"
**Solution**: Train models first with `python scripts/train_all_models.py`

### Issue: "round_num column not found"
**Solution**: Kaggle dataset may lack round numbers. The script will skip round-by-round analysis for that dataset.

### Issue: LaTeX compilation fails
**Solution**:
1. Check that pdflatex is installed: `which pdflatex`
2. Ensure figures exist in `results/round_by_round/`
3. Check LaTeX log for missing packages

### Issue: Kaggle dataset includes CS2 matches
**Solution**: The `preprocess_kaggle.py` script filters for:
- Date < August 2023 (before CS2 release)
- Max rounds ≤ 30 (MR15 format)
- Match format = "bo1" or "bo3" or "bo5"

If CS2 matches slip through, manually filter by date:
```python
df = df[df['date'] < '2023-08-01']
```

## Analysis Insights

The round-by-round analysis reveals:

1. **Prediction accuracy increases dramatically** as matches progress (60% → 95%)
2. **Transformer dominates all rounds** due to sequence modeling
3. **ESTA dataset's rich features** matter most in early rounds (1-10)
4. **Models are well-calibrated**: predicted probabilities match actual win rates
5. **Round 10 is the "reliability threshold"**: predictions become consistently trustworthy

See `report/jackiew_report.pdf` for full details, figures, and interpretation.

## Citation

If using this analysis:

```
@misc{wang2025csgo_roundbyround,
  author = {Wang, Jackie},
  title = {Round-by-Round Win Probability Analysis for CS:GO},
  year = {2025},
  school = {University of South Carolina},
  course = {CSCE768}
}
```

## Contact

For questions or issues:
- Jackie Wang: `jackiew@email.sc.edu`
- GitHub Issues: [Link to your repo]

## Acknowledgments

- Baseline analysis: j-vaught, jack1ew
- ESTA dataset: Skybox Lab
- Kaggle dataset: HLTV/mateusdmachado
