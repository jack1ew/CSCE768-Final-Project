# CSCE768 Final Project: CS:GO Win Probability Prediction

**Course**: CSCE768 - Neural Networks and Deep Learning
**Professor**: Jianjun Hu
**Authors**: Jackie Wang (jackiew@email.sc.edu), JC Vaught (jvaught@sc.edu)
**Semester**: Fall 2025

---

## Quick Start for Grading

This project implements and compares multiple machine learning models for predicting CS:GO match outcomes using round-by-round game data.

### Key Deliverables

1. **[CSCE768_Final_Project_Report.pdf](./CSCE768_Final_Project_Report.pdf)** - Main submission report (8 pages, workshop format) **(START HERE)**
2. **[FINAL_REPORT_EXTENDED.pdf](./FINAL_REPORT_EXTENDED.pdf)** - Extended technical report with complete analysis (39 pages, supplementary)
3. **[PROJECT_PROPOSAL.pdf](./PROJECT_PROPOSAL.pdf)** - Original project proposal
4. **[src/](./src/)** - All source code for data processing, model training, and evaluation
5. **[results_analysis/](./results_analysis/)** - Experimental results, figures, and metrics

> **Note**: See [VERSION_NOTES.md](./VERSION_NOTES.md) for detailed explanation of report versions.

---

## Project Overview

### Research Question
How does prediction accuracy for CS:GO match outcomes evolve from round 1 to round 30, and which model architecture best captures this temporal progression?

### Datasets Used

| Dataset | Matches | Features | Description | Download Link |
|---------|---------|----------|-------------|---------------|
| **ESTA** | 1,558 | Player telemetry + economy | Detailed player-level data from professional tournaments (Jan 2021-May 2022) | [GitHub](https://github.com/pnxenopoulos/esta) |
| **Kaggle** | 31,710 | Economy only | Large-scale professional matches with economic data (Nov 2015-Mar 2020) | [Kaggle](https://www.kaggle.com/datasets/christianlillelund/csgo-round-winner-classification) |
| **Combined** | 33,268 | Merged features | Union of ESTA + Kaggle datasets | Merge both above |

### Models Evaluated

1. **Logistic Regression** (baseline)
2. **Random Forest** (ensemble)
3. **k-Nearest Neighbors** (instance-based)
4. **Multi-Layer Perceptron** (neural network)
5. **Transformer** (sequence model) - **Best performer**

### Key Results

| Dataset | Logistic | Random Forest | k-NN | MLP | Transformer |
|---------|----------|---------------|------|-----|-------------|
| **ESTA** | 0.747 / 0.503 | 0.741 / 0.499 | 0.715 / 1.225 | 0.747 / 0.492 | **0.955 / 0.063** |
| **Kaggle** | 0.695 / 0.584 | 0.665 / 0.712 | 0.648 / 3.080 | 0.695 / 0.580 | **0.847 / 0.395** |
| **Combined** | 0.703 / 0.576 | 0.679 / 0.627 | 0.654 / 2.869 | 0.703 / 0.572 | **0.854 / 0.381** |

*Numbers show accuracy / log-loss. Transformer significantly outperforms all baselines.*

---

## Directory Structure and Key Files

### 📄 Root Level - Reports and Documentation

```
├── CSCE768_Final_Project_Report.pdf  ⭐ MAIN SUBMISSION (8 pages)
├── FINAL_REPORT_EXTENDED.pdf         📚 Extended technical report (39 pages)
├── PROJECT_PROPOSAL.pdf              📋 Original project proposal
├── VERSION_NOTES.md                  📝 Explains report versions
├── README.md                         📖 This file
├── requirements.txt                  📦 Python dependencies
```

**Key Files**:
- **CSCE768_Final_Project_Report.pdf**: Workshop-style paper for course submission (8 pages)
- **FINAL_REPORT_EXTENDED.pdf**: Complete analysis with CNN-Transformer section, data leakage investigation, full experimental details
- **VERSION_NOTES.md**: Detailed comparison between condensed and extended reports

### 🔬 src/ - Source Code (36 Python scripts)

**Data Preprocessing Scripts**:
```
src/
├── preprocess_data.py              # ESTA dataset preprocessing
├── preprocess_kaggle.py            # Kaggle dataset preprocessing
├── build_combined_dataset.py       # Merge ESTA + Kaggle datasets
└── parse_esta.py                   # Raw ESTA JSON parsing
```

**Model Training Scripts** (5 traditional models):
```
src/
├── train_logistic.py               # Logistic Regression (baseline)
├── train_random_forest.py          # Random Forest ensemble
├── train_knn.py                    # k-Nearest Neighbors
├── train_mlp.py                    # Multi-Layer Perceptron
├── train_transformer.py            # Transformer encoder
└── train_all_models.py             # Batch training script
```

**Analysis and Evaluation Scripts**:
```
src/
├── round_accuracy_curves.py        # Round-by-round evolution analysis
├── round_by_round_analysis.py      # Detailed round-level metrics
├── feature_importance_all_datasets.py  # Feature importance ranking
├── compare_all_6_models.py         # Model comparison utilities
├── comprehensive_model_comparison.py   # Cross-dataset comparison
└── generate_evaluation_figures.py  # ROC curves, calibration plots
```

**Report Generation Scripts**:
```
src/
├── generate_report_figures.py      # Main report figures
├── generate_comprehensive_report.py # Extended report figures
├── generate_training_curves.py     # Training/validation curves
└── generate_cnn_transformer_figures.py  # CNN-Transformer analysis
```

**Specialized Analysis Scripts**:
```
src/
├── economic_momentum_analysis.py   # Economic impact on comebacks
├── comeback_analysis_with_economy.py   # Comeback probability by economy state
├── overtime_minigame_analysis.py   # Overtime rounds analysis
├── extended_analysis.py            # Additional ablation studies
└── investigate_data_leakage.py     # Data integrity validation
```

### 🧠 models_code/ - Model Architectures

```
models_code/
├── architectures/
│   ├── cnn_transformer.py          # CNN-Transformer hybrid (Section 7)
│   ├── transformer.py              # Pure Transformer encoder
│   └── mlp.py                      # MLP architecture definition
├── saved_models/                   # Trained model checkpoints (not in git)
└── README.md                       # Model documentation
```

**Where to Find**:
- Transformer implementation: `architectures/transformer.py`
- CNN-Transformer hybrid: `architectures/cnn_transformer.py`
- Trained models: `saved_models/` (excluded from git, generated during training)

### 📊 results_analysis/ - Experimental Results

```
results_analysis/
├── results/                        # CSV metrics and predictions
│   ├── esta_results.csv           # ESTA dataset results
│   ├── kaggle_results.csv         # Kaggle dataset results
│   ├── combined_results.csv       # Combined dataset results
│   ├── round_by_round/            # Per-round accuracy metrics
│   │   └── round_by_round_metrics.csv
│   └── multi_run/                 # 5-run training statistics
└── figures/                        # Generated plots (in git)
    ├── accuracy_curves.png
    ├── roc_curves.png
    ├── calibration_plots.png
    └── feature_importance.png
```

**Where to Find**:
- Model performance metrics: `results/*.csv`
- Round-by-round analysis: `results/round_by_round/`
- All figures used in reports: `figures/`

### 💾 data_processing/ - Datasets and Preprocessing

```
data_processing/
├── data/
│   ├── esta/                      # ESTA dataset files (not in git)
│   ├── kaggle/                    # Kaggle dataset files (not in git)
│   └── processed/                 # Preprocessed feature matrices
│       ├── esta_processed.pkl
│       ├── kaggle_processed.pkl
│       └── combined_processed.pkl
```

**Where to Find**:
- Raw datasets: `data/esta/` and `data/kaggle/` (download required)
- Preprocessed features: `data/processed/*.pkl`
- Dataset setup instructions: `documentation/SETUP.md`

### 📚 documentation/ - Additional Documentation

```
documentation/
├── SETUP.md                       # Environment setup, dataset download
├── EXECUTION_SUMMARY.md           # Implementation timeline
├── ESTA_COMPLETE_ANALYSIS.md      # Deep dive into ESTA dataset
├── DATA_GRANULARITY_ANALYSIS.md   # Feature engineering decisions
├── KAGGLE_VS_ESTA_COMPARISON.md   # Dataset comparison analysis
├── ROUND_BY_ROUND_PLAYER_DATA_OPTIONS.md  # Temporal modeling approach
├── jackiew_report.pdf             # Round-by-round detailed analysis
├── devon_report.pdf               # Alternative analysis perspective
└── [18 total markdown files]      # Various technical notes
```

**Where to Find**:
- Setup instructions: `SETUP.md`
- Dataset analysis: `ESTA_COMPLETE_ANALYSIS.md`, `KAGGLE_VS_ESTA_COMPARISON.md`
- Implementation details: `EXECUTION_SUMMARY.md`

### 🔧 Root Level - Automation Scripts

```
├── setup_esta.sh                  # ESTA dataset download and setup
├── download_datasets.sh           # Automated dataset retrieval
├── run_full_analysis.sh           # Complete training pipeline
└── run_complete_training.sh       # Multi-dataset training batch
```

**Where to Find**:
- Dataset setup: `setup_esta.sh`, `download_datasets.sh`
- Full pipeline execution: `run_full_analysis.sh`
- Model training automation: `run_complete_training.sh`

---

## Quick File Finder

| What You're Looking For | Where to Find It |
|-------------------------|------------------|
| **Main submission report** | `CSCE768_Final_Project_Report.pdf` |
| **Extended technical report** | `FINAL_REPORT_EXTENDED.pdf` |
| **Transformer model code** | `models_code/architectures/transformer.py` |
| **Training script for Transformer** | `src/train_transformer.py` |
| **Model performance results** | `results_analysis/results/*.csv` |
| **Round-by-round analysis** | `src/round_by_round_analysis.py` |
| **Feature importance** | `src/feature_importance_all_datasets.py` |
| **All figures** | `results_analysis/figures/` |
| **Dataset preprocessing** | `src/preprocess_data.py`, `src/preprocess_kaggle.py` |
| **Setup instructions** | `documentation/SETUP.md` |
| **CNN-Transformer analysis** | Section 7 in `FINAL_REPORT_EXTENDED.pdf` |
| **Data leakage investigation** | Section 5.6 in `FINAL_REPORT_EXTENDED.pdf` |

---

## Reproducing Results

### Prerequisites

- Python 3.8+
- pip
- (Optional) CUDA-capable GPU for faster Transformer training

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download datasets (see documentation/SETUP.md for details)
./setup_esta.sh
./download_datasets.sh
```

### Running the Full Pipeline

```bash
# Run complete analysis (2-4 hours without GPU)
./run_full_analysis.sh
```

This script will:
1. Preprocess all datasets
2. Train all 5 models on all 3 datasets (15 models total)
3. Generate round-by-round accuracy curves
4. Create all figures and evaluation metrics

### Running Individual Components

```bash
# Preprocess data
python src/preprocess_data.py
python src/preprocess_kaggle.py
python src/build_combined_dataset.py

# Train specific model
python src/train_transformer.py --dataset combined

# Generate analysis
python src/round_accuracy_curves.py --dataset esta
python src/generate_report_figures.py
```

---

## Key Findings

### 1. Transformer Superiority
The Transformer model significantly outperforms all baselines, achieving **95.5% accuracy** on ESTA dataset with only **0.063 log-loss**.

### 2. Round-by-Round Progression
Prediction accuracy improves dramatically as rounds progress:
- **Round 1-5**: ~60% accuracy (slightly better than random)
- **Round 10**: ~75-80% accuracy
- **Round 20**: ~90-95% accuracy (Transformer)
- **Round 30**: ~95%+ accuracy (very high confidence)

### 3. Dataset Quality Matters
ESTA's detailed player telemetry enables significantly better predictions (+10.8% accuracy) compared to Kaggle's economy-only features.

### 4. Combined Dataset Benefits
Merging ESTA and Kaggle provides modest improvements (+0.7% accuracy) by combining breadth with depth.

---

## Implementation Highlights

### Data Preprocessing
- CS:GO vs CS2 filtering (pre-Aug 2023 matches only)
- MR15 format validation (30 round maximum)
- Feature engineering: economy ratios, equipment values, player statistics
- Time-based train/test splits for temporal validation

### Model Architectures
- **Transformer**: 6 layers, 8 attention heads, 512 hidden dimensions
- **MLP**: 3 hidden layers (256, 128, 64 units) with dropout
- **Random Forest**: 100 estimators with max depth 20
- All models use standardized features and stratified sampling

### Evaluation Metrics
- Accuracy
- Log-loss (calibration)
- ROC-AUC
- Precision-Recall curves
- Brier score
- Round-by-round progression

---

## Code Quality Notes

### Testing
All scripts include error handling and validation. Data preprocessing includes sanity checks for:
- Missing values
- Round number consistency
- Match format validation
- Feature range validation

### Documentation
Every major script includes:
- Docstrings for all functions
- Command-line argument descriptions
- Example usage commands
- Expected input/output formats

### Reproducibility
- Fixed random seeds (42) for all models
- Explicit train/test splits saved to disk
- Complete environment specification in requirements.txt
- Detailed execution logs in results_analysis/

---

## Additional Resources

### Documentation Files
- `documentation/SETUP.md` - Detailed environment setup
- `documentation/EXECUTION_SUMMARY.md` - Implementation timeline
- `documentation/ESTA_COMPLETE_ANALYSIS.md` - Deep dive into ESTA dataset
- `documentation/DATA_GRANULARITY_ANALYSIS.md` - Feature engineering details

### Datasets

#### ESTA Dataset
- **Download**: https://github.com/pnxenopoulos/esta
- **Paper**: [ESTA: An Esports Trajectory and Action Dataset](https://arxiv.org/abs/2209.09861) (arXiv:2209.09861)
- **Description**: 1,558 parsed professional CS:GO matches with rich player telemetry including kills, deaths, damage, utility usage, and spatial positioning
- **Format**: JSON compressed in .xz format (~1-3 MB compressed, 25-75 MB decompressed per match)
- **Compatibility Note**: ⚠️ ESTA data is only compatible with [awpy](https://github.com/pnxenopoulos/awpy) version 1.x (not compatible with 2.x+)
- **License**: CC BY-SA 4.0

#### Kaggle CS:GO Dataset
- **Download**: https://www.kaggle.com/datasets/christianlillelund/csgo-round-winner-classification
- **Alternative**: https://www.kaggle.com/datasets/mvidalg/counterstrike-go-hltvorg-dataset (HLTV.org source)
- **Description**: 31,710 professional matches with detailed economic data (equipment values, spending patterns)
- **Time Period**: November 2015 - March 2020
- **Format**: CSV files including Economy.csv, Results.csv, Picks.csv, Players.csv
- **Source**: Data scraped from www.hltv.org

### External Dependencies
See `requirements.txt` for complete list. Key libraries:
- PyTorch 2.0+ (Transformer implementation)
- scikit-learn 1.3+ (Classical models)
- pandas, numpy (Data processing)
- matplotlib, seaborn (Visualization)

---

## Contact

For questions about this project:
- **Jackie Wang**: jackiew@email.sc.edu
- **JC Vaught**: jvaught@sc.edu

---

## License

This project was completed as coursework for CSCE768 at the University of South Carolina.

---

## Acknowledgments

- ESTA dataset provided by Skybox Security
- Kaggle CS:GO dataset curated by Mateus Machado
- Course instruction by Professor Jianjun Hu
- University of South Carolina, Department of Computer Science and Engineering
