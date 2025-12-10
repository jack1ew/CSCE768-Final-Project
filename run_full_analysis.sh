#!/bin/bash
# Master script to run complete analysis pipeline
# This script: downloads data, trains models, runs analysis, generates report

set -e  # Exit on error

echo "======================================================================"
echo "CS:GO WIN PROBABILITY - FULL ANALYSIS PIPELINE"
echo "======================================================================"
echo ""

# Step 1: Setup environment
echo "Step 1/6: Setting up environment..."
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Run ./setup_esta.sh first."
    exit 1
fi

source .venv/bin/activate

# Step 2: Download and preprocess data
echo ""
echo "======================================================================"
echo "Step 2/6: Downloading and preprocessing datasets..."
echo "======================================================================"
echo ""

# Check if ESTA data exists
if [ ! -d "data/esta" ]; then
    echo "Downloading ESTA dataset..."
    python scripts/download_esta.py || echo "⚠ ESTA download script not found, assuming data exists"
fi

# Preprocess ESTA
echo "Processing ESTA dataset..."
python scripts/preprocess_data.py

# Check if Kaggle data exists
if [ ! -d "data/kaggle" ]; then
    echo ""
    echo "⚠ Kaggle dataset not found!"
    echo "Please download it manually:"
    echo "  1. Go to https://www.kaggle.com/datasets/mateusdmachado/csgo-professional-matches"
    echo "  2. Download and extract to data/kaggle/"
    echo "  3. Re-run this script"
    exit 1
fi

# Preprocess Kaggle (filter for CS:GO only, no CS2)
echo "Processing Kaggle dataset (CS:GO only, pre-CS2)..."
python scripts/preprocess_kaggle.py

# Build combined dataset
echo "Building combined dataset..."
python scripts/build_combined_dataset.py

echo "✓ Data preprocessing complete!"

# Step 3: Train all models
echo ""
echo "======================================================================"
echo "Step 3/6: Training all models on all datasets..."
echo "======================================================================"
echo ""

python scripts/train_all_models.py

echo "✓ Model training complete!"

# Step 4: Run round-by-round analysis
echo ""
echo "======================================================================"
echo "Step 4/6: Running round-by-round analysis..."
echo "======================================================================"
echo ""

python scripts/round_by_round_analysis.py

echo "✓ Round-by-round analysis complete!"

# Step 5: Generate baseline figures
echo ""
echo "======================================================================"
echo "Step 5/6: Generating baseline evaluation figures..."
echo "======================================================================"
echo ""

python scripts/generate_evaluation_figures.py

echo "✓ Baseline figures generated!"

# Step 6: Generate LaTeX report
echo ""
echo "======================================================================"
echo "Step 6/6: Generating jackiew_report.pdf..."
echo "======================================================================"
echo ""

cd report

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "⚠ pdflatex not found. Please install LaTeX:"
    echo "  - macOS: brew install --cask mactex"
    echo "  - Ubuntu: sudo apt-get install texlive-full"
    echo ""
    echo "Skipping PDF generation. You can compile manually later with:"
    echo "  cd report && pdflatex jackiew_report.tex"
else
    # Compile LaTeX (run twice for references)
    pdflatex -interaction=nonstopmode jackiew_report.tex
    pdflatex -interaction=nonstopmode jackiew_report.tex

    if [ -f "jackiew_report.pdf" ]; then
        echo "✓ Report generated successfully: report/jackiew_report.pdf"
    else
        echo "✗ Report generation failed. Check LaTeX errors above."
        cd ..
        exit 1
    fi
fi

cd ..

# Summary
echo ""
echo "======================================================================"
echo "ANALYSIS COMPLETE!"
echo "======================================================================"
echo ""
echo "Generated artifacts:"
echo "  - Trained models: models/saved_models/"
echo "  - Metrics: results/*.csv"
echo "  - Figures: results/figures/ and results/round_by_round/"
echo "  - Report: report/jackiew_report.pdf"
echo ""
echo "To view the report:"
echo "  open report/jackiew_report.pdf   # macOS"
echo "  xdg-open report/jackiew_report.pdf   # Linux"
echo ""
echo "======================================================================"
