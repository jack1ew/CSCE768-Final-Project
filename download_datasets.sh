#!/bin/bash
# Automated dataset download script

set -e

echo "================================================================"
echo "Starting automated dataset download..."
echo "================================================================"

# Check if ESTA data already exists
if [ -d "data/esta" ] && [ "$(ls -A data/esta)" ]; then
    echo "✓ ESTA dataset already exists, skipping download"
else
    echo "Downloading ESTA dataset..."
    # Try to find ESTA dataset download link
    # Note: This is a placeholder - actual download depends on dataset availability
    echo "⚠ ESTA dataset requires manual download"
    echo "  Visit: https://github.com/skybox-sa/esta-dataset"
fi

# Check for Kaggle dataset
if [ -d "data/kaggle" ] && [ "$(ls -A data/kaggle)" ]; then
    echo "✓ Kaggle dataset already exists, skipping download"
else
    echo "Downloading Kaggle dataset..."
    # Check if kaggle CLI is available
    if command -v kaggle &> /dev/null; then
        mkdir -p data/kaggle
        cd data/kaggle
        kaggle datasets download -d mateusdmachado/csgo-professional-matches
        unzip -o csgo-professional-matches.zip
        rm csgo-professional-matches.zip
        cd ../..
        echo "✓ Kaggle dataset downloaded"
    else
        echo "⚠ Kaggle CLI not found"
        echo "  Install: pip install kaggle"
        echo "  Or download manually from: https://www.kaggle.com/datasets/mateusdmachado/csgo-professional-matches"
    fi
fi

echo "================================================================"
echo "Dataset download check complete"
echo "================================================================"
