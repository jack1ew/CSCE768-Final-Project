#!/bin/bash
# ESTA Dataset Setup Script for CSCE768 Final Project
# Team Members: Devon Goshorn, Jackie Wang
#
# This script downloads and sets up the ESTA dataset locally on your machine.
# The data is NOT committed to GitHub (too large - 5-10 GB).

set -e  # Exit on error

echo "======================================================================"
echo "CSCE768 Final Project - ESTA Dataset Setup"
echo "======================================================================"
echo ""
echo "This script will:"
echo "  1. Create data directory structure"
echo "  2. Install awpy (CS:GO demo parser)"
echo "  3. Download ESTA dataset (~5-10 GB)"
echo "  4. Verify installation"
echo ""
echo "⚠️  IMPORTANT: This downloads 5-10 GB of data!"
echo "   Make sure you have enough disk space and good internet connection."
echo ""

read -p "Continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled."
    exit 1
fi

# ============================================================================
# Step 1: Create directory structure
# ============================================================================
echo ""
echo "Step 1: Creating directory structure..."
echo "----------------------------------------------------------------------"

mkdir -p data/esta/raw          # Raw ESTA demo files
mkdir -p data/esta/parsed       # Parsed JSON data
mkdir -p data/processed         # Preprocessed features for training
mkdir -p models/checkpoints     # Model checkpoints
mkdir -p models/saved_models    # Final trained models
mkdir -p results                # Experiment results
mkdir -p logs                   # Training logs

echo "✓ Directories created:"
echo "  - data/esta/raw/          (ESTA demo files)"
echo "  - data/esta/parsed/       (Parsed match data)"
echo "  - data/processed/         (Training features)"
echo "  - models/checkpoints/     (Model checkpoints)"
echo "  - models/saved_models/    (Trained models)"
echo "  - results/                (Experiment outputs)"
echo "  - logs/                   (Training logs)"

# ============================================================================
# Step 2: Setup Python environment
# ============================================================================
echo ""
echo "Step 2: Setting up Python environment..."
echo "----------------------------------------------------------------------"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install required packages
echo "Installing required packages..."
pip install --quiet \
    awpy \
    pandas \
    numpy \
    scikit-learn \
    torch \
    transformers \
    matplotlib \
    seaborn \
    tqdm \
    jupyter

echo "✓ Python packages installed:"
echo "  - awpy (CS:GO demo parser)"
echo "  - pandas, numpy (data processing)"
echo "  - scikit-learn (ML models)"
echo "  - torch, transformers (neural models)"
echo "  - matplotlib, seaborn (visualization)"

# ============================================================================
# Step 3: Download ESTA dataset
# ============================================================================
echo ""
echo "Step 3: Downloading ESTA dataset..."
echo "----------------------------------------------------------------------"
echo ""
echo "⚠️  The ESTA dataset is hosted externally."
echo "   You need to download it manually from:"
echo ""
echo "   https://github.com/pnxenopoulos/esta"
echo ""
echo "Follow these steps:"
echo "  1. Visit the link above"
echo "  2. Find the 'Download' section in the README"
echo "  3. Download the dataset (Google Drive or Kaggle link)"
echo "  4. Extract files to: data/esta/raw/"
echo ""
echo "Once downloaded, run: python scripts/verify_esta.py"
echo ""

# Try to download programmatically (if they provide a script)
if command -v awpy &> /dev/null; then
    echo "Checking for awpy download capabilities..."
    # Note: awpy may not have direct download - check their docs
    # This is a placeholder for future automation
fi

echo ""
echo "⚠️  MANUAL DOWNLOAD REQUIRED"
echo "   Follow instructions above to download ESTA dataset"
echo ""

# ============================================================================
# Step 4: Create verification script
# ============================================================================
echo ""
echo "Step 4: Creating verification script..."
echo "----------------------------------------------------------------------"

cat > scripts/verify_esta.py << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""
Verify ESTA dataset installation
"""
import os
import sys
from pathlib import Path

def verify_installation():
    print("="*70)
    print("ESTA Dataset Installation Verification")
    print("="*70)

    # Check directories
    print("\n1. Checking directory structure...")
    required_dirs = [
        'data/esta/raw',
        'data/esta/parsed',
        'data/processed',
        'models/checkpoints',
        'results'
    ]

    all_exist = True
    for dir_path in required_dirs:
        exists = Path(dir_path).exists()
        status = "✓" if exists else "✗"
        print(f"   {status} {dir_path}")
        if not exists:
            all_exist = False

    if not all_exist:
        print("\n⚠️  Some directories missing. Run: ./setup_esta.sh")
        return False

    # Check for ESTA data
    print("\n2. Checking for ESTA data...")
    esta_raw = Path('data/esta/raw')

    if not esta_raw.exists():
        print("   ✗ data/esta/raw/ not found")
        return False

    # Count demo files or JSON files
    demo_files = list(esta_raw.glob('*.dem'))
    json_files = list(esta_raw.glob('*.json'))

    if len(demo_files) > 0:
        print(f"   ✓ Found {len(demo_files)} .dem files")
    elif len(json_files) > 0:
        print(f"   ✓ Found {len(json_files)} .json files")
    else:
        print("   ✗ No demo or JSON files found in data/esta/raw/")
        print("   → Download ESTA dataset and place files here")
        return False

    # Check Python packages
    print("\n3. Checking Python packages...")
    required_packages = [
        'awpy',
        'pandas',
        'numpy',
        'sklearn',
        'torch'
    ]

    all_installed = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✓ {package}")
        except ImportError:
            print(f"   ✗ {package} not installed")
            all_installed = False

    if not all_installed:
        print("\n⚠️  Some packages missing. Run: ./setup_esta.sh")
        return False

    # Success!
    print("\n" + "="*70)
    print("✓ Installation verified successfully!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run: python scripts/parse_esta.py")
    print("  2. Run: python scripts/preprocess_data.py")
    print("  3. Start training models!")

    return True

if __name__ == '__main__':
    success = verify_installation()
    sys.exit(0 if success else 1)
PYTHON_SCRIPT

chmod +x scripts/verify_esta.py

echo "✓ Created: scripts/verify_esta.py"

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "======================================================================"
echo "Setup Complete!"
echo "======================================================================"
echo ""
echo "✓ Directory structure created"
echo "✓ Python environment configured"
echo "✓ awpy and dependencies installed"
echo ""
echo "⚠️  NEXT STEPS:"
echo "----------------------------------------------------------------------"
echo ""
echo "1. Download ESTA dataset:"
echo "   → Visit: https://github.com/pnxenopoulos/esta"
echo "   → Download files and place in: data/esta/raw/"
echo ""
echo "2. Verify installation:"
echo "   → Run: python scripts/verify_esta.py"
echo ""
echo "3. See SETUP.md for complete instructions"
echo ""
echo "======================================================================"
echo ""
