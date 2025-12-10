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
