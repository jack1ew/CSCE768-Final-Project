#!/usr/bin/env python3
"""
Train CNN + Transformer on All Datasets
Train the CNN + Transformer model on all 3 datasets to match other models' training protocol
"""

import subprocess
import time
from pathlib import Path
from datetime import datetime
import sys

# Paths
VENV_PYTHON = Path("/Users/jackiewang/CSCE768/final_project_combined/CSCE768-Final-Project/.venv/bin/python")
SCRIPTS_DIR = Path("scripts")
TRAIN_SCRIPT = "train_cnn_transformer.py"

# Datasets
DATASETS = ['esta', 'kaggle', 'combined']

def run_training(dataset, split='random'):
    """Run a single CNN + Transformer training"""
    print(f"\n{'='*80}")
    print(f"TRAINING: CNN + Transformer on {dataset.upper()} dataset ({split} split)")
    print(f"{'='*80}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    script_path = SCRIPTS_DIR / TRAIN_SCRIPT
    start_time = time.time()

    try:
        result = subprocess.run(
            [str(VENV_PYTHON), str(script_path), '--dataset', dataset, '--split', split],
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout per model (CNN + Transformer takes longer)
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"✓ SUCCESS ({elapsed:.1f}s)")
            print(result.stdout)
            return True
        else:
            print(f"✗ FAILED ({elapsed:.1f}s)")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"✗ TIMEOUT after {elapsed:.1f}s")
        return False
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ ERROR ({elapsed:.1f}s): {str(e)}")
        return False

def main():
    """Main training orchestrator"""
    print("="*80)
    print("TRAINING CNN + TRANSFORMER ON ALL DATASETS")
    print("="*80)
    print(f"Total trainings: {len(DATASETS)}")
    print(f"Datasets: {', '.join(DATASETS)}")
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    overall_start = time.time()
    results = []

    # Train on each dataset (default random split to match other models)
    for dataset in DATASETS:
        print(f"\n{'#'*80}")
        print(f"# DATASET: {dataset.upper()}")
        print(f"{'#'*80}")

        success = run_training(dataset, split='random')
        results.append({
            'dataset': dataset,
            'success': success
        })

    # Summary
    overall_elapsed = time.time() - overall_start
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Total time: {overall_elapsed/60:.1f} minutes")
    print(f"Successful: {successful}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")

    if failed > 0:
        print("\nFailed trainings:")
        for r in results:
            if not r['success']:
                print(f"  ✗ CNN + Transformer on {r['dataset']}")

    print("\n" + "="*80)
    print("CNN + TRANSFORMER NOW TRAINED ON ALL DATASETS")
    print("="*80)

    # Return status code
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
