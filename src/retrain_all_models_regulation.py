#!/usr/bin/env python3
"""
Retrain All Models on Regulation Data
Train all 5 models on all 3 datasets (regulation rounds only)
Total: 15 model trainings
"""

import subprocess
import time
from pathlib import Path
from datetime import datetime

# Paths
VENV_PYTHON = Path("/Users/jackiewang/CSCE768/CSCE768-Final-Project/.venv/bin/python")
SCRIPTS_DIR = Path("scripts")

# Model training scripts
MODELS = {
    'logistic': 'train_logistic.py',
    'random_forest': 'train_rf.py',
    'knn': 'train_knn.py',
    'mlp': 'train_mlp.py',
    'transformer': 'train_transformer.py'
}

# Datasets
DATASETS = ['esta', 'kaggle', 'combined']

def run_training(model_name, script_name, dataset):
    """Run a single model training"""
    print(f"\n{'='*80}")
    print(f"TRAINING: {model_name.upper()} on {dataset.upper()} dataset")
    print(f"{'='*80}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    script_path = SCRIPTS_DIR / script_name
    start_time = time.time()

    try:
        result = subprocess.run(
            [str(VENV_PYTHON), str(script_path), '--dataset', dataset],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per model
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
    """Main retraining orchestrator"""
    print("="*80)
    print("RETRAINING ALL MODELS ON REGULATION DATA")
    print("="*80)
    print(f"Total trainings: {len(MODELS) * len(DATASETS)}")
    print(f"Models: {', '.join(MODELS.keys())}")
    print(f"Datasets: {', '.join(DATASETS)}")
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    overall_start = time.time()
    results = []

    # Train each model on each dataset
    for dataset in DATASETS:
        print(f"\n{'#'*80}")
        print(f"# DATASET: {dataset.upper()}")
        print(f"{'#'*80}")

        for model_name, script_name in MODELS.items():
            success = run_training(model_name, script_name, dataset)
            results.append({
                'model': model_name,
                'dataset': dataset,
                'success': success
            })

    # Summary
    overall_elapsed = time.time() - overall_start
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful

    print("\n" + "="*80)
    print("RETRAINING COMPLETE!")
    print("="*80)
    print(f"Total time: {overall_elapsed/60:.1f} minutes")
    print(f"Successful: {successful}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")

    if failed > 0:
        print("\nFailed trainings:")
        for r in results:
            if not r['success']:
                print(f"  ✗ {r['model']} on {r['dataset']}")

    print("\n" + "="*80)
    print("MODELS NOW TRAINED ON REGULATION ROUNDS ONLY (1-30)")
    print("="*80)

    # Return status code
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit(main())
