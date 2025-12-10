#!/usr/bin/env python3
"""
Master training script to train all 5 models on all 3 datasets
"""

import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and print status"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(command)}")

    result = subprocess.run(command, capture_output=False)

    if result.returncode == 0:
        print(f"✓ {description} - SUCCESS")
        return True
    else:
        print(f"✗ {description} - FAILED")
        return False

def main():
    datasets = ["esta", "kaggle", "combined"]
    models = [
        ("train_logistic.py", "Logistic Regression"),
        ("train_rf.py", "Random Forest"),
        ("train_knn.py", "k-NN"),
        ("train_mlp.py", "MLP"),
        ("train_transformer.py", "Transformer")
    ]

    scripts_dir = Path("scripts")
    success_count = 0
    total_count = 0

    print("\n" + "="*60)
    print("TRAINING ALL MODELS ON ALL DATASETS")
    print("="*60)

    for dataset in datasets:
        print(f"\n\n{'#'*60}")
        print(f"# DATASET: {dataset.upper()}")
        print(f"{'#'*60}")

        for script_name, model_name in models:
            total_count += 1
            script_path = scripts_dir / script_name

            if not script_path.exists():
                print(f"✗ Script not found: {script_path}")
                continue

            command = [sys.executable, str(script_path), "--dataset", dataset]

            if run_command(command, f"{model_name} on {dataset.upper()}"):
                success_count += 1

    # Print summary
    print("\n\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Successfully trained: {success_count}/{total_count} models")
    print(f"Failed: {total_count - success_count}/{total_count} models")

    if success_count == total_count:
        print("\n✓ All models trained successfully!")
        return 0
    else:
        print(f"\n⚠ {total_count - success_count} model(s) failed to train")
        return 1

if __name__ == "__main__":
    sys.exit(main())
