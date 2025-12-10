#!/usr/bin/env python3
"""
Re-run All Analyses with Best Trained Models
Executes all proposal analyses using the best models from multi-run training
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*80}")

    try:
        result = subprocess.run(
            [sys.executable, f"scripts/{script_name}"],
            cwd=Path.cwd(),
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error in {description}:")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    print("="*80)
    print("RE-RUNNING ALL ANALYSES WITH BEST TRAINED MODELS")
    print("="*80)
    print("This will regenerate all results using the best models from multi-run training")
    print("="*80)

    analyses = [
        ("proposal_analysis.py", "Proposal Analysis (Round-by-Round Accuracy)"),
        ("feature_importance_ablations.py", "Feature Importance and Ablation Studies"),
    ]

    success_count = 0
    total_count = len(analyses)

    for script, description in analyses:
        if run_script(script, description):
            success_count += 1

    print(f"\n{'='*80}")
    print(f"ANALYSIS RE-RUN COMPLETE")
    print(f"{'='*80}")
    print(f"Successful: {success_count}/{total_count}")
    print(f"Failed: {total_count - success_count}/{total_count}")

    if success_count == total_count:
        print("\n✓ All analyses completed successfully")
        print("✓ Results updated with best trained models")
        print("\nNext steps:")
        print("  1. Review results in results/proposal_analysis/")
        print("  2. Update LaTeX report with new figures")
        print("  3. Compile final PDF")
        return 0
    else:
        print("\n⚠️  Some analyses failed. Please review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
