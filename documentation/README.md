# CSCE768 Final Project — CS:GO Win Probability

_Quick progress log. Full setup lives in `SETUP.md`; read the detailed write-up in [report/main.pdf](report/main.pdf) and see the original [project proposal](CSCE768_Final_Project_Proposal.pdf)._ 

---

## Current Progress (as of Nov 11 2025)

| Dataset | Logistic | Random Forest | k-NN | MLP | Transformer |
|---------|----------|---------------|------|-----|-------------|
| **ESTA** (1.6k matches, player telemetry) | 0.747 / 0.503 | 0.741 / 0.499 | 0.715 / 1.225 | 0.747 / 0.492 | **0.955 / 0.063** |
| **Kaggle** (31k matches, economy only)     | 0.695 / 0.584 | 0.665 / 0.712 | 0.648 / 3.080 | 0.695 / 0.580 | **0.847 / 0.395** |
| **Combined** (33k matches)                 | 0.703 / 0.576 | 0.679 / 0.627 | 0.654 / 2.869 | 0.703 / 0.572 | **0.854 / 0.381** |

_Numbers = accuracy / log-loss (from `results/*.csv`). Combined dataset adds Kaggle breadth onto ESTA’s telemetry: Δ(+0.007 accuracy, –0.015 log-loss) for the transformer versus Kaggle alone._

---

## Completed Work?

Git history to date:

| Commits | Contributor | Focus Areas |
|---------|-------------|-------------|
| **3** | **jack1ew** `<jackiew@email.sc.edu>` | Repo bootstrap, upload automation, initial documentation. |
| **1** | **j-vaught** `<jvaught@sc.edu>` | Combined dataset plumbing, multi-dataset training, comprehensive figures/report refresh. |

Planned division of labor:
- **Jackie (jack1ew)** — Owns transformer recipe, diagnostics, and final report polish (time-split validation + ablation appendix still pending).
- **JC (j-vaught)** — Owns data ingestion, classical baselines, combined dataset QA, and release readiness.

If you contribute next, `git config user.name "<your handle>"` + `git config user.email "<you>@sc.edu"` so the blame table stays accurate.

---

## Reproduce the Latest Results
1. **Prep datasets** (local only):
   ```bash
   ./setup_esta.sh                      # one-time env + dirs
   python scripts/preprocess_data.py    # ESTA
   python scripts/preprocess_kaggle.py  # Kaggle economy melt
   python scripts/build_combined_dataset.py
   ```
2. **Train models** (pick dataset via `--dataset {esta,kaggle,combined}`):
   ```bash
   python scripts/train_logistic.py --dataset combined
   python scripts/train_transformer.py --dataset combined
   ```
3. **Generate artifacts**:
   ```bash
   python scripts/generate_report_figures.py
   python scripts/generate_evaluation_figures.py   # renders ESTA/Kaggle/Combined ROC/PR/etc.
   cd report && pdflatex -interaction=nonstopmode -halt-on-error main.tex
   ```

Artifacts (pipelines, torch weights) stay in `models/saved_models/` per `.gitignore`; metrics and plots are tracked in Git.

---

## Round-by-Round Analysis (NEW)

**Extended analysis by Jackie Wang** - Answers: "How does prediction accuracy evolve from round 1 to round 30?"

- **Full instructions**: `JACKIEW_ANALYSIS_README.md`
- **Setup summary**: `EXECUTION_SUMMARY.md`
- **Quick start**: `./run_full_analysis.sh`
- **Report output**: `report/jackiew_report.pdf`

This analysis tracks accuracy, calibration, and confidence for all 5 models across rounds 1-30, revealing that predictions improve from ~60% accuracy (early rounds) to ~95% accuracy (late rounds).

## Need More?
- **Setup / data download:** `SETUP.md`
- **Dataset deep dive:** `ESTA_COMPLETE_ANALYSIS.md`
- **Proposal:** `CSCE768_Final_Project_Proposal.pdf`
- **Baseline analysis:** `report/main.pdf`
- **Round-by-round analysis:** `report/jackiew_report.pdf`

Ping on Slack if you're picking up a new task so the ownership table stays current.
