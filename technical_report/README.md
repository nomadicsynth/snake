# Technical Report: Equilibrium World Models for Sequential Decision Making

**Report ID**: NC-TR-2025-001  
**Status**: In Preparation

## Overview

This directory contains all materials for preparing and creating the technical report on EqM-based world models for Snake RL.

## Structure

- `01_datasets/` - Dataset generation scripts
- `02_baseline_training/` - Baseline policy training experiments
- `03_eqm_training/` - EqM world model training experiments
- `04_ablations/` - Ablation studies (schedules, architectures, hyperparameters)
- `05_evaluation/` - Model evaluation and in-game performance testing
- `06_analysis/` - Results analysis and metric computation
- `07_figures/` - Figure generation and visualization scripts
- `08_writing/` - Report writing, LaTeX source, and drafts
- `09_release/` - Release preparation, model uploads, announcement materials

## Quick Start

Run experiments in order:

```bash
# 1. Generate datasets
cd 01_datasets && bash run_all.sh

# 2. Train baseline
cd ../02_baseline_training && bash run_all.sh

# 3. Train EqM models
cd ../03_eqm_training && bash run_all.sh

# 4. Run ablations
cd ../04_ablations && bash run_all.sh

# 5. Evaluate models
cd ../05_evaluation && bash run_all.sh

# 6. Analyze results
cd ../06_analysis && python collect_metrics.py

# 7. Generate figures
cd ../07_figures && python generate_all_figures.py

# 8. Compile report
cd ../08_writing && make report.pdf
```

## Progress Tracking

See `docs/TECHNICAL_REPORT_PREP.md` for the full checklist.
