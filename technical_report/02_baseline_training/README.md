# 02 - Baseline Training

Train standard transformer policies on Snake to establish comparison baselines for action prediction accuracy, training time, and game performance.

## Required Scripts

- `train_baseline_small.sh` - Train on 10k dataset for quick validation
- `train_baseline_medium.sh` - Train on 50k dataset for medium-scale baseline
- `train_baseline_large.sh` - Train on 200k dataset for final baseline
- `run_all.sh` - Execute all baseline training runs
- `monitor_training.py` - Track training progress and early stopping
- `export_metrics.py` - Extract final metrics from training logs
