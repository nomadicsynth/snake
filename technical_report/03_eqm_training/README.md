# 03 - EqM Training

Train EqM world models with default hyperparameters to establish the main experimental results for joint state-action prediction.

## Required Scripts

- `train_eqm_pilot.sh` - Quick pilot run on 10k dataset
- `train_eqm_medium.sh` - Medium-scale run on 50k dataset
- `train_eqm_large.sh` - Final run on 200k dataset
- `run_all.sh` - Execute all EqM training runs
- `monitor_eqm_training.py` - Track loss components and convergence
- `export_eqm_metrics.py` - Extract metrics from training logs
