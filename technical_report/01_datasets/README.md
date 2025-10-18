# 01 - Datasets

Generate all datasets needed for training baseline and EqM models, including standard state-action pairs and world model state-action-nextstate tuples.

## Required Scripts

- `generate_baseline_small.sh` - 10k samples for quick experiments
- `generate_baseline_medium.sh` - 50k samples for medium-scale training
- `generate_baseline_large.sh` - 200k samples for final baseline
- `generate_world_model_small.sh` - 10k samples with next_state for pilot EqM
- `generate_world_model_medium.sh` - 50k samples with next_state for medium EqM
- `generate_world_model_large.sh` - 200k samples with next_state for final EqM
- `run_all.sh` - Execute all dataset generation scripts
- `verify_datasets.py` - Check dataset integrity and statistics
