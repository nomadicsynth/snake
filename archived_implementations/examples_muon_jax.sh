#!/bin/bash
# Quick examples for training Snake with Muon optimizer

# 1. Basic training with Muon (5M steps)
echo "Example 1: Basic Muon training"
echo ".venv/bin/python train_snake_purejaxrl.py --wandb --use-muon --muon-lr 0.02 --total-timesteps 5000000"
echo ""

# 2. Muon with larger network
echo "Example 2: Muon with larger network"
echo ".venv/bin/python train_snake_purejaxrl.py --wandb --use-muon --muon-lr 0.03 --d-model 128 --num-layers 4"
echo ""

# 3. Standard Adam (for comparison)
echo "Example 3: Standard Adam baseline"
echo ".venv/bin/python train_snake_purejaxrl.py --wandb --lr 0.00025 --total-timesteps 5000000"
echo ""

# 4. Test the Muon implementation
echo "Example 4: Test Muon optimizer"
echo ".venv/bin/python test_muon_jax.py"
echo ""

# 5. Run hyperparameter sweep
echo "Example 5: W&B sweep with Muon"
echo "wandb sweep wandb_sweep_jax_ppo.yaml"
echo "wandb agent <sweep-id>"
