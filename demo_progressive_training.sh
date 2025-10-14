#!/bin/bash
# Quick demo of the progressive training script

echo "🐍 Snake JAX PPO Training Demo"
echo "=============================="
echo ""
echo "This script demonstrates the new progressive training with:"
echo "  ✅ Real-time tqdm progress bar"
echo "  ✅ WandB integration"
echo "  ✅ Live metrics display"
echo "  ✅ Command-line arguments"
echo ""

# Use the virtual environment
PYTHON=".venv/bin/python"

# Quick test run
echo "Running quick test (100K timesteps)..."
echo ""

WANDB_MODE=offline $PYTHON train_snake_purejaxrl_progressive.py \
    --total-timesteps 100000 \
    --num-envs 512 \
    --num-steps 64 \
    --wandb \
    --run-name quick_demo

echo ""
echo "Demo complete!"
echo ""
echo "For a full training run, use:"
echo "  $PYTHON train_snake_purejaxrl_progressive.py --wandb --total-timesteps 5000000"
echo ""
