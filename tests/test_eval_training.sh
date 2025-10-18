#!/bin/bash
# Example: Quick training run with evaluation to verify metrics

source .venv/bin/activate

echo "Training Snake with evaluation enabled..."
echo ""

python train_snake_purejaxrl.py \
    --wandb \
    --width 10 \
    --height 10 \
    --num-envs 2048 \
    --num-steps 128 \
    --total-timesteps 1_000_000 \
    --eval-freq 25 \
    --eval-episodes 128 \
    --lr 2.5e-4 \
    --d-model 64 \
    --num-layers 2 \
    --num-heads 4 \
    --run-name "eval_test_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "Training complete! Check WandB for evaluation metrics."
