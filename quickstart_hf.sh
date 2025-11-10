#!/bin/bash
# Quick start guide for HuggingFace Snake training

set -e

echo "======================================"
echo "Snake HuggingFace Training Quick Start"
echo "======================================"
echo

echo "Step 1: Generating dataset..."
python generate_dataset_hf.py \
    --num-samples 1000000 \
    --width 32 \
    --height 32 \
    --max-length 100 \
    --use-astar \
    --no-augment \
    --output outputs/datasets/snake_dataset_hf_n1000000_w32_h32_ml100_astar_noaug \
    --seed 42

echo
echo "Step 2: Training model with Muon optimizer..."
python train_hf.py \
    --dataset outputs/datasets/snake_dataset_hf_n1000000_w32_h32_ml100_astar_noaug \
    --d-model 128 \
    --num-layers 2 \
    --num-heads 4 \
    --dropout 0.1 \
    --epochs 2 \
    --batch-size 256 \
    --optimizer muon \
    --muon-lr 0.02 \
    --muon-momentum 0.95 \
    --warmup-ratio 0.1 \
    --lr-scheduler cosine \
    --output-dir outputs/models/snake_hf_output_n1000000_w32_h32_ml100_astar_noaug \
    --wandb \
    --wandb-project snake-hf \
    --seed 42

echo
echo "Step 3: Playing with the model..."
python play_pretrained.py \
    --model-path outputs/models/snake_hf_output_n1000000_w32_h32_ml100_astar_noaug/final_model \
    --env-width 32 --env-height 32 \
    --episodes 100 --max-steps 4000 \
    --no-render

echo
echo "======================================"
echo "Training complete!"
echo "======================================"
