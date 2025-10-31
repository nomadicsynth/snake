#!/bin/bash
# Quick start guide for HuggingFace Snake training

set -e

echo "======================================"
echo "Snake HuggingFace Training Quick Start"
echo "======================================"
echo

# 1. Generate dataset
echo "Step 1: Generating dataset..."
python generate_dataset_hf.py \
    --num-samples 50000 \
    --width 20 \
    --height 20 \
    --use-astar \
    --augment \
    --output outputs/datasets/snake_dataset_hf \
    --seed 42

echo
echo "Step 2: Training model with Muon optimizer..."
python train_hf.py \
    --dataset outputs/datasets/snake_dataset_hf \
    --d-model 128 \
    --num-layers 2 \
    --num-heads 4 \
    --dropout 0.1 \
    --epochs 20 \
    --batch-size 256 \
    --optimizer muon \
    --muon-lr 0.02 \
    --muon-momentum 0.95 \
    --warmup-ratio 0.1 \
    --lr-scheduler cosine \
    --output-dir outputs/snake_hf_output \
    --wandb \
    --wandb-project snake-hf \
    --seed 42

echo
echo "======================================"
echo "Training complete!"
echo "======================================"
echo
echo "To use AdamW instead of Muon, replace:"
echo "  --optimizer muon --muon-lr 0.02"
echo "with:"
echo "  --optimizer adamw --lr 1e-3"
