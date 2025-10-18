#!/bin/bash
# Quick start script for Snake pretraining pipeline
# Usage: ./quickstart_pretrain.sh

set -e  # Exit on error

echo "=========================================="
echo "SNAKE TRANSFORMER PRETRAINING PIPELINE"
echo "=========================================="
echo ""

# Configuration
NUM_SAMPLES=${NUM_SAMPLES:-10000}  # Small dataset for quick test
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-256}
DEVICE=${DEVICE:-cuda}

# File paths
DATASET_PATH="snake_pretrain_dataset.pkl"
OUTPUT_DIR="pretrain_output"

echo "Configuration:"
echo "  Dataset samples: $NUM_SAMPLES (x8 with augmentation = $(($NUM_SAMPLES * 8)))"
echo "  Training epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Device: $DEVICE"
echo ""

# Step 1: Generate dataset
echo "=========================================="
echo "Step 1: Generating dataset..."
echo "=========================================="
echo ""

if [ -f "$DATASET_PATH" ]; then
    echo "Dataset already exists at $DATASET_PATH"
    read -p "Regenerate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm "$DATASET_PATH"
        python generate_dataset.py \
            --num-samples $NUM_SAMPLES \
            --use-astar \
            --augment \
            --output "$DATASET_PATH" \
            --seed 42
    fi
else
    python generate_dataset.py \
        --num-samples $NUM_SAMPLES \
        --use-astar \
        --augment \
        --output "$DATASET_PATH" \
        --seed 42
fi

echo ""
echo "✓ Dataset generated!"
echo ""

# Step 2: Pretrain model
echo "=========================================="
echo "Step 2: Pretraining transformer..."
echo "=========================================="
echo ""

python pretrain_snake.py \
    --dataset "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr 3e-4 \
    --warmup-epochs 5 \
    --label-smoothing 0.1 \
    --device $DEVICE \
    --seed 42 \
    --num-workers 4

echo ""
echo "✓ Pretraining complete!"
echo ""

# Step 3: Evaluate pretrained model
echo "=========================================="
echo "Step 3: Evaluating pretrained model..."
echo "=========================================="
echo ""

python pretrain_integration.py "$OUTPUT_DIR/best_model.pth"

echo ""
echo "✓ Evaluation complete!"
echo ""

# Step 4: Show next steps
echo "=========================================="
echo "SUCCESS! Next steps:"
echo "=========================================="
echo ""
echo "1. View training curves:"
echo "   open $OUTPUT_DIR/training_curves.png"
echo ""
echo "2. Use pretrained model with SB3:"
echo "   python -c '"
echo "from pretrain_integration import create_sb3_model_with_pretrained_weights"
echo "from snake import SnakeGame"
echo "env = SnakeGame(width=20, height=20)"
echo "model = create_sb3_model_with_pretrained_weights("
echo "    env, \"$OUTPUT_DIR/best_model.pth\", algorithm=\"DQN\")"
echo "model.learn(total_timesteps=500000)"
echo "model.save(\"snake_finetuned\")"
echo "'"
echo ""
echo "3. Or run the standard SB3 training script:"
echo "   python sb3_snake.py --load-pretrained $OUTPUT_DIR/best_model.pth"
echo ""
echo "Files created:"
echo "  - $DATASET_PATH (dataset)"
echo "  - $OUTPUT_DIR/best_model.pth (best model)"
echo "  - $OUTPUT_DIR/final_model.pth (final model)"
echo "  - $OUTPUT_DIR/config.json (configuration)"
echo "  - $OUTPUT_DIR/history.json (training history)"
echo "  - $OUTPUT_DIR/training_curves.png (plots)"
echo ""
echo "=========================================="
echo "DONE!"
echo "=========================================="
