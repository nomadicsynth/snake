# Snake HuggingFace Training

PyTorch + HuggingFace conversion of the Snake RL pretraining pipeline.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate dataset and train
./quickstart_hf.sh
```

## Manual Usage

### 1. Generate Dataset

```bash
python generate_dataset_hf.py \
    --num-samples 50000 \
    --width 20 \
    --height 20 \
    --use-astar \
    --augment \
    --output snake_dataset_hf
```

### 2. Train Model

**With Muon optimizer (recommended):**

```bash
python train_hf.py \
    --dataset snake_dataset_hf \
    --d-model 64 \
    --num-layers 2 \
    --num-heads 4 \
    --optimizer muon \
    --muon-lr 0.02 \
    --muon-momentum 0.95 \
    --epochs 20 \
    --batch-size 256 \
    --warmup-ratio 0.1 \
    --lr-scheduler cosine \
    --wandb
```

**With AdamW optimizer:**

```bash
python train_hf.py \
    --dataset snake_dataset_hf \
    --d-model 64 \
    --num-layers 2 \
    --num-heads 4 \
    --optimizer adamw \
    --lr 1e-3 \
    --epochs 20 \
    --batch-size 256 \
    --wandb
```

## Key Files

- `model_pytorch.py` - PyTorch Transformer policy network
- `generate_dataset_hf.py` - Generate dataset in HuggingFace format
- `train_hf.py` - Training script using HuggingFace Trainer
- `quickstart_hf.sh` - One-command setup and training

## Features

- ✅ PyTorch native implementation
- ✅ HuggingFace Datasets integration
- ✅ HuggingFace Trainer with full training loop
- ✅ Muon optimizer support (drop-in replacement)
- ✅ AdamW optimizer support
- ✅ Reasoning Snake Model (RSM) support
- ✅ CNN encoder option
- ✅ Mixed precision (FP16/BF16)
- ✅ Weights & Biases integration
- ✅ Learning rate scheduling (linear, cosine, constant)
- ✅ Gradient clipping
- ✅ Checkpointing and resuming

## Optimizer Notes

**Muon** is a momentum-based optimizer optimized for transformers. It's used by passing `--optimizer muon` and typically requires higher learning rates (0.01-0.02) than AdamW.

The implementation uses HuggingFace Trainer's `optimizers` parameter to inject Muon directly - it just works!

## Architecture

The transformer processes the Snake game state (20×20×3 grid) as a sequence of tokens:
- Each grid cell becomes a token
- 2D positional encoding preserves spatial structure
- Optional CNN encoder for spatial feature extraction
- Multi-head self-attention captures game patterns
- Outputs action logits (4 actions: up, right, down, left)
