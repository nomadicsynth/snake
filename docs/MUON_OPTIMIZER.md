# Muon Optimizer Implementation

## Overview

The Muon optimizer has been successfully integrated into both the DQN (`sb3_snake.py`) and PPO (`sb3_snake_ppo.py`) training scripts. Muon is a momentum-based optimizer designed specifically for transformer models that can provide faster convergence and better performance compared to traditional Adam/AdamW optimizers.

## Implementation Details

### Architecture

The implementation follows the pattern from `example-train-with-muon.py`:

1. **Parameter Separation**: Model parameters are divided into two groups:
   - **Hidden weights** (2D+ tensors): Use Muon optimizer with custom learning rate
   - **Auxiliary parameters** (1D tensors, embeddings, biases): Use Adam-like optimization

2. **Optimizer Factory**: A factory function `make_muon_optimizer_factory()` creates custom optimizer instances.

3. **Post-Creation Replacement**: The model is created with default Adam optimizer, then the optimizer is replaced with Muon after initialization. This is necessary because stable-baselines3 doesn't support custom optimizers in the constructor.

### Command-Line Arguments

Three new arguments have been added to both scripts:

- `--use-muon`: Enable Muon optimizer (default: False)
- `--muon-lr`: Learning rate for hidden weights (default: 0.02)
- `--aux-adam-lr`: Learning rate for auxiliary parameters (default: uses `--lr` value)

### Usage Examples

#### DQN with Muon

```bash
python sb3_snake.py train \
    --use-muon \
    --muon-lr 0.02 \
    --lr 0.0001 \
    --episodes 1000 \
    --batch-size 64
```

#### PPO with Muon

```bash
python sb3_snake_ppo.py train \
    --use-muon \
    --muon-lr 0.02 \
    --aux-adam-lr 0.0003 \
    --episodes 1000 \
    --batch-size 64
```

### Configuration Details

**Hidden Weight Parameters (Muon)**:

- All 2D+ tensors in the model
- Transformer attention weights, feed-forward layers
- Learning rate: `--muon-lr` (default: 0.02)
- Momentum: 0.95 (Muon default)

**Auxiliary Parameters (Adam-like)**:

- 1D tensors (biases, layer norms)
- Embedding layers
- Output projection heads
- Learning rate: `--aux-adam-lr` or fallback to `--lr`
- Momentum: 0.95

## Benefits

1. **Faster Convergence**: Muon is optimized for transformer architectures
2. **Better Performance**: Can achieve better final performance on some tasks
3. **Flexible Configuration**: Separate learning rates for different parameter groups
4. **Drop-in Replacement**: Works seamlessly with existing training configurations

## Installation

The Muon optimizer requires the `muon` package:

```bash
pip install muon
```

If Muon is not available, the scripts will automatically fall back to the standard Adam optimizer with a warning message.

## Training Output

When Muon is enabled, the training configuration summary will show:

```text
============================================================
Training Configuration Summary:
============================================================
Batch size: 64
Optimizer: Muon
  Muon LR (hidden): 0.02
  Adam LR (aux): 0.0001
torch.compile: ✓ Enabled
FlashAttention: ✓ Enabled
BF16 (mixed precision): ✓ Enabled
============================================================
```

The optimizer will also print the number of parameters in each group:

```text
Muon optimizer: 42 hidden weight tensors (lr=0.02), 18 auxiliary params (lr=0.0001)
```

## Compatibility

- Works with both DQN and PPO algorithms
- Compatible with all existing features (curriculum learning, loop detection, W&B logging, etc.)
- Can be combined with other optimizations (torch.compile, FlashAttention, BF16)

## Notes

- The default Muon learning rate (0.02) is typically much higher than Adam learning rates
- The auxiliary learning rate should usually match your regular `--lr` setting
- Experiment with different learning rate combinations for optimal performance
- Muon is particularly effective for larger transformer models
