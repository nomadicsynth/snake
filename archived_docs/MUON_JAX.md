# Muon Optimizer for JAX/PureJaxRL

## Overview

The Muon (Momentum Orthogonalized by Newton-schulz) optimizer has been implemented for JAX and integrated into `train_snake_purejaxrl.py`. This provides the same benefits as the PyTorch version but for GPU-native JAX training.

## Implementation

The JAX implementation is in `muon_jax.py` and provides:

- **`muon()`**: Core Muon optimizer using Newton-Schulz orthogonalization
- **`multi_transform_with_muon()`**: Applies Muon to weight matrices and Adam to auxiliary params
- **`chain_with_muon()`**: Combines gradient clipping with Muon/Adam (recommended)

### Key Features

1. **Newton-Schulz Orthogonalization**: Iteratively orthogonalizes momentum updates for 2D+ weight matrices
2. **Automatic Parameter Separation**:
   - 2D+ tensors (weight matrices) → Muon optimizer
   - 1D tensors (biases, layer norms) → Adam optimizer
3. **Nesterov Momentum**: Optional Nesterov-style momentum updates
4. **Learning Rate Schedules**: Compatible with optax learning rate schedules

## Usage

### Command-Line Arguments

```bash
python train_snake_purejaxrl.py \
    --use-muon \
    --muon-lr 0.02 \
    --aux-adam-lr 0.0003 \
    --muon-momentum 0.95 \
    --muon-nesterov
```

### Arguments

- `--use-muon`: Enable Muon optimizer (default: False)
- `--muon-lr`: Learning rate for weight matrices (default: 0.02)
- `--aux-adam-lr`: Learning rate for auxiliary params (default: uses `--lr`)
- `--muon-momentum`: Momentum coefficient (default: 0.95)
- `--muon-nesterov`: Use Nesterov momentum (default: True)

### Example Training Commands

#### Basic Muon Training

```bash
python train_snake_purejaxrl.py \
    --wandb \
    --use-muon \
    --muon-lr 0.02 \
    --aux-adam-lr 0.00025 \
    --total-timesteps 10000000
```

#### Muon with Larger Network

```bash
python train_snake_purejaxrl.py \
    --wandb \
    --use-muon \
    --muon-lr 0.03 \
    --aux-adam-lr 0.0003 \
    --d-model 128 \
    --num-layers 4 \
    --num-heads 8 \
    --total-timesteps 20000000
```

#### Without Muon (Standard Adam)

```bash
python train_snake_purejaxrl.py \
    --wandb \
    --lr 0.00025 \
    --total-timesteps 10000000
```

## Hyperparameter Tuning with W&B Sweeps

The updated `wandb_sweep_jax_ppo.yaml` includes Muon parameters:

```bash
# Initialize sweep
wandb sweep wandb_sweep_jax_ppo.yaml

# Run sweep agent
wandb agent <sweep-id>
```

The sweep will optimize:

- `muon-lr`: 0.01 - 0.05
- `aux-adam-lr`: 0.00005 - 0.001  
- `muon-momentum`: 0.90 - 0.99
- Plus all standard PPO and network architecture parameters

## Benefits

### 1. Faster Convergence

Muon's orthogonalization helps maintain stable weight matrices, leading to:

- Faster training convergence
- More stable gradient flow through transformer layers
- Better handling of vanishing/exploding gradients

### 2. Better Performance

- Can achieve higher final performance on complex tasks
- More robust to hyperparameter choices
- Works well with deeper transformer architectures

### 3. GPU-Native Training

- Fully compatible with JAX's JIT compilation
- No performance overhead on GPU
- Scales efficiently with batch size and model size

## Technical Details

### Newton-Schulz Iteration

The orthogonalization process iteratively refines the momentum matrix $M$ using:

$$M_{t+1} = \frac{3}{2}M_t - \frac{1}{2}M_t M_t^T M_t$$

This converges to an orthogonal matrix, which helps preserve the geometry of the weight space during optimization.

### Parameter Grouping

```python
# Weight matrices (2D+): Use Muon
- Transformer attention weights: Q, K, V projections
- Feed-forward layers: Linear transformations
- Output projection heads

# Auxiliary params (1D): Use Adam
- Biases and layer norm scales
- Embedding parameters
- Any 1D parameter tensors
```

### Integration with PPO

The Muon optimizer is integrated into the PPO training loop via optax:

```python
tx = chain_with_muon(
    muon_lr=0.02,
    aux_lr=0.00025,
    max_grad_norm=0.5,
    momentum=0.95,
    nesterov=True,
)
```

This creates a gradient transformation that:

1. Clips gradients by global norm
2. Applies Muon to weight matrices
3. Applies Adam to auxiliary parameters

## Comparison: PyTorch vs JAX Muon

| Feature | PyTorch (sb3_snake.py) | JAX (train_snake_purejaxrl.py) |
|---------|------------------------|--------------------------------|
| Backend | torch.optim | optax |
| Parameter detection | ndim >= 2 | ndim >= 2 |
| Orthogonalization | Newton-Schulz | Newton-Schulz (JAX) |
| Nesterov momentum | ✓ | ✓ |
| Learning rate schedules | ✓ | ✓ |
| Gradient clipping | Pre-optimizer | Chained with optimizer |
| JIT compilation | torch.compile | jax.jit (full training loop) |

## Performance Tips

### Learning Rates

- **Muon LR**: Start with 0.02, range 0.01-0.05
  - Higher for smaller models
  - Lower for larger models or deeper networks
  
- **Aux Adam LR**: Typically 10x smaller than standard Adam
  - Start with 0.0002-0.0005
  - Scale with model size

### Momentum

- Default 0.95 works well for most cases
- Increase to 0.97-0.99 for more stable training
- Decrease to 0.90-0.93 for faster adaptation

### When to Use Muon

✅ **Good for:**

- Transformer architectures
- Deep networks (3+ layers)
- Long training runs
- Tasks requiring stable learning

❌ **May not help:**

- Very small models (1-2 layers)
- Short training runs (< 1M steps)
- Simple MLP architectures

## Troubleshooting

### "muon_jax not available"

Make sure `muon_jax.py` is in the same directory as `train_snake_purejaxrl.py`.

### NaN/Inf in training

- Reduce `muon-lr` (try 0.01 or lower)
- Decrease `muon-momentum` (try 0.90)
- Check gradient clipping: increase `--max-grad-norm`

### Slower than Adam

- Muon adds computation overhead for orthogonalization
- Benefits appear after ~100K-1M steps
- Try longer training runs to see convergence improvements

## References

- Original Muon: <https://github.com/KellerJordan/Muon>
- Optax documentation: <https://optax.readthedocs.io/>
- PureJaxRL: <https://github.com/luchris429/purejaxrl>
