# ✅ Muon Optimizer Added to Sweep Configuration

## Summary

Successfully added Muon optimizer as an option in the WandB sweep configuration, allowing the sweep to compare Adam vs Muon optimizer performance.

## Changes Made

### 1. Updated `train_snake_purejaxrl.py`

**Added boolean argument parser:**

```python
def str2bool(v):
    """Convert string to boolean for argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
```

**Updated `--use-muon` argument:**

```python
# Before
parser.add_argument("--use-muon", "--use_muon", action="store_true", default=False, ...)

# After
parser.add_argument("--use-muon", "--use_muon", type=str2bool, nargs='?', const=True, default=False, ...)
```

This allows the sweep to pass `--use_muon=true` or `--use_muon=false` as values instead of just flags.

### 2. Updated `wandb_sweep_jax_ppo_eval.yaml`

**Added Muon parameters:**

```yaml
parameters:
  # Optimizer selection
  use_muon:
    values: [true, false]
  
  # Learning rate (for Adam optimizer)
  lr:
    distribution: log_uniform_values
    min: 0.00001
    max: 0.001
  
  # Muon optimizer parameters (only used when use_muon=true)
  muon_lr:
    distribution: log_uniform_values
    min: 0.001
    max: 0.1
  
  aux_adam_lr:
    distribution: log_uniform_values
    min: 0.00001
    max: 0.001
  
  muon_momentum:
    distribution: uniform
    min: 0.90
    max: 0.99
```

## How It Works

### Optimizer Selection

The sweep will test both optimizers:

- **`use_muon=false`**: Uses Adam optimizer with `lr` parameter
- **`use_muon=true`**: Uses Muon optimizer with `muon_lr`, `aux_adam_lr`, and `muon_momentum`

### Parameter Behavior

When **Adam** is selected (`use_muon=false`):

- Uses `lr` for learning rate
- Ignores `muon_lr`, `aux_adam_lr`, `muon_momentum`

When **Muon** is selected (`use_muon=true`):

- Uses `muon_lr` for weight matrix learning rate (typically higher, 0.001-0.1)
- Uses `aux_adam_lr` for auxiliary parameters (biases, norms, etc.)
- Uses `muon_momentum` for momentum value
- Ignores `lr` parameter

### Expected Performance

Based on previous research:

- **Muon**: Often faster convergence with higher learning rates
- **Adam**: More stable, well-tested baseline
- The sweep will determine which works better for Snake!

## Testing

Verified the changes work correctly:

```bash
$ python train_snake_purejaxrl.py --use_muon=true --muon_lr=0.02 --aux_adam_lr=0.0001 --muon_momentum=0.95 --eval_freq=0

Configuration:
  ...
  Optimizer: Muon
    Muon LR (weights): 2.00e-02
    Adam LR (aux): 1.00e-04
    Momentum: 0.95
  ...
```

✅ **Muon parameters are correctly recognized and used!**

## Running the Sweep

```bash
# Create sweep (same as before)
wandb sweep wandb_sweep_jax_ppo_eval.yaml

# Run agent(s)
wandb agent <sweep-id>
```

The sweep will now automatically test both Adam and Muon optimizers with different hyperparameter combinations.

## Expected Outcomes

The sweep will reveal:

1. **Which optimizer is better for Snake** (Adam vs Muon)
2. **Optimal learning rates** for each optimizer
3. **Best hyperparameter combinations** for each optimizer
4. **Performance differences** between the two approaches

## Hyperparameter Ranges

### Adam (when `use_muon=false`)

- `lr`: 1e-5 to 1e-3 (log uniform)

### Muon (when `use_muon=true`)

- `muon_lr`: 0.001 to 0.1 (log uniform) - for weight matrices
- `aux_adam_lr`: 1e-5 to 1e-3 (log uniform) - for auxiliary params
- `muon_momentum`: 0.90 to 0.99 (uniform)

Note: Muon typically uses higher learning rates (0.01-0.05) than Adam.

## Documentation

- See [MUON_OPTIMIZER.md](MUON_OPTIMIZER.md) for details about Muon
- See [docs/SWEEP_GUIDE.md](docs/SWEEP_GUIDE.md) for sweep usage
- See [EVALUATION_SUMMARY.md](EVALUATION_SUMMARY.md) for evaluation metrics

---

## Ready to discover if Muon beats Adam for Snake RL! 🚀⚡
