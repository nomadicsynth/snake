# Learning Rate Tracking and Annealing

## Summary

Added comprehensive learning rate tracking to the training metrics and implemented learning rate annealing for the Muon optimizer.

## Changes Made

### 1. Learning Rate Metrics (`train_snake_purejaxrl_impl.py`)

All training functions now track and report the current learning rate(s) in the metrics:

- **`learning_rate`**: The current learning rate for the primary optimizer
  - For Adam: tracks the Adam learning rate (with or without annealing)
  - For Muon: tracks the Muon optimizer's learning rate for weight matrices

- **`aux_learning_rate`**: (Muon only) The learning rate for auxiliary parameters (biases, embeddings, etc.)

### 2. Muon Optimizer Annealing (`muon_jax.py`)

Updated the Muon optimizer functions to support learning rate schedules:

- `multi_transform_with_muon()`: Now accepts either float or callable (schedule) for both `muon_lr` and `aux_lr`
- `chain_with_muon()`: Now accepts either float or callable (schedule) for both `muon_lr` and `aux_lr`

### 3. Learning Rate Schedules (`train_snake_purejaxrl_impl.py`)

Added schedule functions for all training modes:

**For `make_train_step()`:**
- `linear_schedule(count)`: Standard Adam LR annealing
- `muon_schedule(count)`: Muon LR annealing (respects `ANNEAL_LR` flag)
- `aux_schedule(count)`: Auxiliary Adam LR annealing for Muon (respects `ANNEAL_LR` flag)

**For `make_train_with_callback()`:**
- Same three schedule functions with adjusted counting for the callback version

### 4. WandB Logging (`train_snake_purejaxrl.py`)

Added learning rate logging to WandB:

- `train/learning_rate`: Logged every update
- `train/aux_learning_rate`: Logged every update (when using Muon)

These metrics allow you to:
- Verify that learning rate annealing is working correctly
- Compare training runs with different LR schedules
- Debug convergence issues related to learning rate

## Usage

### Standard Adam with Annealing (default)

```bash
python train_snake_purejaxrl.py \
    --lr 2.5e-4 \
    --anneal-lr \
    --wandb
```

The learning rate will linearly decrease from `2.5e-4` to `0` over the course of training.

### Muon Optimizer with Annealing

```bash
python train_snake_purejaxrl.py \
    --use-muon \
    --muon-lr 0.02 \
    --aux-adam-lr 2.5e-4 \
    --anneal-lr \
    --wandb
```

Both learning rates will linearly decrease:
- `muon_lr`: `0.02` → `0` (for weight matrices)
- `aux_adam_lr`: `2.5e-4` → `0` (for biases, embeddings, etc.)

### Muon Optimizer WITHOUT Annealing

```bash
python train_snake_purejaxrl.py \
    --use-muon \
    --muon-lr 0.02 \
    --aux-adam-lr 2.5e-4 \
    --wandb
```

Both learning rates remain constant throughout training.

## Testing

Run the test suite to verify learning rate tracking and annealing:

```bash
python test_lr_metrics.py
```

This test verifies:
1. ✅ Learning rate appears in metrics
2. ✅ LR annealing works correctly for Adam
3. ✅ LR annealing works correctly for Muon (both muon_lr and aux_lr)
4. ✅ Both LRs are tracked separately for Muon

## WandB Visualization

In your WandB dashboard, you can now plot:

- `train/learning_rate` over time
- `train/aux_learning_rate` over time (Muon only)

This helps you understand how the learning rate schedule affects training dynamics and convergence.

## Implementation Details

### Linear Schedule Formula

```python
frac = 1.0 - update_idx / num_updates
current_lr = initial_lr * frac
```

This linearly anneals from `initial_lr` at the start to `0` at the end of training.

### Schedule Functions as Callables

Optax optimizers accept either:
- A float: constant learning rate
- A callable `schedule(count) -> float`: dynamic learning rate based on optimizer step count

The Muon implementation now supports both formats, making it consistent with standard Optax optimizers.
