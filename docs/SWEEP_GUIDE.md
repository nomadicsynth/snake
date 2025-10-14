# Running Hyperparameter Sweeps with Evaluation

This guide explains how to run WandB sweeps with the new evaluation metrics for reliable hyperparameter optimization.

## Quick Start

### 1. Initialize the Sweep

```bash
wandb sweep wandb_sweep_jax_ppo_eval.yaml
```

This will output a sweep ID like: `your-entity/snake-jax-ppo-sweep/abc123xyz`

### 2. Run Sweep Agents

```bash
# Single agent
wandb agent your-entity/snake-jax-ppo-sweep/abc123xyz

# Multiple agents in parallel (if you have multiple GPUs)
wandb agent your-entity/snake-jax-ppo-sweep/abc123xyz &
wandb agent your-entity/snake-jax-ppo-sweep/abc123xyz &
```

### 3. Monitor Results

Visit WandB to see:
- Real-time sweep progress
- Hyperparameter importance
- Best configurations
- Evaluation metric trends

## Why Evaluation Metrics?

### Training Metrics (`episode/*`)
- ❌ Noisy (exploration adds variance)
- ❌ Not comparable across runs (stochastic)
- ✅ Show learning dynamics
- ✅ High throughput (collected during training)

### Evaluation Metrics (`eval/*`)
- ✅ Stable (greedy policy, no exploration)
- ✅ Comparable across runs
- ✅ Reliable for optimization
- ❌ Lower throughput (separate rollouts)

**For sweeps, always optimize on `eval/mean_return`!**

## Sweep Configuration

The sweep is configured in `wandb_sweep_jax_ppo_eval.yaml`:

```yaml
metric:
  name: eval/mean_return  # Optimize this!
  goal: maximize

parameters:
  eval_freq:
    value: 25  # Evaluate every 25 updates
  eval_episodes:
    value: 256  # 256 episodes for stable estimates
```

### Key Settings

- **`eval_freq: 25`**: Evaluate every 25 updates (every ~650k steps)
  - More frequent = better tracking but slower
  - Less frequent = faster but coarser tracking

- **`eval_episodes: 256`**: Run 256 greedy episodes per evaluation
  - More episodes = more stable metrics but slower
  - Fewer episodes = faster but noisier

## Customizing the Sweep

### Focus on Specific Hyperparameters

Edit `wandb_sweep_jax_ppo_eval.yaml` to focus on specific parameters:

```yaml
# Example: Only tune learning rate and network size
parameters:
  lr:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-3
  
  d_model:
    values: [32, 64, 128, 256]
  
  # Fix other parameters
  gamma:
    value: 0.99
  clip_eps:
    value: 0.2
  # ... etc
```

### Adjust Training Budget

```yaml
parameters:
  total_timesteps:
    value: 10_000_000  # Train longer
  
  eval_freq:
    value: 50  # Evaluate less often for speed
```

### Try Different Optimizers

```yaml
# Test Muon optimizer
parameters:
  use_muon:
    values: [true, false]
  
  muon_lr:
    distribution: log_uniform_values
    min: 0.001
    max: 0.1
  
  muon_momentum:
    values: [0.90, 0.95, 0.99]
```

## Early Stopping

The sweep uses Hyperband early stopping to kill underperforming runs:

```yaml
early_terminate:
  type: hyperband
  min_iter: 50  # At least 50 evaluations (50 * 25 = 1,250 updates)
  eta: 2
  s: 3
```

This saves compute by stopping bad configurations early.

## Interpreting Results

### Primary Metrics

1. **`eval/mean_return`**: Main optimization target
   - Higher is better
   - Best model automatically saved

2. **`eval/mean_score`**: Apples eaten
   - Easier to interpret than raw returns
   - Integer metric (0, 1, 2, ...)

3. **`eval/std_return`**: Performance variance
   - Lower is better (more consistent)
   - High variance may indicate instability

### Secondary Metrics

- **`eval/max_return`**: Best-case performance
- **`eval/max_score`**: Best score achieved
- **`eval/min_return`**: Worst-case performance

### Comparing to Training

Plot both together in WandB to see:
- Training metrics (noisy, with exploration)
- Eval metrics (stable, greedy)

The gap shows exploration overhead.

## Best Practices

### 1. Start with a Quick Sweep

```yaml
parameters:
  total_timesteps:
    value: 1_000_000  # Quick runs
  eval_freq:
    value: 50
  eval_episodes:
    value: 128
```

Find promising regions fast, then refine.

### 2. Increase Budget for Final Sweep

```yaml
parameters:
  total_timesteps:
    value: 10_000_000  # Longer training
  eval_freq:
    value: 25
  eval_episodes:
    value: 512  # More stable
```

### 3. Run Multiple Seeds

Once you find good hyperparameters, test stability:

```bash
for seed in 42 123 456 789 1337; do
  python train_snake_purejaxrl.py \
    --wandb \
    --eval-freq 25 \
    --eval-episodes 256 \
    --lr 0.0001 \
    --d-model 128 \
    --seed $seed \
    --run-name "best_config_seed_${seed}"
done
```

### 4. Monitor GPU Usage

Evaluation adds overhead. Monitor with:

```bash
watch -n 1 nvidia-smi
```

If GPU usage drops during evaluation, consider:
- Decreasing `eval_episodes`
- Increasing `eval_freq` (evaluate less often)

## Example Commands

### Manual Test with Evaluation

```bash
# Test a specific configuration with evaluation
python train_snake_purejaxrl.py \
  --wandb \
  --eval-freq 25 \
  --eval-episodes 256 \
  --total-timesteps 5_000_000 \
  --lr 2.5e-4 \
  --d-model 128 \
  --num-layers 2 \
  --num-heads 4
```

### Grid Search (Simple)

Create a simpler sweep config for grid search:

```yaml
method: grid
parameters:
  lr:
    values: [1e-4, 2.5e-4, 5e-4]
  d_model:
    values: [64, 128]
```

### Random Search

```yaml
method: random
parameters:
  lr:
    distribution: log_uniform_values
    min: 1e-5
    max: 1e-3
  # ... etc
```

## Troubleshooting

### Evaluation is Too Slow

Reduce overhead:
```yaml
eval_freq: 100  # Less frequent
eval_episodes: 64  # Fewer episodes
```

### Metrics are Noisy

Increase stability:
```yaml
eval_episodes: 512  # More episodes
```

### Out of Memory

Reduce batch size during evaluation (currently fixed at `num_episodes`). The episodes run in parallel via `vmap`.

If needed, modify `make_evaluate_fn` to batch evaluations.

## Related Files

- `train_snake_purejaxrl.py` - Main training script
- `train_snake_purejaxrl_impl.py` - Contains `make_evaluate_fn`
- `wandb_sweep_jax_ppo_eval.yaml` - Sweep configuration
- `docs/EVALUATION_METRICS.md` - Detailed metric explanations
- `docs/WANDB_INTEGRATION.md` - WandB setup

## Summary

✅ **Do**: Use `eval/mean_return` for sweep optimization
✅ **Do**: Balance eval frequency vs training speed
✅ **Do**: Use more eval episodes for final sweeps
❌ **Don't**: Optimize on training metrics (too noisy)
❌ **Don't**: Evaluate too frequently (slows training)

Happy sweeping! 🚀
