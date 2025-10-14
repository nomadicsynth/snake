# Evaluation Metrics

This document explains the evaluation metrics used during training for hyperparameter sweeps and model selection.

## Overview

Evaluation runs periodically during training (controlled by `--eval-freq`) to assess the policy's performance using **greedy action selection** (no exploration). This provides a more stable and comparable measure of learned behavior than training episode returns, which include exploration noise.

## Evaluation Metrics

### Core Metrics

- **`eval/mean_return`**: Average cumulative reward across evaluation episodes
  - Primary metric for model selection
  - More stable than training returns (no exploration)
  - Best model is saved based on this metric

- **`eval/std_return`**: Standard deviation of returns
  - Measures consistency of performance
  - Lower is generally better (more reliable policy)

- **`eval/mean_score`**: Average number of apples eaten per episode
  - Direct measure of game performance
  - Integer metric (0, 1, 2, 3, ...)
  - Easier to interpret than raw returns

### Performance Bounds

- **`eval/max_return`**: Best return achieved in any evaluation episode
  - Shows the policy's peak performance
  - Useful for understanding best-case behavior

- **`eval/max_score`**: Best score (apples) in any evaluation episode
  - Maximum apples eaten in a single episode
  - Good indicator of mastery level

- **`eval/min_return`**: Worst return in any evaluation episode
  - Shows worst-case performance
  - Useful for detecting failure modes

### Best Model Tracking

- **`eval/best_return`**: Best `eval/mean_return` seen so far during training
  - Used for model checkpointing
  - Monotonically increasing
  - Final value indicates peak performance achieved

## Usage

### Basic Evaluation

```bash
# Enable evaluation every 50 updates with 128 episodes
python train_snake_purejaxrl.py \
    --wandb \
    --eval-freq 50 \
    --eval-episodes 128
```

### For Sweeps

```bash
# More frequent evaluation for fine-grained tracking
python train_snake_purejaxrl.py \
    --wandb \
    --eval-freq 25 \
    --eval-episodes 256 \
    --total-timesteps 10_000_000
```

### Disable Evaluation

```bash
# Set eval-freq to 0 to disable (faster training)
python train_snake_purejaxrl.py \
    --wandb \
    --eval-freq 0
```

## Configuration

- **`--eval-freq N`**: Evaluate every N updates (default: 50, 0 to disable)
- **`--eval-episodes N`**: Number of episodes per evaluation (default: 128)

## Evaluation vs Training Metrics

### Training Metrics (`episode/*`)
- Collected during normal training rollouts
- Include exploration (stochastic policy sampling)
- More noisy but higher throughput
- Reflect actual training experience

### Evaluation Metrics (`eval/*`)
- Run separately with greedy action selection (argmax)
- No exploration noise
- Lower throughput (separate episodes)
- Better for comparing models and hyperparameters

## Best Practices

1. **Use evaluation for sweeps**: Training metrics are too noisy for reliable hyperparameter comparison
2. **Balance frequency vs cost**: More frequent evaluation = better tracking but slower overall training
3. **Increase episodes for stability**: More episodes = more reliable metrics but more compute
4. **Monitor both**: Training metrics show learning dynamics, eval metrics show true performance

## Model Checkpointing

The best model (highest `eval/mean_return`) is automatically saved to:
```
models/{run_name}/best_model.pkl
```

This can be used for:
- Deployment
- Further analysis
- Transfer learning
- Comparison across runs

## Example Output

```
Configuration:
  Run name: snake_jax_10x10_20251014_120000
  ...
  Evaluation: every 50 updates, 128 episodes

Training: 100%|████████| 5M/5M [10:23<00:00, 8023steps/s]

✅ TRAINING COMPLETE!

⏱️  Timing:
  Training: 623.45s
  
📊 Performance:
  Total steps: 5,000,000
  Training FPS: 8,023
  Time per update: 0.129s
  Best eval return: 42.35
```

## WandB Integration

All metrics are logged to WandB under the `eval/` namespace:
- Allows easy comparison across runs
- Automated best model tracking
- Visual comparison with training metrics
- Sweep optimization based on `eval/mean_return`

## Related Documentation

- [METRICS_EXPLAINED.md](../METRICS_EXPLAINED.md) - Training metrics
- [WANDB_INTEGRATION.md](WANDB_INTEGRATION.md) - WandB setup and usage
