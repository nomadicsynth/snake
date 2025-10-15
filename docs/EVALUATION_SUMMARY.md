# Evaluation Feature Summary

## What's New

Added comprehensive evaluation metrics to the training pipeline for reliable hyperparameter sweeps and model selection.

## Key Features

### 1. Greedy Evaluation During Training
- Runs periodically during training (configurable frequency)
- Uses greedy action selection (argmax) instead of sampling
- Provides stable, comparable metrics across runs
- Automatically saves best model based on eval performance

### 2. Rich Evaluation Metrics
- `eval/mean_return` - Primary metric for optimization
- `eval/mean_score` - Average apples eaten (easier to interpret)
- `eval/std_return` - Performance consistency
- `eval/max_return`, `eval/max_score`, `eval/min_return` - Performance bounds
- `eval/best_return` - Best performance seen during training

### 3. WandB Integration
- All eval metrics logged to WandB under `eval/` namespace
- Best model automatically tracked and saved
- Ready for sweep optimization

## Usage

### Basic Training with Evaluation

```bash
python train_snake_purejaxrl.py \
    --wandb \
    --eval-freq 50 \
    --eval-episodes 128
```

### For Hyperparameter Sweeps

```bash
# Initialize sweep
wandb sweep wandb_sweep_jax_ppo_eval.yaml

# Run agents
wandb agent your-entity/project/sweep-id
```

### Disable Evaluation (Faster Training)

```bash
python train_snake_purejaxrl.py --eval-freq 0
```

## Configuration Options

- `--eval-freq N` - Evaluate every N updates (default: 50, 0 to disable)
- `--eval-episodes N` - Episodes per evaluation (default: 128)

## Files Added/Modified

### New Files
- `train_snake_purejaxrl_impl.py` - Added `make_evaluate_fn()`
- `test_eval.py` - Test script for evaluation function
- `test_eval_training.sh` - Example training with eval
- `wandb_sweep_jax_ppo_eval.yaml` - Sweep config using eval metrics
- `docs/EVALUATION_METRICS.md` - Detailed metric documentation
- `docs/SWEEP_GUIDE.md` - Complete guide for running sweeps

### Modified Files
- `train_snake_purejaxrl.py` - Integrated evaluation into training loop

## Performance Impact

- **Evaluation overhead**: ~1-5% depending on `eval_freq` and `eval_episodes`
- **GPU utilization**: Evaluation runs on GPU via JIT-compiled function
- **Compilation**: First evaluation triggers JIT compilation (one-time cost)

## Why This Matters for Sweeps

### Problem: Training Metrics Are Noisy
- Training uses stochastic policy (exploration)
- High variance across runs
- Hard to compare hyperparameters reliably

### Solution: Evaluation Metrics Are Stable
- Greedy policy (no exploration)
- Low variance, comparable across runs
- Reliable signal for hyperparameter optimization

### Result
- **Better sweep results**: Find truly optimal hyperparameters
- **Faster convergence**: Early stopping based on stable metrics
- **Reproducible**: Consistent evaluation across runs

## Example Output

```
Configuration:
  ...
  Evaluation: every 50 updates, 128 episodes

Training: 100%|████████| 5M/5M [10:23<00:00, 8023steps/s, eval_ret=42.35, best=42.35]

✅ TRAINING COMPLETE!

📊 Performance:
  Total steps: 5,000,000
  Training FPS: 8,023
  Time per update: 0.129s
  Best eval return: 42.35

💾 Model saved to: models/snake_jax_10x10_20251014_120000/best_model.pkl
```

## Quick Test

```bash
# Test the evaluation function
source .venv/bin/activate
python test_eval.py

# Test training with evaluation
./test_eval_training.sh
```

## Next Steps

1. **Run a quick sweep** to test the setup:
   ```bash
   wandb sweep wandb_sweep_jax_ppo_eval.yaml
   wandb agent <sweep-id>
   ```

2. **Customize the sweep** for your needs (edit `wandb_sweep_jax_ppo_eval.yaml`)

3. **Scale up** with multiple agents and longer training budgets

## Documentation

- [docs/EVALUATION_METRICS.md](docs/EVALUATION_METRICS.md) - Detailed metric explanations
- [docs/SWEEP_GUIDE.md](docs/SWEEP_GUIDE.md) - Complete sweep guide
- [METRICS_EXPLAINED.md](METRICS_EXPLAINED.md) - Training metrics (for comparison)

---

**Ready to sweep!** 🧹✨
