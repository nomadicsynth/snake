# Weights & Biases Integration

This document describes the W&B integration added to the Snake RL project.

## Features Added

### 1. WandbCallback Class

A custom callback (`WandbCallback`) that logs metrics to W&B during training:

- Episode rewards, lengths, and scores
- Training timesteps
- All metrics from SB3's logger (synced from TensorBoard)
- Loop detection metrics (when enabled)

### 2. Configuration Parameters

The `train_sb3()` function now accepts:

- `use_wandb` (bool): Enable/disable W&B logging
- `wandb_project` (str): W&B project name (default: "snake-rl")
- `wandb_run_name` (str): Custom run name (optional)
- `wandb_tags` (list): Tags for organizing runs (optional)

### 3. Command-Line Arguments

New CLI flags for the training mode:

- `--wandb`: Enable W&B logging
- `--wandb-project PROJECT`: Set project name
- `--wandb-run-name NAME`: Set run name
- `--wandb-tags TAG1 TAG2 ...`: Add tags

### 4. TensorBoard Sync

W&B is configured with `sync_tensorboard=True`, which automatically syncs all TensorBoard logs to W&B for a unified view.

## Usage Examples

### Basic Usage

```bash
# First time setup - login to W&B
wandb login

# Train with W&B logging enabled
python sb3_snake.py train --wandb
```

### Custom Project and Run Name

```bash
python sb3_snake.py train \
  --wandb \
  --wandb-project "snake-experiments" \
  --wandb-run-name "baseline-20x20"
```

### With Tags for Organization

```bash
python sb3_snake.py train \
  --wandb \
  --wandb-tags transformer dqn baseline \
  --width 20 --height 20
```

### Full Training Command with W&B

```bash
python sb3_snake.py train \
  --wandb \
  --wandb-project "snake-rl-experiments" \
  --wandb-run-name "cosine-lr-experiment" \
  --wandb-tags cosine-schedule loop-penalty \
  --batch-size 64 \
  --lr 3e-2 \
  --lr-schedule cosine \
  --lr-end 1e-5 \
  --width 20 \
  --height 20 \
  --exploration-fraction 0.6 \
  --step-penalty -0.00 \
  --shaping-coef 0.1 \
  --apple-reward 20 \
  --max-score 10 \
  --max-steps 500 \
  --eval-episodes 50 \
  --loop-penalty-coef 0.02 \
  --loop-end-bonus 0.1 \
  --loop-min-period 4 \
  --loop-max-period 16
```

## What Gets Logged

W&B captures comprehensive metrics during training:

### Episode Metrics

- `episode/reward`: Total episode reward
- `episode/length`: Episode length in steps
- `episode/time`: Wall-clock time for episode
- `episode/score`: Game score (apples eaten)
- `episode/count`: Episode counter

### Training Metrics (from SB3/TensorBoard)

- `rollout/ep_len_mean`: Mean episode length
- `rollout/ep_rew_mean`: Mean episode reward
- `rollout/score`: Individual episode scores
- `rollout/score_mean`: Running mean of scores
- `train/learning_rate`: Current learning rate
- `train/loss`: DQN loss
- `rollout/exploration_rate`: Current epsilon value

### Loop Detection Metrics (if enabled)

- `rollout/loop_is_loop`: Binary indicator of loop state
- `rollout/loop_repeats`: Number of loop repetitions
- `rollout/loop_period`: Loop cycle length
- `rollout/loop_penalty`: Applied loop penalty
- `rollout/loop_end_bonus`: Bonus for breaking loop
- `rollout/loops_started_ep`: Loops started per episode
- `rollout/loops_ended_ep`: Loops ended per episode

### Evaluation Metrics

- `eval/mean_reward`: Mean evaluation reward
- `eval/mean_ep_length`: Mean evaluation episode length
- `eval/score`: Evaluation scores
- `eval/loop_*`: Loop metrics during evaluation

### Hyperparameters

All training hyperparameters are logged in the W&B config:

- Grid dimensions, learning rates, exploration params
- Reward shaping coefficients
- Loop detection parameters
- Network architecture (d_model, layers, heads, dropout)

## Implementation Details

### Code Changes

1. **Import**: Added `import wandb` at the top of `sb3_snake.py`

2. **WandbCallback**: New callback class that:
   - Inherits from `stable_baselines3.common.callbacks.BaseCallback`
   - Logs metrics on each step via `wandb.log()`
   - Tracks episode count and episode-level statistics

3. **train_sb3() modifications**:
   - Added wandb initialization with `wandb.init()`
   - Configured hyperparameters in `wandb_config` dict
   - Added `WandbCallback` to the callback list when enabled
   - Added `wandb.finish()` in the finally block for cleanup

4. **CLI Arguments**: Extended argparse with wandb-related flags

5. **requirements.txt**: Added `wandb` dependency

### Integration with Existing Logging

The implementation works alongside existing logging:

- TensorBoard logs continue to work normally
- W&B syncs TensorBoard logs automatically
- Custom callbacks (ScoreLogger, LoopLogger) work unchanged
- No breaking changes to existing functionality

### Error Handling

The implementation includes proper cleanup:

- `wandb.finish()` is called in the `finally` block
- Training can be interrupted (Ctrl+C) without leaving orphan W&B processes
- W&B is only initialized when `use_wandb=True`

## Benefits

1. **Cloud Storage**: Metrics are stored in the cloud, accessible from anywhere
2. **Visualization**: Interactive charts and custom dashboards
3. **Experiment Tracking**: Compare multiple runs easily
4. **Collaboration**: Share results with team members
5. **Hyperparameter Tracking**: Automatic logging of all config parameters
6. **Model Versioning**: Optional artifact tracking for saved models
7. **Notes & Tags**: Organize experiments with tags and descriptions

## Offline Mode

To use W&B in offline mode (no internet required):

```bash
export WANDB_MODE=offline
python sb3_snake.py train --wandb
```

Later sync offline runs:

```bash
wandb sync wandb/offline-run-*
```

## Disabling W&B

Simply omit the `--wandb` flag to train without W&B logging:

```bash
python sb3_snake.py train  # No W&B, only TensorBoard
```
