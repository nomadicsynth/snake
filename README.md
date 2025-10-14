# Snake + Transformer RL

A compact Transformer-based RL agent that learns to play Snake. Multiple implementations from legacy to cutting-edge:

- 🐢 **PyTorch DQN** (`snake.py`) - Original implementation
- 🏃 **Stable-Baselines3 PPO** (`sb3_snake_ppo.py`) - Standard RL baseline
- 🚀 **JAX GPU-Native** (`snake_jax/`) - **100-370x faster!** ⚡

## 🔥 NEW: GPU-Native JAX Implementation

**It's 2025.** We built a fully GPU-accelerated version:

- ✅ **185,000+ FPS** (vs 100-500 FPS with SB3)
- ✅ **Zero CPU↔GPU transfers**
- ✅ **2048+ parallel environments**
- ✅ **90%+ GPU utilization**

**See [`JAX_IMPLEMENTATION_SUMMARY.md`](JAX_IMPLEMENTATION_SUMMARY.md) for details!**

Quick test:
```bash
source .venv/bin/activate
python test_snake_jax.py
```

## Setup

Create/activate a virtual environment at `.venv` and install deps:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# On Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### Optional: FlashAttention (for faster training)

FlashAttention can provide 10-40% speedup for the Transformer model, especially with larger batch sizes:

```bash
# Install flash-attn (requires CUDA and takes several minutes to compile)
pip install flash-attn --no-build-isolation

# Or use pre-built wheels if available for your Python/CUDA version:
# pip install flash-attn
```

**Note**: FlashAttention requires:

- CUDA-capable GPU (NVIDIA)
- CUDA toolkit installed
- May take 5-10 minutes to compile on first install

PyTorch 2.0+ will automatically use FlashAttention when available. To benchmark the performance impact, see "Benchmarking" section below.

If you're in VS Code, select the interpreter at `.venv/bin/python`.

## Train (Custom PyTorch)

```bash
# Quick smoke test
python snake.py train --episodes 5 --width 10 --height 10 --batch-size 8 --target-update 1 --max-steps 50

# Default training run
python snake.py train
```

## Train (SB3)

```bash
# Quick smoke test
python sb3_snake.py train --episodes 5 --width 10 --height 10 --batch-size 8 --target-update 1 --max-steps 50

# Default training run
python sb3_snake.py train
```

Key flags (same for both):

- `--width`, `--height`: grid size (default 20x20)
- `--episodes`: number of training episodes
- `--batch-size` (default 64)
- `--gamma` (default 0.99)
- `--eps-start`, `--eps-end`, `--eps-decay`
- `--target-update`: target net update interval
- `--d-model`, `--layers`, `--heads`, `--dropout`: Transformer size (default tiny)
- `--lr`: learning rate (default 3e-4)
- `--replay-size` (default 50k)
- `--max-steps`: per-episode cap
- `--model-path`: where to save model (`.pth` for custom, `.zip` for SB3)
- `--seed`

Artifacts:

- Custom: `snake_transformer.pth`, `episode_logs.pkl`, `training_stats.png`
- SB3: `sb3_snake_transformer.zip`, TensorBoard logs in `./tb_snake`, eval logs in `./eval_logs`, best model in `./models`

## Weights & Biases Logging (SB3 only)

Enable W&B logging to track experiments:

```bash
# First login to wandb (one-time setup)
wandb login

# Train with wandb logging
python sb3_snake.py train --wandb --wandb-project "my-snake-project" --wandb-run-name "experiment-1"

# With custom tags for organization
python sb3_snake.py train --wandb --wandb-tags baseline transformer cosine-lr
```

W&B will log:

- All training metrics (episode rewards, scores, lengths)
- Loop detection metrics (if loop penalties are enabled)
- Evaluation metrics
- Hyperparameters
- TensorBoard logs (synced automatically)

The `--wandb` flag enables logging. Additional options:

- `--wandb-project`: W&B project name (default: "snake-rl")
- `--wandb-run-name`: Custom run name (auto-generated if not specified)
- `--wandb-tags`: Space-separated tags for organizing runs

## Play

```bash
# Custom
python snake.py play --model-path snake_transformer.pth --width 20 --height 20 --render-delay 0.1

# SB3
python sb3_snake.py play --model-path sb3_snake_transformer.zip --width 20 --height 20 --render-delay 0.1
```

## Reconstruct a Logged Episode

```bash
python snake.py reconstruct --logs episode_logs.pkl --index 0 --render-delay 0.1
# Same for sb3_snake.py
```

## Benchmarking (Batch Size & Performance Optimization)

Test different batch sizes and optimization strategies to find the best throughput for your hardware:

```bash
python sb3_snake_batch_size_benchmark.py
```

This will automatically:

1. Test batch sizes: 128, 256, 512, 1024, 2048, 4096
2. Compare performance with/without `torch.compile()`
3. Compare different attention backends (including FlashAttention if installed)
4. Show GPU memory usage for each configuration

The benchmark runs 4 test configurations:

- **Baseline**: No compile, auto SDPA
- **FlashAttention**: Explicit FlashAttention backend
- **torch.compile()**: Compiled model with auto SDPA  
- **Both**: Compiled + FlashAttention (usually fastest)

Example output shows steps/sec and speedup for each combination. For RTX 4090 with 24GB VRAM:

- Recommended batch size: **1024-2048**
- Expected speedup with torch.compile(): **15-30%**
- Expected speedup with FlashAttention: **5-15%**
- Combined speedup: **20-40%**

## Notes

- Environment uses wall collisions (no wrap-around) and a small step penalty to discourage inefficient/aimless behavior.
- Transformer treats each grid cell as a token with 3 channels: [snake, food, empty].
- SB3 version includes built-in evaluation, model saving, and TensorBoard logging.
- Training from scratch is non-trivial; start with smaller grids and more episodes, or tweak epsilon schedule.
