# Snake Environment Implementations

This directory contains various implementations of the Snake game environment, each designed for different reinforcement learning frameworks and use cases. These implementations were previously in `archived_implementations/` and are being organized here as a first step toward creating a dedicated repository for high-quality RL environments.

The goal is to eventually move these implementations into their own repository dedicated to building "The Universe's Best RL Environments"™, where they can be properly maintained, documented, and shared with the broader RL community.

## Environment Implementations

### 1. `snake.py` - Pure PyTorch DQN Implementation

**Description:** A standalone Snake environment with a built-in Transformer-based DQN agent for training and playing Snake from scratch.

**Framework:** PyTorch (pure, no Stable-Baselines3)

**Key Features:**

- Transformer encoder with positional encodings for grid-based observations
- DQN with experience replay buffer and epsilon-greedy exploration
- Support for configurable grid sizes, wall collisions, and multiple apples
- Includes terminal rendering and reward shaping options
- Complete training loop with CLI arguments

**Used By:**

- `tests/test_shapes.py` - Tests the Transformer model shapes
- `archived_implementations/pretrain_integration.py` - Legacy pretraining integration
- `generate_world_model_dataset.py` - World model dataset generation (imports `SnakeEnv` which appears to be a variant)

**Observation Space:** Grid state (H × W × 3) with channels for snake body, head, and food

**Action Space:** Discrete(4) - Up, Down, Left, Right

---

### 2. `sb3_snake.py` - Stable-Baselines3 DQN with Transformer

**Description:** A Gymnasium-compatible Snake environment wrapper designed for Stable-Baselines3's DQN algorithm with advanced optimization features.

**Framework:** Stable-Baselines3 + PyTorch + Gymnasium

**Key Features:**

- Custom `TransformerExtractor` feature extractor for SB3
- Loop detection and penalty system to discourage repetitive behavior
- Reward shaping with Manhattan distance to food
- Support for multiple simultaneous apples
- WandB integration for experiment tracking
- Advanced optimization: Flash Attention, BF16 precision, torch.compile support
- Optional Muon optimizer integration
- Environment variable-based configuration (.env support)

**Used By:**

- `tests/test_metrics.py` - Metrics testing
- `tests/test_optimizations.py` - Testing optimization features
- `tests/test_save.py` and `tests/test_real_results.py` - Result handling tests

**Observation Space:** Box(0, 1, shape=(H, W, 3), dtype=float32)

**Action Space:** Discrete(4)

**Notable:** This is a production-ready implementation with extensive performance optimizations and debugging features.

---

### 3. `sb3_snake_ppo.py` - Stable-Baselines3 PPO with Transformer

**Description:** Nearly identical to `sb3_snake.py` but configured for PPO (Proximal Policy Optimization) instead of DQN.

**Framework:** Stable-Baselines3 + PyTorch + Gymnasium

**Key Features:**

- All features from `sb3_snake.py` (loop detection, reward shaping, optimizations)
- PPO-specific configurations and training loop
- Supports both continuous learning rate schedules (cosine) and constant rates
- Includes sophisticated callback system with evaluation and progress tracking

**Used By:**

- Generally used as a drop-in replacement for `sb3_snake.py` when PPO is preferred over DQN
- Shares test infrastructure with `sb3_snake.py`

**Observation Space:** Box(0, 1, shape=(H, W, 3), dtype=float32)

**Action Space:** Discrete(4)

**Notable:** PPO typically provides more stable training than DQN for this environment, especially with the Transformer architecture.

---

### 4. `snake_jax/` - GPU-Accelerated JAX Implementation

**Description:** A fully vectorized, GPU-native Snake environment using JAX with massive parallelization capabilities.

**Framework:** JAX + Flax + Gymnax

**Key Features:**

- Pure functional JAX environment (fully JIT-compiled)
- Vectorized for 2048-16384 parallel environments
- 100-370x faster than PyTorch implementations
- Zero CPU↔GPU data transfer during training
- Flax-based Transformer policy network
- Gymnax wrapper for compatibility
- PPO training implementation adapted from PureJaxRL

**Used By:**

- `archived_implementations/train_snake_jax.py` - Training script
- `archived_implementations/train_snake_purejaxrl.py` - PureJaxRL-based training
- `archived_implementations/train_snake_purejaxrl_impl.py` - Custom PPO implementation
- `archived_implementations/play_snake_jax.py` - Play trained agents
- `archived_implementations/validate_dataset.py` - Dataset validation
- `tests/test_eval.py` - Evaluation testing
- `tests/test_rsm_generation.py` - RSM (Recurrent State Model) testing
- `play_pretrained.py` - Main script for playing pretrained models

**Directory Structure:**

```text
snake_jax/
├── __init__.py         # Package initialization
├── config.py          # EnvConfig dataclass
├── env.py             # Pure JAX SnakeEnv
├── gymnax_wrapper.py  # Gymnax compatibility wrapper
├── network.py         # Flax Transformer policy
├── train_ppo.py       # PPO training skeleton
└── README.md          # Comprehensive documentation
```

**Observation Space:** Grid state (vectorized across environments)

**Action Space:** Discrete(4)

**Performance:** ~50,000-185,000 FPS with 256-16384 parallel environments, ~85-95% GPU utilization

**Notable:** This is the highest-performance implementation and represents the state-of-the-art for massively parallel RL training.

---

## Comparison Summary

| Implementation | Algorithm | Speed | Parallelization | Best For |
|----------------|-----------|-------|-----------------|----------|
| `snake.py` | DQN | Baseline | Single env | Learning/prototyping |
| `sb3_snake.py` | DQN | Medium | Low (1-64 envs) | Production DQN training |
| `sb3_snake_ppo.py` | PPO | Medium | Low (1-64 envs) | Production PPO training |
| `snake_jax/` | PPO | Very High | Extreme (2048-16384 envs) | Large-scale experiments |

## Future Plans

These environments are being prepared for extraction into a standalone "Universe's Best RL Environments" repository, where they will be:

- Properly packaged and distributed via PyPI
- Enhanced with additional features and variants
- Thoroughly documented with tutorials
- Continuously benchmarked and optimized
- Extended with new game environments beyond Snake
