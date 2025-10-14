# DQN to PPO Conversion Summary

## Overview

Successfully converted `sb3_snake_ppo.py` from using DQN to PPO (Proximal Policy Optimization).

## Key Changes

### 1. Algorithm Switch

- **Changed**: `from stable_baselines3 import DQN` → `from stable_baselines3 import PPO`
- **Changed**: All `DQN()` instantiations to `PPO()`
- **Changed**: All `DQN.load()` calls to `PPO.load()`

### 2. Loss Function - Now Includes KL/Entropy Terms

PPO's loss function includes:

- **Policy Loss**: Clipped surrogate objective with KL penalty (implicit via clipping)
- **Entropy Bonus**: Controlled by `ent_coef` (default: 0.01) - encourages exploration
- **Value Function Loss**: Controlled by `vf_coef` (default: 0.5)

**This addresses your collapse issue!** The entropy term prevents the policy from becoming too deterministic.

### 3. Removed DQN-Specific Parameters

Removed parameters that don't apply to PPO:

- `eps_start`, `eps_end` - PPO doesn't use epsilon-greedy exploration
- `exploration_fraction`, `learning_starts` - PPO uses stochastic policy instead
- `target_update` - PPO doesn't use target networks
- `replay_size` - PPO is on-policy, no replay buffer

### 4. Added PPO-Specific Parameters

New hyperparameters for PPO:

- `n_epochs` (default: 10) - Number of optimization epochs per update
- `ent_coef` (default: 0.01) - **Entropy coefficient** for exploration
- `vf_coef` (default: 0.5) - Value function loss coefficient
- `clip_range` (default: 0.2) - PPO clipping parameter
- `gae_lambda` (default: 0.95) - Generalized Advantage Estimation parameter

### 5. Network Architecture

- Changed from single Q-network to separate policy and value networks
- `net_arch` now uses `dict(pi=[128], vf=[128])` for separate actor-critic architecture

### 6. Compilation Optimizations

Updated torch.compile() to work with PPO's architecture:

- Compiles `mlp_extractor`, `action_net`, and `value_net` instead of Q-networks

### 7. Updated Default Model Paths

- Changed default model path from `sb3_snake_transformer.zip` to `sb3_snake_ppo_transformer.zip`

## Why This Helps With Collapse

### DQN Issues

- Pure value-based learning with no entropy regularization
- Can converge to deterministic, suboptimal policies
- Prone to overestimation bias and catastrophic forgetting

### PPO Advantages

1. **Entropy Bonus** (`ent_coef=0.01`): Keeps policy stochastic, prevents premature convergence
2. **Clipped Objective**: Prevents large policy updates that cause instability
3. **On-Policy Learning**: More stable than off-policy DQN
4. **Dual Optimization**: Learns both policy and value function simultaneously

## Usage

Train with default PPO settings:

```bash
python sb3_snake_ppo.py train
```

Adjust entropy coefficient for more exploration:

```bash
python sb3_snake_ppo.py train --ent-coef 0.05
```

Adjust clipping for more conservative updates:

```bash
python sb3_snake_ppo.py train --clip-range 0.1
```

## Recommended Starting Hyperparameters

For Snake game, consider:

- `--ent-coef 0.01` to 0.05 - Higher values encourage more exploration
- `--clip-range 0.2` - Standard PPO clipping
- `--n-epochs 10` - Standard number of optimization epochs
- `--gae-lambda 0.95` - Standard GAE parameter
- `--vf-coef 0.5` - Balances policy and value learning

## Monitoring

Watch these metrics in TensorBoard/W&B:

- `train/entropy_loss` - Should stay positive; if it drops to zero, increase `ent_coef`
- `train/policy_gradient_loss` - Main policy objective
- `train/value_loss` - Value function learning
- `train/approx_kl` - Monitors KL divergence between old and new policies
- `train/clip_fraction` - Percentage of updates that were clipped

If the model still collapses, try:

1. Increase `--ent-coef` to 0.05 or 0.1
2. Decrease `--clip-range` to 0.1 for more conservative updates
3. Increase `--dropout` in the transformer
4. Add reward shaping with `--shaping-coef`
