# Training Metrics Explained

This document explains what all the metrics logged to WandB mean during Snake training.

## Episode Metrics

These metrics track the performance of the agent playing Snake:

### `episode/mean_return`

**What it is:** The average cumulative reward across all episodes that completed during this update.

**How to interpret:**

- Higher is better (the snake is performing better)
- With default rewards: +10 per apple, -10 for death, -0.01 per step
- Example: A return of 50 means the snake ate about 6 apples (6×10=60) minus death penalty (-10) and step penalties
- Ideal performance: This should steadily increase as training progresses
- Typical progression: Starts negative (dying quickly), gradually improves to 0-20 (eating a few apples), then to 50+ (eating many apples)

### `episode/max_return` & `episode/min_return`

**What it is:** The best and worst episode returns during this update.

**How to interpret:**

- Shows the range of performance across parallel environments
- Large gap between min/max means inconsistent performance (still learning)
- Converging min/max means more consistent behavior (better policy)

### `episode/count`

**What it is:** Number of episodes that completed during this update step.

**How to interpret:**

- More episodes = faster learning (more data)
- Fewer episodes might mean the agent is surviving longer (good!)
- With 2048 parallel environments and 128 steps per update, you typically see 100-2000 completed episodes per update
- If this drops significantly mid-training, it usually means agents are living longer

### `episode/mean_length`

**What it is:** Average number of steps per episode before the snake died.

**How to interpret:**

- Higher is better (snake survives longer)
- Max possible is the `--max-steps` parameter (default 500)
- Early training: 10-50 steps (dying quickly)
- Mid training: 100-200 steps (learning to survive)
- Late training: 300-500 steps (skilled play or hitting max_steps limit)

## Loss Metrics

These track how well the neural network is optimizing:

### `loss/total`

**What it is:** Combined PPO loss = `actor_loss + 0.5×value_loss - 0.01×entropy`

**How to interpret:**

- Should generally decrease over time (network is improving)
- Large fluctuations are normal in early training
- If it increases significantly, the learning rate might be too high
- Typical range: 0.1 to 2.0 for this task

### `loss/actor` (Policy Loss)

**What it is:** How well the policy (action selection) matches the advantages.

**How to interpret:**

- Measures if the agent is learning to take better actions
- Should decrease as the policy improves
- PPO clips this loss to prevent too-large updates
- Typical range: -0.5 to 0.5 (note: can be negative due to PPO formulation)

### `loss/value` (Critic Loss)

**What it is:** Mean squared error between predicted state values and actual returns.

**How to interpret:**

- How accurately the network predicts future rewards
- Should decrease as value estimation improves
- High value loss = network can't predict rewards well
- Typical range: 0.01 to 5.0

### `loss/entropy`

**What it is:** Average entropy of the action distribution (how random the policy is).

**How to interpret:**

- Higher entropy = more exploration (more random actions)
- Lower entropy = more exploitation (more confident/deterministic actions)
- Should naturally decrease during training as the policy becomes more confident
- Typical progression: Starts at ~1.2-1.4, decreases to 0.1-0.5
- If it drops to near 0 too early, the agent might be stuck in a local optimum
- The entropy coefficient (0.01) encourages exploration by adding entropy to the objective

## Training Progress Metrics

### `train/update`

**What it is:** Current update number (0 to NUM_UPDATES-1).

**How to interpret:**

- Just a counter for tracking progress
- Each update processes NUM_ENVS × NUM_STEPS environment steps

### `train/timesteps`

**What it is:** Total environment steps completed so far.

**How to interpret:**

- This is the main measure of training progress
- With default settings: 2048 envs × 128 steps = 262,144 steps per update
- Total training: 5,000,000 steps (default)
- More timesteps = more data = better learning (usually)

### `train/env_timestep`

**What it is:** Internal environment step counter.

**How to interpret:**

- Total steps executed across all parallel environments
- Should match `train/timesteps`
- Useful for debugging synchronization issues

## Understanding the Learning Curve

### Healthy Training Looks Like

1. **Episode returns** start negative, gradually increase
2. **Episode length** increases (snake survives longer)
3. **Episode count** stays high initially, then decreases as episodes get longer
4. **Total loss** decreases with some noise
5. **Entropy** gradually decreases from ~1.3 to ~0.2-0.5
6. **Value loss** decreases as predictions improve

### Warning Signs

- **Returns plateau early:** Learning rate too low or agent stuck in local optimum
- **Returns oscillate wildly:** Learning rate too high or numerical instability
- **Entropy drops to 0 quickly:** Agent stopped exploring, might be stuck
- **Value loss stays high:** Network architecture might be inadequate
- **Episode count drops to near 0:** Agents might be hitting max_steps without dying (could be good or bad depending on returns)

## Reward Structure Impact

With default rewards:

- `--apple-reward 10.0`: Eating an apple gives +10
- `--death-penalty -10.0`: Dying gives -10
- `--step-penalty -0.01`: Each step gives -0.01

This means:

- Breaking even (return = 0) requires eating 1 apple and surviving ~100 steps
- Return of 50 ≈ 6 apples eaten, 0 steps survived after accounting for death
- Return of 100 ≈ 11 apples eaten
- Maximum theoretical return ≈ (board_area - starting_length) × 10 - max_steps × 0.01

For a 10×10 board:

- Max apples: ~95 (board_area 100 - snake starting length ~5)
- Max return: ~95×10 - 500×0.01 = ~945 (practically impossible)
- Good performance: 100-200 (eating 10-20 apples consistently)
- Great performance: 300+ (eating 30+ apples)
