# Snake World Model with Equilibrium Matching (EqM)

This implements a world model for the Snake game using **Equilibrium Matching** (EqM), a novel approach from the paper "Equilibrium Matching: Generative Modeling with Implicit Energy-Based Models" by Runqian Wang and Yilun Du.

## Key Idea

Unlike traditional approaches that separately predict:
1. Next state given (current_state, action)
2. Action given current_state

EqM learns a **time-invariant gradient field** that jointly optimizes both the next state and action together. This creates an energy landscape where valid (next_state, action) pairs are local minima.

## Architecture

```
Current State (grid_t)
    ↓
Encoder (Transformer + optional CNN)
    ↓
Hidden Representation (d_model)
    ↓
EqM Gradient Field Network
    ↓
Gradients for (latent_next_state, action_logits)
    ↓
Gradient Descent Optimization (NAG)
    ↓
Optimized (next_state, action)
```

### Components

1. **Encoder**: Processes the current grid state into a hidden representation
   - Same transformer architecture as the standard policy
   - Optional CNN for spatial feature extraction
   - Positional encoding for grid structure

2. **Latent Projection**: Maps grid states to a lower-dimensional latent space
   - Reduces dimensionality for efficient optimization
   - Learned representation captures game semantics

3. **EqM Gradient Field**: Neural network that predicts gradients
   - Conditioned on current state's hidden representation
   - Outputs gradients for both next-state (latent) and action (logits)
   - Shared MLP with separate heads for state and action gradients

4. **Optimizer**: Nesterov Accelerated Gradient (NAG) descent
   - Iteratively optimizes (next_state, action) jointly
   - Optional adaptive compute (stops when gradient norm < threshold)

## Training

### EqM Objective

The model learns to match gradients at interpolated states:

```python
# Sample interpolation factor
gamma ~ Uniform(0, 1)

# Interpolate between target and noise
x_gamma = gamma * target + (1 - gamma) * noise

# Predict gradient
grad_pred = f(hidden_t, x_gamma)

# Target gradient descends from noise to target
grad_target = (noise - target) * c(gamma)

# Loss
loss = MSE(grad_pred, grad_target)
```

### Gradient Magnitude Schedule `c(gamma)`

Controls gradient magnitude, vanishing at real data (gamma=1):

- **Linear**: `c(γ) = λ(1 - γ)`
- **Truncated**: Constant until γ=a, then linear decay
- **Piecewise**: Two-segment linear function

## Usage

### 1. Generate Dataset

First, generate a dataset with next_state:

```bash
python generate_world_model_dataset.py \
  --num-samples 10000 \
  --output snake_world_dataset \
  --use-astar \
  --augment
```

This creates a dataset with `(state, action, next_state)` tuples.

### 2. Train the Model

```bash
python train_snake_world.py \
  --dataset outputs/datasets/snake_world_dataset/ \
  --epochs 20 \
  --batch-size 256 \
  --d-model 128 \
  --num-layers 3 \
  --latent-dim 64 \
  --eqm-sampling-steps 10 \
  --gradient-schedule linear \
  --wandb \
  --output-dir outputs/models/snake_world_eqm
```

### 3. Key Hyperparameters

**Encoder:**
- `--d-model`: Transformer dimension (default: 128)
- `--num-layers`: Transformer depth (default: 3)
- `--num-heads`: Attention heads (default: 4)
- `--use-cnn`: Enable CNN encoder
- `--cnn-mode`: "replace" or "append"

**EqM Architecture:**
- `--latent-dim`: Dimension of latent next-state (default: 64)
- `--eqm-hidden-dim`: Hidden dim of gradient network (default: 128)
- `--eqm-num-layers`: Depth of gradient network (default: 3)

**EqM Training:**
- `--gradient-schedule`: "linear", "truncated", or "piecewise"
- `--gradient-multiplier`: Overall gradient scale λ (default: 1.0)

**EqM Sampling:**
- `--eqm-sampling-steps`: Number of GD steps (default: 10)
- `--eqm-step-size`: Learning rate for GD (default: 0.1)
- `--eqm-nag-momentum`: Nesterov momentum (default: 0.9)
- `--eqm-adaptive-threshold`: Stop when grad_norm < threshold (optional)

## Advantages

1. **Joint Optimization**: State and action are optimized together, ensuring coherence
2. **Adaptive Compute**: Can allocate more steps to critical decisions
3. **Energy Landscape**: Learns implicit energy where valid states are minima
4. **Planning**: Energy landscape can be used for multi-step lookahead
5. **Uncertainty**: Gradient norm indicates confidence in predictions

## Inference

```python
from model.model_eqm import SnakeWorldEqM, SnakeWorldEqMConfig

# Load model
config = SnakeWorldEqMConfig.from_pretrained(model_path)
model = SnakeWorldEqM.from_pretrained(model_path)

# Sample action
result = model.sample(current_obs)
action = result['action']  # Discrete action
latent_next = result['latent_next']  # Predicted next state (latent)

# Optional: decode latent to grid
next_state_hidden = model.from_latent(latent_next)
```

## Extending with Language Model

The EqM world model can be integrated with a language model for reasoning:

```python
# Get latent next state and action logits from EqM
eqm_output = eqm_model.sample(obs)
latent_next = eqm_output['latent_next']
action_logits = eqm_output['action_logits']

# LM can:
# 1. Reason about the predicted future state
# 2. Refine the action selection
# 3. Plan multi-step sequences

final_action = language_model(latent_next, action_logits)
```

## Files

- `model/model_eqm.py`: EqM model implementation
- `train_snake_world.py`: Training script
- `generate_world_model_dataset.py`: Dataset generation with next_state
- `docs/EQM_WORLD_MODEL.md`: This file

## References

**Equilibrium Matching: Generative Modeling with Implicit Energy-Based Models**  
Runqian Wang, Yilun Du  
*arXiv preprint, 2024*

Key concepts:
- Time-invariant gradient field (vs. flow matching's time-varying velocity)
- Energy landscape with data manifold as stationary points
- Gradient descent sampling with adaptive compute
- Compatible with various optimization techniques (NAG, momentum, etc.)
