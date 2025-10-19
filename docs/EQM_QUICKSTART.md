# Quick Start: Snake World Model with EqM

## What is this?

A world model for Snake that uses **Equilibrium Matching** to jointly predict:

- The next game state (after taking an action)
- The action to reach that state

Unlike traditional approaches, EqM learns an energy landscape where (state, action) pairs are optimized together via gradient descent.

## Setup

### 1. **Generate a world model dataset** (includes next_state)

```bash
python generate_world_model_dataset.py \
  --num-samples 10000 \
  --output snake_world_test \
  --use-astar
```

### 2. **Train the EqM world model**

```bash
python train_snake_world.py \
  --dataset outputs/datasets/snake_world_test/ \
  --epochs 10 \
  --batch-size 256 \
  --output-dir outputs/models/snake_world_eqm_test \
  --logging-steps 10 \
  --eval-steps 100
```

## How It Works

```text
Grid → Encoder → Hidden State → EqM Gradient Field
                                        ↓
                            Optimize (next_state, action) via GD
```

1. **Encoder**: Transformer processes current grid
2. **EqM Gradient Field**: Predicts gradients for (next_state, action)
3. **Sampling**: Gradient descent finds optimal (next_state, action) pair

## Key Parameters

- `--latent-dim 64`: Dimension for next-state representation
- `--eqm-sampling-steps 10`: Number of GD optimization steps
- `--gradient-schedule linear`: How gradient magnitude decays
- `--eqm-step-size 0.1`: Learning rate for GD

## Benefits

- **Joint optimization**: State and action are coherent
- **Adaptive compute**: More steps for hard decisions
- **Energy landscape**: Captures valid state-action pairs
- **Planning-ready**: Can extend to multi-step lookahead

See `docs/EQM_WORLD_MODEL.md` for full documentation.
