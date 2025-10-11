# Snake + Transformer DQN

A compact Transformer-based DQN agent that learns to play a simple terminal Snake. Includes a larger grid, wall collisions, CLI arguments, and episode reconstruction.

Two implementations: custom PyTorch (`snake.py`) and Stable-Baselines3 (`sb3_snake.py`).

## Setup

Create/activate a virtual environment at `.venv` and install deps:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# On Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

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

## Notes

- Environment uses wall collisions (no wrap-around) and a small step penalty to encourage progress.
- Transformer treats each grid cell as a token with 3 channels: [snake, food, empty].
- SB3 version includes built-in evaluation, model saving, and TensorBoard logging.
- Training from scratch is non-trivial; start with smaller grids and more episodes, or tweak epsilon schedule.
