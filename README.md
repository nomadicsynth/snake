# Snake Transformer

A Transformer-based model for playing Snake, trained via supervised learning on expert demonstrations.

## What Works

### Dataset Generation

Generate training datasets from expert demonstrations:

```bash
python generate_dataset_hf.py \
    --num-samples 50000 \
    --width 20 \
    --height 20 \
    --use-astar \
    --output outputs/datasets/snake_dataset_hf \
    --seed 42
```

**Tested and working:** Plain, unaugmented, fixed-size datasets. These produce models that reach 90%+ accuracy.

### Training

Train a model using HuggingFace Trainer:

```bash
python train_hf.py \
    --dataset outputs/datasets/snake_dataset_hf \
    --d-model 128 \
    --num-layers 2 \
    --num-heads 4 \
    --dropout 0.1 \
    --epochs 20 \
    --batch-size 256 \
    --optimizer muon \
    --muon-lr 0.02 \
    --muon-momentum 0.95 \
    --output-dir outputs/snake_hf_output \
    --wandb \
    --seed 42
```

**Tested results:** Test-model trained on fixed 32x32 grids with max snake length 100 achieves:
- 97.255% accuracy on the evaluation set
- Performance Metrics:
  - Rewards: Mean 617.21, Std 158.85, Min 68.62, Max 967.87, Median 622.83
  - Apples Eaten: Mean 64.40, Std 16.40, Min 8, Max 101, Median 65.00
  - Steps Survived: Mean 1733.90, Std 567.93, Min 147, Max 3591, Median 1737.50
- **Observed limitation:** Poor long-range planning (will turn into dead-ends)

### Gameplay

Play with a trained model:

```bash
python play_pretrained.py \
    --model-path outputs/snake_hf_output/run_name/final_model \
    --env-width 32 --env-height 32 \
    --episodes 10 --max-steps 3000 \
    --delay 0.1 \
```

**Note:** Video generation is currently broken and needs fixing.

## What Doesn't Work

- **Augmented datasets:** Currently break the model. May be a regression or require tuning. Not actively pursuing since the model isn't meant to be transform-invariant, and the dataset will have a greater variety of states.
- **Variable grid size datasets:** Code exists but hasn't been validated. Training runs have resulted in model collapse. This is a planned feature for future work.

## What's Untested

- **Reasoning datasets:** Training on datasets with reasoning tokens is untested. The current implementation may not make sense. Future work may involve:
  - Transformer decoder for autoregressive reasoning generation
  - Cross-attention to embedded game state
  - Unified vocabulary including action tokens (vision-language model style)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

## Project Structure

- `generate_dataset_hf.py` - Dataset generation
- `train_hf.py` - Training script
- `play_pretrained.py` - Gameplay visualization
- `model/model_pytorch.py` - Transformer model implementation
- `environments/snake.py` - Snake game environment
- `archived_implementations/` - Abandoned experimental code (various states of completion)

## Notes

- Environment uses wall collisions (no wrap-around) and step penalties
- The game state is treated as an image, with the snake assigned to the green channel, the food assigned to the red channel, and the empty cells assigned to the black channel. In variable grid size environments, the image is padded with `-1.0` to match the input shape of the transformer, with optional attention masking to ignore the padded cells.
- Training is supervised learning on expert demonstrations, akin to next-token prediction pretraining in language models. Future work will involve RL-style fine-tuning, perhaps with GRPO (Group-relative policy optimization). `trl.GRPOTrainer` is a good candidate for this. It should be possible to implement the game environment as the reward functions.
