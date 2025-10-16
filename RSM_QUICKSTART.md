# RSM Quick Start Guide

## What is RSM?

**Reasoning Snake Model (RSM)** is a variant of the Snake AI that generates explicit chain-of-thought reasoning before making decisions, similar to OpenAI's o1 reasoning models.

Instead of:
```
Grid → Action
```

RSM does:
```
Grid → "THINK: U:wall D:safe(d=5) L:death R:safe(d=3) | BEST:R" → Action:RIGHT
```

## Key Features

✅ **Autoregressive reasoning generation** - Model generates its own reasoning text token-by-token  
✅ **Causal attention** - Proper triangular masking like GPT  
✅ **Teacher forcing training** - Learn from expert reasoning during training  
✅ **Self-generating inference** - Model creates reasoning at test time  
✅ **Interpretable** - See exactly what the model is "thinking"  

## Usage

### 1. Generate RSM Dataset

```bash
python generate_dataset.py \
  --num-samples 50000 \
  --reasoning \
  --reasoning-depth 1 \
  --output snake_rsm_dataset.pkl
```

This creates a dataset where each sample has:
- `state`: Grid observation
- `action`: Expert action
- `reasoning`: Expert reasoning text (e.g., "THINK: U:wall D:safe...")
- `reasoning_tokens`: Tokenized reasoning (ASCII values)

### 2. Train RSM Model

```bash
python pretrain_jax.py \
  --dataset snake_rsm_dataset.pkl \
  --epochs 20 \
  --batch-size 256 \
  --d-model 128 \
  --num-layers 4 \
  --lr 3e-4
```

The script auto-detects RSM datasets and:
- Enables causal masking for reasoning tokens
- Uses extended vocabulary (4 actions + 128 ASCII tokens)
- Trains with teacher forcing

### 3. Play with RSM Model

```bash
python play_pretrained.py \
  --model pretrain_models/best_model.pkl \
  --episodes 10 \
  --show-reasoning
```

At inference, the model:
1. Sees the grid
2. Autoregressively generates reasoning tokens
3. Stops when it predicts an action (0-3)
4. Executes the action

Example output:
```
🧠 Generated reasoning: THINK: U:safe(d=3,f=142) R:apple(f=138) D:safe(d=7) L:death | BEST:R | ACT:1
   Action: RIGHT
```

## How It Works

### Training (Teacher Forcing)

```python
# Dataset contains pre-generated expert reasoning
for state, reasoning, action in dataset:
    # Feed grid + reasoning → predict action
    logits = model(state, reasoning_tokens=reasoning)
    loss = cross_entropy(logits[:4], action)  # Only train on action prediction
```

### Inference (Autoregressive)

```python
reasoning_tokens = []
for t in range(max_tokens):
    logits = model(grid, reasoning_tokens=reasoning_tokens)
    next_token = argmax(logits)
    
    if next_token < 4:
        return next_token  # It's an action - stop!
    else:
        reasoning_tokens.append(next_token - 4)  # It's reasoning - continue
```

## Architecture Details

### Vocabulary

- **Tokens 0-3**: Actions (UP, RIGHT, DOWN, LEFT)
- **Tokens 4-131**: ASCII characters (0-127) offset by 4

### Attention Pattern

```
Grid tokens:      [████████████]  ← Bidirectional (see all grid)
                         ↓
Reasoning tokens: [████▲▲▲▲▲▲▲▲]  ← Causal (see grid + past reasoning)
                         ↓
Action prediction: logits[:4]
```

### Token Sequence

```
[grid₁, grid₂, ..., grid₄₀₀, reason₁, reason₂, ..., reasonₙ]
 ←─── H×W=20×20=400 ───→  ←──── up to 128 tokens ────→
```

## Reasoning DSL Format

### Compact (depth=1)
```
THINK: U:safe(d=3,f=142) R:safe(d=1,f=138) D:safe(d=7,f=140) L:death | BEST:R | ACT:1
```

- **U/R/D/L**: Directions
- **safe/death/wall/apple**: Outcome of move
- **d=N**: Distance to food after move
- **f=N**: Freedom (reachable cells)

### Multi-step (depth=3)
```
THINK: U:[1:safe->2:safe->3:apple] R:[1:death] L:[1:safe->2:wall] | BEST:U | ACT:0
```

## Debugging

### Test Autoregressive Generation

```bash
python test_rsm_generation.py
```

This verifies:
- ✅ Forward pass with/without reasoning
- ✅ Autoregressive token generation
- ✅ Causal mask correctness
- ✅ Output shape validation

### Common Issues

**Model generates gibberish**
- This is normal for untrained models
- Train on expert reasoning to learn the DSL format

**Shape mismatch errors**
- Check `max_reasoning_length` parameter
- Default is 128 tokens

**Action never predicted**
- Model might not have learned when to stop reasoning
- Increase training epochs or check dataset quality
- Add action prediction loss weighting

## Performance Tips

1. **Start simple**: Use depth=1 reasoning during initial training
2. **Increase gradually**: Move to depth=2 or 3 after model learns basics
3. **Balance dataset**: Mix expert and suboptimal actions for robustness
4. **Monitor reasoning quality**: Use `--show-reasoning` to inspect outputs
5. **Consider curriculum**: Train first on short reasoning, then longer

## Comparison: Standard vs RSM

| Metric | Standard | RSM |
|--------|----------|-----|
| Training speed | Fast | Medium (-20-30%) |
| Inference speed | Fast | Medium (-30-50%) |
| Interpretability | ❌ Black box | ✅ Shows reasoning |
| Sample efficiency | Good | Better (with good reasoning) |
| Model size | Smaller | Larger (+10-20% params) |

## Advanced: Custom Reasoning Format

You can define your own reasoning DSL by modifying `reasoning_dsl.py`:

```python
def generate_reasoning_text(...):
    # Your custom format here
    return "MyFormat: ..."
```

The model will learn whatever format you provide in the training data!

## Next Steps

1. Generate a small RSM dataset (10k samples)
2. Train for 10-20 epochs
3. Evaluate with `play_pretrained.py --show-reasoning`
4. Iterate on reasoning format if needed
5. Scale to larger dataset (100k-1M samples)

---

Happy reasoning! 🧠🐍
