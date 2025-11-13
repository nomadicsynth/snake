# CNN Integration Guide

## Overview

The TransformerPolicy network now supports optional CNN (Convolutional Neural Network) preprocessing of the grid-based observations before passing them to the Transformer encoder.

## Why Use CNNs?

Your Snake environment provides observations as spatial grids (H, W, 3) where:

- Channel 0: Snake body positions
- Channel 1: Food position  
- Channel 2: Empty space

CNNs excel at extracting local spatial features from grid-based data, making them ideal for:

1. **Feature Extraction**: CNNs can identify patterns like snake body shapes, proximity to walls, food locations, etc.
2. **Dimensionality Reduction**: CNNs can compress spatial information before it reaches the Transformer
3. **Improved Performance**: Combining CNNs with Transformers has shown benefits in vision and RL tasks

## CNN Modes

The implementation supports two modes for integrating CNN features:

### 1. Replace Mode (`cnn_mode='replace'`)

The CNN processes the raw observation and its output **replaces** the original input to the Transformer.

**Flow**: `Raw Observation → CNN → Transformer → Policy/Value`

**When to use**:

- When you want the CNN to be the primary feature extractor
- When you believe the CNN features are more informative than the raw grid
- When you want to reduce the input dimensionality to the Transformer

### 2. Append Mode (`cnn_mode='append'`)

The CNN features are **concatenated** with the raw observation, giving each token both raw and processed information.

**Flow**: `Raw Observation → [Raw Obs, CNN Features] → Transformer → Policy/Value`

**When to use**:

- When you want the Transformer to have access to both raw and processed features
- When you want to preserve all information while adding CNN features
- When you believe combining raw and processed features will improve learning

## Usage

### Command Line Arguments

All training scripts (`train_snake_purejaxrl.py`, `train_snake_purejaxrl_progressive.py`) now support:

```bash
# Enable CNN preprocessing
--use-cnn

# Specify CNN layer features (default: [32, 64])
--cnn-features 32 64 128

# Set CNN mode (default: 'replace')
--cnn-mode replace  # or 'append'
```

### Example Commands

**Basic usage with default CNN settings:**

```bash
python train_snake_purejaxrl.py \
    --use-cnn \
    --total-timesteps 1000000
```

**Custom CNN architecture with append mode:**

```bash
python train_snake_purejaxrl.py \
    --use-cnn \
    --cnn-features 64 128 256 \
    --cnn-mode append \
    --total-timesteps 1000000
```

**Disable CNN (default behavior):**

```bash
python train_snake_purejaxrl.py \
    --total-timesteps 1000000
```

### Python API

```python
from snake_jax.network import TransformerPolicy

# Without CNN (default)
network = TransformerPolicy(
    d_model=64,
    num_layers=2,
    num_heads=4,
    num_actions=4,
    use_cnn=False
)

# With CNN in replace mode
network = TransformerPolicy(
    d_model=64,
    num_layers=2,
    num_heads=4,
    num_actions=4,
    use_cnn=True,
    cnn_features=(32, 64, 128),
    cnn_mode='replace'
)

# With CNN in append mode
network = TransformerPolicy(
    d_model=64,
    num_layers=2,
    num_heads=4,
    num_actions=4,
    use_cnn=True,
    cnn_features=(32, 64),
    cnn_mode='append'
)
```

## Architecture Details

### CNNEncoder Module

The `CNNEncoder` module consists of:

- Multiple convolutional layers with configurable feature dimensions
- 3x3 kernels with SAME padding (maintains spatial dimensions)
- GELU activation functions
- Output shape: (batch, height, width, features[-1])

### Integration with Transformer

1. **Input Processing**:
   - Raw observation: (batch, H, W, 3)

2. **CNN Processing** (if enabled):
   - CNN output: (batch, H, W, cnn_features[-1])
   - Replace mode: Use CNN output directly
   - Append mode: Concatenate [raw, CNN output] → (batch, H, W, 3 + cnn_features[-1])

3. **Tokenization**:
   - Flatten to tokens: (batch, H*W, channels)
   - Project to d_model: (batch, H*W, d_model)

4. **Transformer Processing**:
   - Add positional encoding
   - Apply Transformer blocks
   - Pool and generate policy/value outputs

## Performance Considerations

- **Memory**: CNN append mode uses more memory (3 + cnn_features[-1] channels vs just cnn_features[-1])
- **Computation**: CNNs add computational overhead but may improve sample efficiency
- **JIT Compilation**: First call will be slower due to JAX JIT compilation

## Experimentation Tips

1. **Start Simple**: Begin with `--use-cnn --cnn-features 32 64` to see if CNNs help
2. **Compare Modes**: Try both replace and append modes to see which works better
3. **Tune Architecture**: Experiment with different feature sizes (e.g., [64, 128], [32, 64, 128])
4. **Monitor Training**: Use WandB to compare CNN vs non-CNN performance
5. **Hyperparameter Sweep**: Use wandb sweeps to find optimal CNN configurations

## Integration Status

✅ **Implemented in**:

- `snake_jax/network.py` - Core network architecture
- `train_snake_purejaxrl.py` - Main training script
- `train_snake_purejaxrl_impl.py` - Implementation details
- `train_snake_purejaxrl_progressive.py` - Progressive training

✅ **Testing**: All modes verified with functional tests

## Example WandB Sweep

```yaml
program: train_snake_purejaxrl.py
method: bayes
metric:
  name: eval/mean_return
  goal: maximize
parameters:
  use-cnn:
    values: [true, false]
  cnn-features:
    values: 
      - [32, 64]
      - [64, 128]
      - [32, 64, 128]
  cnn-mode:
    values: [replace, append]
  d-model:
    values: [64, 128]
  num-layers:
    values: [2, 3, 4]
```

## Next Steps

Consider experimenting with:

- Different kernel sizes (currently fixed at 3x3)
- Pooling layers between CNN layers
- Residual connections in the CNN
- Different activation functions
- Batch normalization or layer normalization in CNN
