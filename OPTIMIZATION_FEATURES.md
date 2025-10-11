# Optimization Features Implementation

This document describes the performance optimization features that have been implemented in `sb3_snake.py` using environment variables from `.env`.

## Features Implemented

### 1. **Automatic Batch Size from Benchmarks**

- Default batch size is now read from `OPTIMAL_BATCH_SIZE` in `.env`
- Current optimal value: **8192** (from benchmark results)
- Can still be overridden with `--batch-size` CLI argument

### 2. **torch.compile() Support**

- Controlled by `USE_TORCH_COMPILE` environment variable
- Compiles the policy network, Q-network, and target Q-network for potential speedup
- Uses `mode="reduce-overhead"` for optimal performance
- Can be disabled with `--no-compile` flag

### 3. **FlashAttention (SDPA Backend)**

- Controlled by `USE_FLASH_ATTENTION` environment variable
- Uses PyTorch's Scaled Dot-Product Attention with FlashAttention backend
- Provides memory-efficient attention computation
- Can be disabled with `--no-flash-attention` flag

### 4. **BF16 Mixed Precision Training**

- Controlled by `USE_BF16` environment variable
- Uses automatic mixed precision (AMP) with bfloat16 dtype
- Reduces memory usage and increases throughput
- Works seamlessly with FlashAttention
- Can be disabled with `--no-bf16` flag

## Current Configuration (from .env)

```env
OPTIMAL_BATCH_SIZE=8192
USE_TORCH_COMPILE=false
USE_FLASH_ATTENTION=true
USE_BF16=true
```

Based on benchmark results, this configuration provides:

- **Peak throughput**: 24,334.8 samples/sec
- **Updates/sec**: 3.0
- **GPU memory**: 10.99 GB

## Usage

### Default Usage (uses .env settings)

```bash
python sb3_snake.py train --episodes 1000
```

This will automatically use:

- Batch size: 8192
- FlashAttention: Enabled
- BF16: Enabled
- torch.compile: Disabled

### Override Individual Settings

```bash
# Disable FlashAttention
python sb3_snake.py train --episodes 1000 --no-flash-attention

# Disable BF16 mixed precision
python sb3_snake.py train --episodes 1000 --no-bf16

# Use a different batch size
python sb3_snake.py train --episodes 1000 --batch-size 4096

# Combine multiple overrides
python sb3_snake.py train --episodes 1000 --batch-size 4096 --no-bf16 --no-flash-attention
```

### Testing Your Configuration

```bash
python test_optimizations.py
```

This will show:

- Current environment variable values
- PyTorch feature availability
- What optimizations will be active
- CLI flags to override settings

## Training Output

When training starts, you'll see a configuration summary:

```text
============================================================
Training Configuration Summary:
============================================================
Batch size: 8192
torch.compile: ✗ Disabled
FlashAttention: ✓ Enabled
BF16 (mixed precision): ✓ Enabled
============================================================

🔧 Using FlashAttention backend for SDPA
🚀 Starting training with BF16 mixed precision...
```

## Implementation Details

### Environment Variable Loading

```python
from dotenv import load_dotenv
load_dotenv()

# Helper functions for type conversion
def get_env_int(key: str, default: int) -> int
def get_env_float(key: str, default: float) -> float
def get_env_bool(key: str, default: bool) -> bool

# Read settings
ENV_BATCH_SIZE = get_env_int('OPTIMAL_BATCH_SIZE', DEFAULT_BATCH_SIZE)
ENV_USE_TORCH_COMPILE = get_env_bool('USE_TORCH_COMPILE', False)
ENV_USE_FLASH_ATTENTION = get_env_bool('USE_FLASH_ATTENTION', False)
ENV_USE_BF16 = get_env_bool('USE_BF16', False)
```

### torch.compile() Application

```python
if use_compile and hasattr(torch, 'compile'):
    model.policy.features_extractor = torch.compile(
        model.policy.features_extractor,
        mode="reduce-overhead"
    )
    model.q_net = torch.compile(model.q_net, mode="reduce-overhead")
    if hasattr(model, 'q_net_target'):
        model.q_net_target = torch.compile(model.q_net_target, mode="reduce-overhead")
```

### FlashAttention Context

```python
def get_sdpa_context():
    if use_flash and hasattr(torch.nn.attention, 'sdpa_kernel'):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])
    return nullcontext()
```

### BF16 Mixed Precision

```python
with get_sdpa_context():
    if use_bf16 and torch.cuda.is_available():
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            model.learn(total_timesteps=total_timesteps, callback=callbacks)
    else:
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
```

## Weights & Biases Integration

When using `--wandb`, optimization settings are automatically tracked:

```python
wandb_config = {
    # ... other configs ...
    "use_torch_compile": use_compile,
    "use_flash_attention": use_flash,
    "use_bf16": use_bf16,
}
```

## Requirements

Added to `requirements.txt`:

- `python-dotenv` - for loading `.env` files

Existing requirements for optimizations:

- PyTorch 2.0+ for `torch.compile()`
- PyTorch 2.0+ for FlashAttention SDPA backend
- CUDA GPU with BF16 support (e.g., RTX 4090)

## Performance Comparison

From benchmark results (`benchmark_results_full_20251012_001032.json`):

| Configuration | Samples/sec | Speedup |
|--------------|-------------|---------|
| Baseline (fp32) | ~15,000 | 1.0x |
| BF16 + FlashAttention | **24,334** | **1.62x** |
| BF16 + torch.compile | ~22,000 | 1.47x |
| BF16 + Both | ~23,000 | 1.53x |

**Recommendation**: The current `.env` configuration (BF16 + FlashAttention) provides the best throughput.

## Troubleshooting

### OOM (Out of Memory) Errors

If you encounter OOM errors with batch size 8192:

```bash
python sb3_snake.py train --batch-size 4096
```

### Compilation Warnings

If you see torch.compile warnings, you can disable it:

```bash
python sb3_snake.py train --no-compile
```

### CUDA/CUDNN Errors

If you encounter CUDA errors with FlashAttention:

```bash
python sb3_snake.py train --no-flash-attention
```

## Future Enhancements

Potential future improvements:

1. Add support for other SDPA backends (efficient_attention, math)
2. Add torch.compile mode selection (default, reduce-overhead, max-autotune)
3. Auto-detect optimal batch size based on available GPU memory
4. Add benchmarking results to wandb runs automatically
