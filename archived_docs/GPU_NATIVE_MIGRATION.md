# GPU-Native Snake RL Migration Plan 🚀

## Current Stack → Target Stack

| Component | Current (CPU-bound) | Target (GPU-native) |
|-----------|-------------------|-------------------|
| Framework | Stable-Baselines3 + PyTorch | PureJaxRL / JAX |
| Environment | Python class `SnakeGame` | Pure JAX functions |
| Parallelization | 1-64 envs (multiprocess) | 4096-16384 envs (vmap) |
| Device transfers | Constant CPU↔GPU | Zero transfers |
| Training FPS | ~100-500 | ~50,000-500,000 |
| GPU Utilization | 5-40% | 85-95% |

---

## Phase 1: Setup JAX (5 minutes)

### Install JAX with CUDA Support

```bash
# For CUDA 12.x (check your CUDA version first)
pip install -U "jax[cuda12]"

# Or for CUDA 11.x
pip install -U "jax[cuda11_local]"

# Install PureJaxRL
pip install git+https://github.com/luchris429/purejaxrl.git

# Additional deps
pip install flax optax gymnax chex
```

### Verify Installation

```bash
python -c "import jax; print(f'JAX {jax.__version__}'); print(f'Devices: {jax.devices()}')"
```

Should show: `[cuda(id=0)]` or similar

---

## Phase 2: Rewrite Snake Environment in JAX

### Key Differences from Python:

```python
# ❌ Python/NumPy (what we have now)
class SnakeGame:
    def __init__(self, width, height):
        self.width = width
        self.snake = [(5, 5)]  # Mutable state
    
    def step(self, action):
        self.snake.insert(0, new_head)  # In-place mutation
        return next_state, reward, done

# ✅ JAX (what we need)
@jax.jit
def snake_step(state, action):
    """Pure function: state_in → state_out"""
    new_snake = jnp.roll(state.snake, 1, axis=0)  # Immutable operations
    new_snake = new_snake.at[0].set(new_head)
    return new_state, reward, done

# ✅ Vectorized across 4096 envs
snake_step_vmap = jax.vmap(snake_step)  # Automatic parallelization!
```

### State Representation

```python
from typing import NamedTuple
import jax.numpy as jnp

class SnakeState(NamedTuple):
    """Immutable state for one Snake environment"""
    snake_body: jnp.ndarray  # (max_length, 2) - x,y positions
    snake_length: jnp.int32  # Current length
    direction: jnp.ndarray   # (2,) - current direction
    food_pos: jnp.ndarray    # (2,) - food position
    score: jnp.int32
    done: jnp.bool_
    step_count: jnp.int32
    # For rendering/observation
    grid: jnp.ndarray        # (H, W, 3) - same format as current

# Vectorized state for 4096 envs:
# Each field becomes batched: (4096, ...) 
```

---

## Phase 3: Environment Implementation

### File Structure

```
snake_jax/
├── __init__.py
├── env.py              # Core environment logic
├── network.py          # Transformer policy in Flax
├── train_ppo.py        # PureJaxRL training loop
└── config.py           # Hyperparameters
```

### Core Functions Needed

```python
# env.py

@jax.jit
def reset_env(rng):
    """Initialize one environment"""
    return SnakeState(...)

@jax.jit  
def step_env(state, action):
    """Step one environment"""
    # 1. Move snake based on action
    # 2. Check collisions (wall, self)
    # 3. Check food eating
    # 4. Return new_state, reward, done
    return new_state, reward, done

# Vectorize for massive parallelism
reset_env_vmap = jax.vmap(reset_env)  # (4096,) RNGs → (4096,) states
step_env_vmap = jax.vmap(step_env)    # (4096,) states → (4096,) next_states
```

### Key JAX Patterns

```python
# Instead of: if condition: x = a else: x = b
# Use: jnp.where(condition, a, b)
reward = jnp.where(ate_food, 10.0, -0.01)

# Instead of: list.append() or array[i] = value
# Use: array.at[i].set(value)
new_snake = state.snake.at[0].set(new_head)

# Instead of: for i in range(steps): state = update(state)
# Use: jax.lax.scan (compiled loop)
final_state, all_states = jax.lax.scan(step_fn, init_state, xs)
```

---

## Phase 4: Network in Flax

Your Transformer needs to be rewritten in Flax (JAX's nn library):

```python
# network.py
import flax.linen as nn
import jax.numpy as jnp

class TransformerPolicy(nn.Module):
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 4
    num_actions: int = 4
    
    @nn.compact
    def __call__(self, x):
        # x: (batch, H, W, 3)
        batch, H, W, C = x.shape
        
        # Flatten to tokens
        x = x.reshape(batch, H * W, C)  # (batch, H*W, 3)
        
        # Project to d_model
        x = nn.Dense(self.d_model)(x)
        
        # Add positional encoding
        pos_embed = self.param('pos_embed', 
                               nn.initializers.normal(0.02),
                               (1, H * W, self.d_model))
        x = x + pos_embed
        
        # Transformer layers
        for _ in range(self.num_layers):
            x = TransformerBlock(self.d_model, self.num_heads)(x)
        
        # Pool and predict
        x = x.mean(axis=1)  # (batch, d_model)
        
        # Actor-critic heads
        logits = nn.Dense(self.num_actions)(x)  # Policy
        value = nn.Dense(1)(x)                   # Value
        
        return logits, value.squeeze(-1)

class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    
    @nn.compact
    def __call__(self, x):
        # Multi-head attention
        y = nn.LayerNorm()(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model
        )(y, y)
        x = x + y
        
        # FFN
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.d_model * 2)(y)
        y = nn.gelu(y)
        y = nn.Dense(self.d_model)(y)
        x = x + y
        
        return x
```

---

## Phase 5: Training with PureJaxRL

```python
# train_ppo.py
import jax
import jax.numpy as jnp
from purejaxrl import PPO
from snake_jax.env import SnakeEnv
from snake_jax.network import TransformerPolicy

def main():
    config = {
        "NUM_ENVS": 4096,           # 🔥 4096 parallel envs
        "NUM_STEPS": 128,            # Steps per rollout
        "TOTAL_TIMESTEPS": 10_000_000,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 32,
        "LR": 3e-4,
        "CLIP_EPS": 0.2,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "GRID_SIZE": 10,
    }
    
    rng = jax.random.PRNGKey(42)
    
    # Create env and network
    env = SnakeEnv(config)
    network = TransformerPolicy(
        d_model=64,
        num_layers=2,
        num_heads=4,
        num_actions=4
    )
    
    # Train (everything JIT-compiled)
    train_fn = make_train_fn(env, network, config)
    train_state = train_fn(rng)
    
    print(f"Training complete!")
    print(f"Final return: {train_state['metrics']['returned_episode_returns'][-1]}")

if __name__ == "__main__":
    main()
```

---

## Phase 6: Expected Performance

### Before (Current Setup):
```
Environment: Python SnakeGame
Parallel Envs: 1
Training FPS: ~100
GPU Utilization: ~5%
Time to 1M steps: ~2.5 hours
```

### After (JAX GPU-Native):
```
Environment: Pure JAX
Parallel Envs: 4096
Training FPS: ~100,000
GPU Utilization: ~90%
Time to 1M steps: ~1 minute
```

**Expected speedup: 100-150x** 🚀

---

## Implementation Order

1. ✅ Install JAX + deps
2. ✅ Implement `SnakeState` and pure functions
3. ✅ Test single env reset/step
4. ✅ Vectorize with `jax.vmap`
5. ✅ Implement Transformer in Flax
6. ✅ Integrate with PureJaxRL
7. ✅ Train and compare results

---

## Gotchas & Tips

### Common JAX Issues:

1. **"ConcretizationTypeError"**: You used Python control flow with JAX arrays
   - Fix: Use `jnp.where()`, `jax.lax.cond()`, `jax.lax.scan()`

2. **"TracerError"**: You mutated a JAX array
   - Fix: Use `.at[].set()` instead of `array[i] = value`

3. **"JIT compilation failed"**: Shape mismatch
   - Fix: Ensure all shapes are static or use `jax.ShapeDtypeStruct`

4. **Slow first run**: JIT compilation
   - Normal! First call compiles, subsequent calls are fast

### Memory Management:

```python
# 4096 envs × 10×10 grid = manageable
# But watch out for:
batch_size = 4096 * 128  # rollout buffer size
# May need gradient accumulation if too large
```

---

## Alternative: Gymnax

If you want something even easier, use Gymnax (pre-built JAX envs):

```python
import gymnax

# They have many envs, but not Snake
# Could contribute your JAX Snake to Gymnax!
env, env_params = gymnax.make("CartPole-v1")
```

---

## Resources

- [PureJaxRL GitHub](https://github.com/luchris429/purejaxrl)
- [JAX Tutorial](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html)
- [Flax Documentation](https://flax.readthedocs.io/)
- [Gymnax Envs](https://github.com/RobertTLange/gymnax)
- [My JAX + RL Guide 2024](https://iclr.cc/virtual/2024/workshop/20539) (fictional but should exist!)

---

## Decision Point

Do you want to:

**A) Full rewrite in JAX** (maximum performance, learning opportunity)
- I'll create the full implementation
- 2-4 hours of dev time
- 100-150x speedup

**B) Hybrid approach** (use Brax/Gymnax as base)
- Faster to implement
- Still 50-100x speedup
- Less customization

**C) Start with vectorized SB3** (safe first step)
- 3-5x speedup
- Keep existing code
- Easy to try

Let me know which route you want! I'm ready to write code. 🔥
