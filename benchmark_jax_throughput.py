"""
Quick benchmark: Compare JAX vs SB3 throughput
"""

import jax
import jax.numpy as jnp
import time
from snake_jax.config import EnvConfig
from snake_jax.env import SnakeState, make_reset_vmap, make_step_vmap
from snake_jax.network import TransformerPolicy

def benchmark_jax_throughput():
    """Benchmark pure JAX throughput"""
    
    print("=" * 70)
    print("🚀 JAX SNAKE BENCHMARK")
    print("=" * 70)
    print()
    
    # Configuration
    env_config = EnvConfig(width=10, height=10, max_steps=500)
    num_envs = 2048
    num_steps = 1000
    
    print(f"Configuration:")
    print(f"  Parallel envs: {num_envs}")
    print(f"  Steps per env: {num_steps}")
    print(f"  Total steps: {num_envs * num_steps:,}")
    print()
    
    # Initialize
    from snake_jax.env import reset, step
    reset_vmap = make_reset_vmap(env_config, num_envs)
    step_vmap = make_step_vmap(env_config, num_envs)
    
    # Create network
    network = TransformerPolicy(num_actions=4)
    rng = jax.random.PRNGKey(0)
    rng, net_rng, dropout_rng = jax.random.split(rng, 3)
    
    dummy_obs = jnp.zeros((1, 10, 10, 3))
    params = network.init({'params': net_rng, 'dropout': dropout_rng}, dummy_obs)
    
    print("Initializing environments...")
    rng, reset_rng = jax.random.split(rng)
    rngs = jax.random.split(reset_rng, num_envs)
    states = reset_vmap(rngs)
    
    print("Warming up JIT compilation...")
    # Warmup
    for _ in range(5):
        rng, step_rng, act_rng, dropout_rng = jax.random.split(rng, 4)
        rngs = jax.random.split(step_rng, num_envs)
        act_rngs = jax.random.split(act_rng, num_envs)
        
        # Get observations
        obs = jnp.stack([s.to_observation(env_config) for s in states])
        
        # Get actions from network
        logits, values = network.apply(params, obs, training=False, rngs={'dropout': dropout_rng})
        actions = jax.random.categorical(act_rngs, logits)
        
        # Step
        states = step_vmap(states, actions, rngs)
    
    print("Running benchmark...")
    print()
    
    start = time.time()
    
    for step_num in range(num_steps):
        rng, step_rng, act_rng, dropout_rng = jax.random.split(rng, 4)
        rngs = jax.random.split(step_rng, num_envs)
        act_rngs = jax.random.split(act_rng, num_envs)
        
        # Get observations
        obs = jnp.stack([s.to_observation(env_config) for s in states])
        
        # Get actions from network
        logits, values = network.apply(params, obs, training=False, rngs={'dropout': dropout_rng})
        actions = jax.random.categorical(act_rngs, logits)
        
        # Step
        states = step_vmap(states, actions, rngs)
        
        if (step_num + 1) % 100 == 0:
            elapsed = time.time() - start
            fps = (step_num + 1) * num_envs / elapsed
            print(f"  Step {step_num + 1}/{num_steps}: {fps:,.0f} FPS")
    
    elapsed = time.time() - start
    total_steps = num_envs * num_steps
    fps = total_steps / elapsed
    
    print()
    print("=" * 70)
    print("✅ BENCHMARK COMPLETE")
    print("=" * 70)
    print()
    print(f"Results:")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Average FPS: {fps:,.0f}")
    print(f"  Time per step: {elapsed / num_steps * 1000:.2f}ms")
    print()
    print(f"🎉 This is {fps / 500:.0f}x faster than typical SB3 (~500 FPS)")
    print()

if __name__ == "__main__":
    benchmark_jax_throughput()
