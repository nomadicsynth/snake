"""
Train Snake using PureJaxRL's PPO implementation

This integrates our GPU-native Snake environment with PureJaxRL's 
battle-tested PPO algorithm.
"""

# Set XLA flags to avoid ptxas compilation errors
import os
# Disable Triton GEMM and use more conservative compilation
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_gemm=false '
    '--xla_gpu_autotune_level=0 '
    '--xla_gpu_force_compilation_parallelism=1'
)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'

import jax
import jax.numpy as jnp
import sys

from snake_jax.config import EnvConfig
from snake_jax.gymnax_wrapper import SnakeGymnaxWrapper
from snake_jax.network import TransformerPolicy
import time


def main():
    print("=" * 70)
    print("🚀 GPU-NATIVE SNAKE TRAINING WITH PUREJAXRL PPO")
    print("=" * 70)
    print()
    
    # Environment configuration
    env_config = EnvConfig(
        width=10,
        height=10,
        max_steps=500,
        apple_reward=10.0,
        death_penalty=-10.0,
        step_penalty=-0.01
    )
    
    # Training hyperparameters (dict format for PureJaxRL)
    config = {
        "NUM_ENVS": 2048,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 5_000_000,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 32,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "LR": 2.5e-4,
        "ANNEAL_LR": True,
    }
    
    # Derived values
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    
    print(f"Configuration:")
    print(f"  Environment: {env_config.width}x{env_config.height} Snake")
    print(f"  Device: {jax.devices()[0]}")
    print(f"  Parallel envs: {config['NUM_ENVS']}")
    print(f"  Steps per rollout: {config['NUM_STEPS']}")
    print(f"  Total timesteps: {config['TOTAL_TIMESTEPS']:,}")
    print(f"  Number of updates: {config['NUM_UPDATES']:,}")
    print(f"  Minibatch size: {config['MINIBATCH_SIZE']}")
    print()
    
    # Create environment
    env = SnakeGymnaxWrapper(env_config)
    env_params = env.default_params
    
    print(f"Environment:")
    print(f"  Observation shape: {env.observation_space(env_params).shape}")
    print(f"  Action space: {env.action_space(env_params).n}")
    print()
    
    # Modify make_train to use our environment
    # We'll create a custom training function based on PureJaxRL's structure
    from train_snake_purejaxrl_impl import make_train_custom
    
    train_fn = make_train_custom(config, env, env_params)
    
    # Initialize RNG
    rng = jax.random.PRNGKey(42)
    
    print("Starting training...")
    print("=" * 70)
    print()
    
    start_time = time.time()
    
    # Train!
    train_state = train_fn(rng)
    
    elapsed = time.time() - start_time
    
    print()
    print("=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print()
    print(f"Results:")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Total steps: {config['TOTAL_TIMESTEPS']:,}")
    print(f"  FPS: {config['TOTAL_TIMESTEPS'] / elapsed:,.0f}")
    print(f"  Time per update: {elapsed / config['NUM_UPDATES']:.3f}s")
    print()
    
    # Try to extract metrics if available
    if hasattr(train_state, 'metrics'):
        metrics = train_state.metrics
        print(f"Training Metrics:")
        for key, val in metrics.items():
            if 'return' in key.lower():
                print(f"  {key}: {val}")
    
    print()
    print("🎉 GPU-native training complete!")
    print(f"   Compare to SB3: {elapsed:.1f}s vs ~30-50 minutes")
    print(f"   Speedup: ~{(30*60)/elapsed:.0f}x faster!")
    print()


if __name__ == "__main__":
    main()
