#!/usr/bin/env python
"""
Quick test of the evaluation function
"""

import os
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_gemm=false '
    '--xla_gpu_autotune_level=0 '
    '--xla_gpu_force_compilation_parallelism=1'
)

import jax
import jax.numpy as jnp
from snake_jax.config import EnvConfig
from snake_jax.gymnax_wrapper import SnakeGymnaxWrapper
from snake_jax.network import TransformerPolicy
from train_snake_purejaxrl_impl import make_evaluate_fn

print("Testing evaluation function...")
print(f"Device: {jax.devices()[0]}")
print()

# Create environment
env_config = EnvConfig(width=10, height=10, max_steps=500)
env = SnakeGymnaxWrapper(env_config)
env_params = env.default_params

# Create random network
network = TransformerPolicy(
    d_model=64,
    num_layers=2,
    num_heads=4,
    num_actions=4,
    dropout_rate=0.1
)

# Initialize network with random params
rng = jax.random.PRNGKey(42)
rng, init_rng, dropout_rng = jax.random.split(rng, 3)
init_obs = jnp.zeros(env.observation_space(env_params).shape)
params = network.init({'params': init_rng, 'dropout': dropout_rng}, init_obs[None], training=False)

print("Network initialized")
print()

# Create evaluation function
evaluate_fn = make_evaluate_fn(env, env_params, num_episodes=32)

print("Running evaluation (this will compile on first run)...")
rng, eval_rng = jax.random.split(rng)
eval_metrics = evaluate_fn(network, params, eval_rng)

# Convert to Python scalars for printing
eval_metrics = jax.tree_util.tree_map(lambda x: float(x), eval_metrics)

print()
print("Evaluation results:")
print(f"  Mean return: {eval_metrics['mean_return']:.2f} ± {eval_metrics['std_return']:.2f}")
print(f"  Mean length: {eval_metrics['mean_length']:.2f}")
print(f"  Mean score: {eval_metrics['mean_score']:.2f}")
print(f"  Max return: {eval_metrics['max_return']:.2f}")
print(f"  Max score: {eval_metrics['max_score']:.2f}")
print(f"  Min return: {eval_metrics['min_return']:.2f}")
print()

print("✅ Evaluation function works!")
