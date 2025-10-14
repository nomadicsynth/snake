"""
Test GPU-native Snake JAX implementation
"""

import jax
import jax.numpy as jnp
from jax import random
import time

from snake_jax.env import SnakeEnv, make_reset_vmap, make_step_vmap
from snake_jax.network import TransformerPolicy
from snake_jax.config import EnvConfig


def test_single_env():
    """Test single environment"""
    print("=" * 60)
    print("TEST 1: Single Environment")
    print("=" * 60)
    
    config = EnvConfig(width=10, height=10)
    env = SnakeEnv(config)
    
    rng = random.PRNGKey(42)
    
    # Reset
    state = env.reset(rng)
    print(f"✓ Reset successful")
    print(f"  Snake length: {state.snake_length}")
    print(f"  Snake head: {state.snake_body[0]}")
    print(f"  Food position: {state.food_pos}")
    
    # Get observation
    obs = env._get_observation(state)
    print(f"✓ Observation shape: {obs.shape}")
    print(f"  Expected: ({config.height}, {config.width}, 3)")
    
    # Step
    action = jnp.int32(1)  # Move right
    state, obs, reward, done, info = env.step(state, action)
    print(f"✓ Step successful")
    print(f"  Reward: {reward}")
    print(f"  Done: {done}")
    print(f"  New snake head: {state.snake_body[0]}")
    
    # Run episode
    print("\n Running full episode...")
    episode_reward = 0.0
    steps = 0
    max_steps = 100
    
    while not state.done and steps < max_steps:
        # Random action
        rng, action_rng = random.split(state.rng)
        action = random.randint(action_rng, (), 0, 4)
        state, obs, reward, done, info = env.step(state, action)
        episode_reward += reward
        steps += 1
    
    print(f"✓ Episode complete")
    print(f"  Steps: {steps}")
    print(f"  Total reward: {episode_reward:.2f}")
    print(f"  Final score: {state.score}")
    print()


def test_vectorized_envs():
    """Test vectorized environments"""
    print("=" * 60)
    print("TEST 2: Vectorized Environments")
    print("=" * 60)
    
    config = EnvConfig(width=10, height=10)
    env = SnakeEnv(config)
    
    num_envs = 256
    print(f"Testing with {num_envs} parallel environments...")
    
    rng = random.PRNGKey(42)
    
    # Vectorized reset
    reset_vmap = make_reset_vmap(env, num_envs)
    
    start = time.time()
    states = reset_vmap(rng)
    jax.block_until_ready(states)
    reset_time = time.time() - start
    
    print(f"✓ Vectorized reset: {reset_time*1000:.2f}ms")
    print(f"  States shape: {states.snake_body.shape}")
    
    # Vectorized step
    step_vmap = make_step_vmap(env)
    actions = random.randint(rng, (num_envs,), 0, 4)
    
    start = time.time()
    states, obs, rewards, dones, infos = step_vmap(states, actions)
    jax.block_until_ready(states)
    step_time = time.time() - start
    
    print(f"✓ Vectorized step: {step_time*1000:.2f}ms")
    print(f"  Observations shape: {obs.shape}")
    print(f"  Rewards shape: {rewards.shape}")
    
    # Benchmark throughput
    print("\n Benchmarking throughput...")
    num_steps = 100
    
    start = time.time()
    for _ in range(num_steps):
        actions = random.randint(rng, (num_envs,), 0, 4)
        states, obs, rewards, dones, infos = step_vmap(states, actions)
    jax.block_until_ready(states)
    elapsed = time.time() - start
    
    fps = (num_envs * num_steps) / elapsed
    print(f"✓ Benchmark complete")
    print(f"  Total steps: {num_steps}")
    print(f"  Elapsed time: {elapsed:.3f}s")
    print(f"  FPS: {fps:,.0f}")
    print()


def test_network():
    """Test network"""
    print("=" * 60)
    print("TEST 3: Transformer Network")
    print("=" * 60)
    
    config = EnvConfig(width=10, height=10)
    
    # Create network
    network = TransformerPolicy(
        d_model=64,
        num_layers=2,
        num_heads=4,
        num_actions=4
    )
    
    # Initialize
    rng = random.PRNGKey(42)
    obs_shape = (config.height, config.width, 3)
    params = network.init_params(rng, obs_shape)
    
    print(f"✓ Network initialized")
    
    # Count parameters
    num_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"  Total parameters: {num_params:,}")
    
    # Forward pass
    batch_size = 32
    dummy_obs = jnp.zeros((batch_size, *obs_shape), dtype=jnp.float32)
    
    start = time.time()
    logits, values = network.apply(params, dummy_obs, training=False)
    jax.block_until_ready((logits, values))
    forward_time = time.time() - start
    
    print(f"✓ Forward pass: {forward_time*1000:.2f}ms")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Values shape: {values.shape}")
    
    # Benchmark with larger batch
    batch_size = 2048
    dummy_obs = jnp.zeros((batch_size, *obs_shape), dtype=jnp.float32)
    
    # Warmup
    logits, values = network.apply(params, dummy_obs, training=False)
    jax.block_until_ready((logits, values))
    
    # Benchmark
    num_iters = 100
    start = time.time()
    for _ in range(num_iters):
        logits, values = network.apply(params, dummy_obs, training=False)
    jax.block_until_ready((logits, values))
    elapsed = time.time() - start
    
    samples_per_sec = (batch_size * num_iters) / elapsed
    print(f"\n Benchmark (batch={batch_size}):")
    print(f"  Iterations: {num_iters}")
    print(f"  Time per iter: {elapsed/num_iters*1000:.2f}ms")
    print(f"  Throughput: {samples_per_sec:,.0f} samples/sec")
    print()


def test_full_integration():
    """Test full integration"""
    print("=" * 60)
    print("TEST 4: Full Integration Test")
    print("=" * 60)
    
    config = EnvConfig(width=8, height=8)
    env = SnakeEnv(config)
    num_envs = 1024
    
    # Create network
    network = TransformerPolicy(
        d_model=64,
        num_layers=2,
        num_heads=4,
        num_actions=4
    )
    
    # Initialize
    rng = random.PRNGKey(42)
    obs_shape = env.observation_space_shape()
    params = network.init_params(rng, obs_shape)
    
    print(f"✓ Setup complete")
    print(f"  Environments: {num_envs}")
    print(f"  Network params: {sum(x.size for x in jax.tree.leaves(params)):,}")
    
    # Reset environments
    reset_vmap = make_reset_vmap(env, num_envs)
    states = reset_vmap(rng)
    
    print(f"✓ Environments reset")
    
    # Collect rollout
    print("\n Collecting rollout...")
    num_steps = 128
    
    @jax.jit
    def collect_step(states, rng):
        # Get observations
        obs = jax.vmap(env._get_observation)(states)
        
        # Get actions from network
        logits, values = jax.vmap(lambda o: network.apply(params, o[None], training=False))(obs)
        
        # Sample actions
        rng, action_rng = random.split(rng)
        action_rngs = random.split(action_rng, num_envs)
        actions = jax.vmap(lambda l, r: random.categorical(r, l))(logits.squeeze(1), action_rngs)
        
        # Step environments
        step_vmap = make_step_vmap(env)
        new_states, next_obs, rewards, dones, _ = step_vmap(states, actions)
        
        return new_states, rewards, rng
    
    total_reward = 0.0
    start = time.time()
    
    for step in range(num_steps):
        states, rewards, rng = collect_step(states, rng)
        jax.block_until_ready(states)
        total_reward += rewards.sum()
    
    elapsed = time.time() - start
    fps = (num_envs * num_steps) / elapsed
    
    print(f"✓ Rollout complete")
    print(f"  Steps: {num_steps}")
    print(f"  Total environments steps: {num_envs * num_steps:,}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  FPS: {fps:,.0f}")
    print(f"  Average reward per env: {total_reward / num_envs:.2f}")
    print()


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("🚀 GPU-NATIVE SNAKE JAX TESTS")
    print("=" * 60)
    print()
    
    # Check JAX setup
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    print()
    
    try:
        test_single_env()
        test_vectorized_envs()
        test_network()
        test_full_integration()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print()
        print("Ready to train with:")
        print("  python -m snake_jax.train_ppo")
        print()
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
