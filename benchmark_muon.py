"""
Benchmark Muon vs Adam optimizer performance on Snake Transformer network

This script measures the training step time for Muon vs Adam optimizers
using the same network architecture and data as the Snake training.
"""

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import time
from typing import Dict, Any
import argparse

# Import the network and muon optimizer
from snake_jax.network import TransformerPolicy
from muon_jax import chain_with_muon


def create_fake_batch(batch_size: int, height: int, width: int, num_actions: int):
    """Create fake training batch similar to Snake environment"""
    # Fake observations: (batch, height, width, 3)
    obs = jax.random.normal(jax.random.PRNGKey(0), (batch_size, height, width, 3))

    # Fake actions: (batch,)
    actions = jax.random.randint(jax.random.PRNGKey(1), (batch_size,), 0, num_actions)

    # Fake advantages and returns
    advantages = jax.random.normal(jax.random.PRNGKey(2), (batch_size,))
    returns = jax.random.normal(jax.random.PRNGKey(3), (batch_size,))

    # Fake log probs from old policy
    old_log_probs = jax.random.normal(jax.random.PRNGKey(4), (batch_size,))

    return obs, actions, advantages, returns, old_log_probs


def ppo_loss_fn(network, params, batch, rng, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    """PPO loss function (simplified version)"""
    obs, actions, advantages, returns, old_log_probs = batch

    # Forward pass
    logits, values = network.apply(params, obs, training=True, rngs={'dropout': rng})

    # Policy loss (simplified - just cross entropy for benchmarking)
    log_probs = jax.nn.log_softmax(logits)
    action_log_probs = log_probs[jnp.arange(logits.shape[0]), actions]

    # Simple policy loss (not full PPO for benchmarking)
    policy_loss = -jnp.mean(action_log_probs)

    # Value loss
    value_loss = jnp.mean((values - returns) ** 2)

    # Entropy bonus
    probs = jax.nn.softmax(logits)
    entropy = -jnp.sum(probs * log_probs, axis=-1).mean()
    entropy_loss = -ent_coef * entropy

    # Total loss
    total_loss = policy_loss + vf_coef * value_loss + entropy_loss

    return total_loss


def benchmark_optimizer(
    optimizer_name: str,
    network: nn.Module,
    params: Dict,
    batch: tuple,
    rng: jax.random.PRNGKey,
    num_steps: int = 100,
    warmup_steps: int = 10
) -> Dict[str, float]:
    """Benchmark an optimizer for multiple steps"""

    # Create optimizer
    if optimizer_name == "muon":
        # Use same settings as training
        tx = chain_with_muon(
            muon_lr=0.02,
            aux_lr=2.5e-4,
            max_grad_norm=0.5,
            momentum=0.95,
            nesterov=True,
        )
    elif optimizer_name == "adam":
        tx = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(2.5e-4, eps=1e-5),
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Create train state
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx,
    )

    # JIT compile the step function
    @jax.jit
    def train_step(state, batch, rng):
        def loss_fn(params):
            return ppo_loss_fn(network, params, batch, rng)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    # Warmup
    print(f"  Warming up {optimizer_name}...")
    state = train_state
    rng_step = rng
    for _ in range(warmup_steps):
        rng_step, step_rng = jax.random.split(rng_step)
        state, _ = train_step(state, batch, step_rng)

    # Benchmark
    print(f"  Benchmarking {optimizer_name} for {num_steps} steps...")
    start_time = time.time()

    losses = []
    for _ in range(num_steps):
        rng_step, step_rng = jax.random.split(rng_step)
        state, loss = train_step(state, batch, step_rng)
        losses.append(loss)

    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_step = total_time / num_steps
    avg_loss = jnp.mean(jnp.array(losses))

    return {
        "total_time": total_time,
        "avg_time_per_step": avg_time_per_step,
        "avg_loss": float(avg_loss),
        "steps_per_sec": num_steps / total_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Muon vs Adam optimizers")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size")
    parser.add_argument("--height", type=int, default=10, help="Board height")
    parser.add_argument("--width", type=int, default=10, help="Board width")
    parser.add_argument("--d-model", type=int, default=64, help="Model dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num-steps", type=int, default=100, help="Number of benchmark steps")
    parser.add_argument("--warmup-steps", type=int, default=10, help="Warmup steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print("=" * 70)
    print("🚀 MUON VS ADAM OPTIMIZER BENCHMARK")
    print("=" * 70)
    print()

    print("Configuration:")
    print(f"  Device: {jax.devices()[0]}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Board size: {args.height}x{args.width}")
    print(f"  Network: d_model={args.d_model}, layers={args.num_layers}, heads={args.num_heads}")
    print(f"  Benchmark steps: {args.num_steps}")
    print()

    # Set random seed
    rng = jax.random.PRNGKey(args.seed)

    # Create network
    network = TransformerPolicy(
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_actions=4,  # Snake has 4 actions
        dropout_rate=0.1
    )

    # Initialize parameters
    print("Initializing network...")
    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.zeros((1, args.height, args.width, 3), dtype=jnp.float32)
    params = network.init(init_rng, dummy_obs, training=False)

    # Count parameters
    param_count = sum(jnp.prod(jnp.array(p.shape)) for p in jax.tree_util.tree_leaves(params))
    print(f"  Parameters: {param_count:,}")
    # Create fake batch
    print("Creating fake training batch...")
    batch = create_fake_batch(args.batch_size, args.height, args.width, 4)

    print()
    print("Benchmarking optimizers...")
    print("-" * 50)

    # Split rng for benchmarks
    rng_adam, rng_muon = jax.random.split(rng)

    # Benchmark Adam
    adam_results = benchmark_optimizer(
        "adam", network, params, batch, rng_adam,
        num_steps=args.num_steps, warmup_steps=args.warmup_steps
    )

    # Benchmark Muon
    muon_results = benchmark_optimizer(
        "muon", network, params, batch, rng_muon,
        num_steps=args.num_steps, warmup_steps=args.warmup_steps
    )

    print()
    print("RESULTS:")
    print("=" * 50)

    print("Adam:")
    print(f"  Avg time per step: {adam_results['avg_time_per_step']:.4f}s")
    print(f"  Steps per second: {adam_results['steps_per_sec']:.2f}")
    print(f"  Final loss: {adam_results['avg_loss']:.4f}")

    print()
    print("Muon:")
    print(f"  Avg time per step: {muon_results['avg_time_per_step']:.4f}s")
    print(f"  Steps per second: {muon_results['steps_per_sec']:.2f}")
    print(f"  Final loss: {muon_results['avg_loss']:.4f}")

    print()
    print("Comparison:")
    print("-" * 30)
    speedup = muon_results["avg_time_per_step"] / adam_results["avg_time_per_step"]
    print(f"  Muon slowdown factor: {speedup:.2f}x")
    print(f"  Muon is {speedup:.2f}x slower than Adam")

    if speedup > 1:
        print("⚠️  Muon is slower than Adam!")
    else:
        print("✅ Muon is faster than Adam!")

    print()
    print("Raw timing data:")
    print(f"  Adam total time: {adam_results['total_time']:.4f}s")
    print(f"  Muon total time: {muon_results['total_time']:.4f}s")


if __name__ == "__main__":
    main()