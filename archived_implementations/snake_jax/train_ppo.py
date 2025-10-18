"""
Pure JAX PPO Implementation for Snake

All training happens on GPU with massive parallelization.
"""

import jax
import jax.numpy as jnp
from jax import random
import optax
from functools import partial
from typing import NamedTuple, Tuple
import time

from snake_jax.env import SnakeEnv, SnakeState
from snake_jax.network import TransformerPolicy
from snake_jax.config import EnvConfig


class PPOConfig(NamedTuple):
    """PPO hyperparameters"""
    # Environment
    num_envs: int = 2048  # Number of parallel environments
    num_steps: int = 128   # Steps per rollout
    
    # Training
    num_epochs: int = 4
    num_minibatches: int = 32
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    
    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # Logging
    log_interval: int = 10  # Log every N updates
    
    @property
    def batch_size(self):
        return self.num_envs * self.num_steps
    
    @property
    def minibatch_size(self):
        return self.batch_size // self.num_minibatches
    
    @property
    def num_updates_per_epoch(self):
        return self.num_minibatches


class RolloutBatch(NamedTuple):
    """Batch of rollout data"""
    obs: jnp.ndarray           # (num_envs * num_steps, H, W, C)
    actions: jnp.ndarray       # (num_envs * num_steps,)
    log_probs: jnp.ndarray     # (num_envs * num_steps,)
    values: jnp.ndarray        # (num_envs * num_steps,)
    rewards: jnp.ndarray       # (num_envs * num_steps,)
    dones: jnp.ndarray         # (num_envs * num_steps,)
    advantages: jnp.ndarray    # (num_envs * num_steps,)
    returns: jnp.ndarray       # (num_envs * num_steps,)


class TrainState(NamedTuple):
    """Training state"""
    params: dict
    opt_state: optax.OptState
    env_states: SnakeState
    rng: jax.random.PRNGKey
    step: int


@partial(jax.jit, static_argnums=(0, 1))
def sample_action(network: TransformerPolicy, params: dict, obs: jnp.ndarray, 
                 rng: jax.random.PRNGKey, training: bool = True):
    """Sample action from policy"""
    logits, value = network.apply(params, obs, training=training)
    
    # Sample action from categorical distribution
    action = random.categorical(rng, logits, axis=-1)
    
    # Compute log probability
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    log_prob = jnp.take_along_axis(log_probs, action[..., None], axis=-1).squeeze(-1)
    
    return action, log_prob, value


@partial(jax.jit, static_argnums=(0,))
def rollout_step(env: SnakeEnv, carry, unused):
    """Single step of rollout collection"""
    network, params, env_states, rng = carry
    
    # Get observations
    obs_list = jax.vmap(env._get_observation)(env_states)
    
    # Sample actions
    rng, action_rng = random.split(rng)
    action_rngs = random.split(action_rng, env.config.num_envs if hasattr(env.config, 'num_envs') else 1)
    actions, log_probs, values = jax.vmap(
        lambda o, r: sample_action(network, params, o[None], r, training=True)
    )(obs_list, action_rngs)
    
    # Step environments
    step_results = jax.vmap(env.step)(env_states, actions)
    new_states, next_obs, rewards, dones, infos = step_results
    
    # Store transition
    transition = (obs_list, actions, log_probs, values, rewards, dones)
    
    carry = (network, params, new_states, rng)
    return carry, transition


@jax.jit
def compute_gae(rewards, values, dones, last_value, gamma=0.99, gae_lambda=0.95):
    """
    Compute Generalized Advantage Estimation
    
    Args:
        rewards: (num_steps, num_envs)
        values: (num_steps, num_envs)
        dones: (num_steps, num_envs)
        last_value: (num_envs,)
    
    Returns:
        advantages: (num_steps, num_envs)
        returns: (num_steps, num_envs)
    """
    num_steps = rewards.shape[0]
    
    advantages = jnp.zeros_like(rewards)
    last_gae = 0.0
    
    # Backward loop for GAE
    def scan_fn(gae, t):
        t_idx = num_steps - 1 - t
        
        # Next value (either next state value or last_value for final step)
        next_value = jnp.where(
            t_idx == num_steps - 1,
            last_value,
            values[t_idx + 1]
        )
        
        # TD residual
        next_non_terminal = 1.0 - dones[t_idx]
        delta = rewards[t_idx] + gamma * next_value * next_non_terminal - values[t_idx]
        
        # GAE
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        
        return gae, gae
    
    _, advantages = jax.lax.scan(
        scan_fn,
        last_gae,
        jnp.arange(num_steps)
    )
    
    # Reverse to get correct order
    advantages = advantages[::-1]
    
    # Returns
    returns = advantages + values
    
    return advantages, returns


@partial(jax.jit, static_argnums=(0,))
def compute_ppo_loss(network: TransformerPolicy, params: dict, batch: RolloutBatch, 
                    clip_eps: float, value_coef: float, entropy_coef: float):
    """Compute PPO loss"""
    # Forward pass
    logits, values = network.apply(params, batch.obs, training=True)
    
    # Policy loss
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    new_log_probs = jnp.take_along_axis(
        log_probs, batch.actions[..., None], axis=-1
    ).squeeze(-1)
    
    # Ratio and clipped surrogate
    ratio = jnp.exp(new_log_probs - batch.log_probs)
    advantages_normalized = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
    
    surr1 = ratio * advantages_normalized
    surr2 = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages_normalized
    policy_loss = -jnp.minimum(surr1, surr2).mean()
    
    # Value loss (clipped)
    value_pred_clipped = batch.values + jnp.clip(
        values - batch.values, -clip_eps, clip_eps
    )
    value_loss1 = jnp.square(values - batch.returns)
    value_loss2 = jnp.square(value_pred_clipped - batch.returns)
    value_loss = 0.5 * jnp.maximum(value_loss1, value_loss2).mean()
    
    # Entropy bonus
    probs = jax.nn.softmax(logits, axis=-1)
    entropy = -jnp.sum(probs * log_probs, axis=-1).mean()
    
    # Total loss
    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    
    metrics = {
        'loss': total_loss,
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'entropy': entropy,
        'approx_kl': ((ratio - 1) - jnp.log(ratio)).mean()
    }
    
    return total_loss, metrics


def train_ppo(
    config: PPOConfig,
    env_config: EnvConfig,
    total_timesteps: int,
    seed: int = 42
):
    """
    Train PPO on Snake
    
    Args:
        config: PPO configuration
        env_config: Environment configuration
        total_timesteps: Total number of environment steps
        seed: Random seed
    """
    # Initialize
    rng = random.PRNGKey(seed)
    
    # Create environment
    env = SnakeEnv(env_config)
    
    # Create network
    network = TransformerPolicy(
        d_model=64,
        num_layers=2,
        num_heads=4,
        num_actions=env_config.num_actions,
        dropout_rate=0.1
    )
    
    # Initialize network
    rng, init_rng = random.split(rng)
    obs_shape = env.observation_space_shape()
    params = network.init_params(init_rng, obs_shape)
    
    # Create optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate)
    )
    opt_state = optimizer.init(params)
    
    # Initialize environments
    rng, reset_rng = random.split(rng)
    reset_rngs = random.split(reset_rng, config.num_envs)
    env_states = jax.vmap(env.reset)(reset_rngs)
    
    # Training state
    train_state = TrainState(
        params=params,
        opt_state=opt_state,
        env_states=env_states,
        rng=rng,
        step=0
    )
    
    # Training loop
    num_updates = total_timesteps // (config.num_envs * config.num_steps)
    
    print(f"🚀 Starting GPU-native PPO training")
    print(f"   Parallel envs: {config.num_envs}")
    print(f"   Steps per rollout: {config.num_steps}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Minibatch size: {config.minibatch_size}")
    print(f"   Total updates: {num_updates}")
    print(f"   Total timesteps: {num_updates * config.num_envs * config.num_steps:,}")
    print()
    
    start_time = time.time()
    
    for update in range(num_updates):
        update_start = time.time()
        
        # Collect rollouts
        rng, rollout_rng = random.split(train_state.rng)
        
        # TODO: Implement full rollout collection and training update
        # For now, this is a skeleton - next step is to complete the training loop
        
        print(f"Update {update + 1}/{num_updates} - "
              f"Time: {time.time() - update_start:.3f}s")
    
    total_time = time.time() - start_time
    print(f"\n✅ Training complete in {total_time:.2f}s")
    print(f"   FPS: {(num_updates * config.num_envs * config.num_steps) / total_time:,.0f}")
    
    return train_state.params
