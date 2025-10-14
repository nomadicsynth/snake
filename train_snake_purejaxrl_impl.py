"""
Custom PPO training function adapted from PureJaxRL for our Snake environment

This is based on PureJaxRL's ppo.py but adapted to use our Transformer network
and Snake environment.
"""

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from typing import NamedTuple
import distrax

from snake_jax.network import TransformerPolicy


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train_custom(config, env, env_params):
    """
    Create PPO training function for Snake
    
    Args:
        config: Training configuration dict
        env: SnakeGymnaxWrapper instance
        env_params: Environment parameters
    
    Returns:
        train_fn: JIT-compiled training function
    """
    
    def linear_schedule(count):
        """Learning rate schedule"""
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac
    
    def train(rng):
        # INIT NETWORK
        network = TransformerPolicy(
            d_model=64,
            num_layers=2,
            num_heads=4,
            num_actions=env.action_space(env_params).n,
            dropout_rate=0.1
        )
        
        rng, _rng, dropout_rng = jax.random.split(rng, 3)
        init_obs = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init({'params': _rng, 'dropout': dropout_rng}, init_obs[None], training=False)
        
        # INIT OPTIMIZER
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)
        
        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state
                
                # SELECT ACTION
                rng, _rng, dropout_rng = jax.random.split(rng, 3)
                logits, value = network.apply(train_state.params, last_obs, training=True, rngs={'dropout': dropout_rng})
                pi = distrax.Categorical(logits=logits)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition
            
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            
            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs, training=False)
            
            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae
                
                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value
            
            advantages, targets = _calculate_gae(traj_batch, last_val)
            
            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minibatch(update_info, batch_info):
                    train_state, rng = update_info
                    traj_batch, advantages, targets = batch_info
                    
                    def _loss_fn(params, traj_batch, gae, targets, dropout_rng):
                        # RERUN NETWORK
                        logits, value = network.apply(params, traj_batch.obs, training=True, rngs={'dropout': dropout_rng})
                        pi = distrax.Categorical(logits=logits)
                        log_prob = pi.log_prob(traj_batch.action)
                        
                        # CALCULATE VALUE LOSS (clipped)
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        
                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()
                        
                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)
                    
                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    rng, dropout_rng = jax.random.split(rng)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets, dropout_rng
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return (train_state, rng), total_loss
                
                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                
                # Batching and shuffling
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                
                # Mini-batch updates
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                (train_state, rng), total_loss = jax.lax.scan(
                    _update_minibatch, (train_state, rng), minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss
            
            # Update for multiple epochs
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]
            
            # Print progress
            if config.get("DEBUG"):
                def callback(info):
                    if 'returned_episode_returns' in info and 'returned_episode' in info:
                        return_values = info["returned_episode_returns"][info["returned_episode"]]
                        if len(return_values) > 0:
                            print(f"  Episode return: {return_values.mean():.2f}")
                
                jax.debug.callback(callback, metric)
            
            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric
        
        # Main training scan
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}
    
    return train
