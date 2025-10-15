"""
Custom PPO training function adapted from PureJaxRL for our Snake environment

This is based on PureJaxRL's ppo.py but adapted to use our Transformer network
and Snake environment.
"""

import warnings

# Suppress pydantic warnings from dependencies (gymnax/brax/purejaxrl)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from typing import NamedTuple
import distrax

from snake_jax.network import TransformerPolicy

try:
    from muon_jax import chain_with_muon
    _MUON_AVAILABLE = True
except ImportError:
    _MUON_AVAILABLE = False


def make_evaluate_fn(env, env_params, num_episodes=128):
    """
    Create an evaluation function that runs the policy without exploration
    
    Args:
        env: SnakeGymnaxWrapper instance
        env_params: Environment parameters
        num_episodes: Number of episodes to run for evaluation
    
    Returns:
        evaluate_fn: JIT-compiled evaluation function
    """
    
    def evaluate(network, params, rng):
        """
        Evaluate the policy greedily (no exploration)
        
        Returns:
            Dict with evaluation metrics:
                - mean_return: Average episode return
                - mean_length: Average episode length
                - mean_score: Average game score (apples eaten)
                - max_return: Best episode return
                - max_score: Best game score
        """
        
        def run_episode(rng):
            """Run a single episode greedily"""
            # Reset environment
            rng, reset_rng = jax.random.split(rng)
            obs, state = env.reset(reset_rng, env_params)
            
            def step_fn(carry, unused):
                obs, state, rng, done = carry
                
                # Get greedy action (argmax, no sampling)
                rng, dropout_rng = jax.random.split(rng)
                logits, value = network.apply(
                    params, obs[None], 
                    training=False,
                    rngs={'dropout': dropout_rng}
                )
                action = jnp.argmax(logits[0])
                
                # Step environment
                rng, step_rng = jax.random.split(rng)
                next_obs, next_state, reward, next_done, info = env.step(
                    step_rng, state, action, env_params
                )
                
                # Only update if not already done
                obs = jnp.where(done, obs, next_obs)
                state = jax.tree_util.tree_map(
                    lambda x, y: jnp.where(done, x, y), state, next_state
                )
                done = done | next_done
                
                return (obs, state, rng, done), (reward, next_done)
            
            # Run until episode ends (max_steps)
            max_steps = env_params.max_steps if hasattr(env_params, 'max_steps') else 1000
            (_, final_state, _, _), (rewards, dones) = jax.lax.scan(
                step_fn,
                (obs, state, rng, jnp.array(False)),
                None,
                length=max_steps
            )
            
            # Calculate metrics
            episode_return = jnp.sum(rewards)
            episode_length = jnp.sum(dones)  # Count number of steps until done
            # Get final score from state - LogWrapper wraps the env_state
            final_score = final_state.env_state.snake_state.score
            
            return {
                'return': episode_return,
                'length': episode_length,
                'score': final_score,
            }
        
        # Run multiple episodes in parallel
        rng_episodes = jax.random.split(rng, num_episodes)
        episode_metrics = jax.vmap(run_episode)(rng_episodes)
        
        # Aggregate metrics
        return {
            'mean_return': jnp.mean(episode_metrics['return']),
            'std_return': jnp.std(episode_metrics['return']),
            'mean_length': jnp.mean(episode_metrics['length']),
            'mean_score': jnp.mean(episode_metrics['score']),
            'max_return': jnp.max(episode_metrics['return']),
            'max_score': jnp.max(episode_metrics['score']),
            'min_return': jnp.min(episode_metrics['return']),
        }
    
    return jax.jit(evaluate, static_argnums=(0,))


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train_step(config, env, env_params):
    """
    Create a single PPO update step function for Snake
    
    Args:
        config: Training configuration dict
        env: SnakeGymnaxWrapper instance
        env_params: Environment parameters
    
    Returns:
        init_fn: Function to initialize training state
        update_fn: JIT-compiled single update function
    """
    
    def linear_schedule(count):
        """Learning rate schedule"""
        frac = 1.0 - count / config["NUM_UPDATES"]
        return config["LR"] * frac
    
    def muon_schedule(count):
        """Muon learning rate schedule"""
        if config.get("ANNEAL_LR", False):
            frac = 1.0 - count / config["NUM_UPDATES"]
            return config.get("MUON_LR", 0.02) * frac
        else:
            return config.get("MUON_LR", 0.02)
    
    def aux_schedule(count):
        """Auxiliary (Adam) learning rate schedule for Muon"""
        if config.get("ANNEAL_LR", False):
            frac = 1.0 - count / config["NUM_UPDATES"]
            return config.get("AUX_ADAM_LR", config["LR"]) * frac
        else:
            return config.get("AUX_ADAM_LR", config["LR"])
    
    def init_train_state(rng):
        """Initialize network and training state"""
        # INIT NETWORK
        network = TransformerPolicy(
            d_model=config.get("D_MODEL", 64),
            num_layers=config.get("NUM_LAYERS", 2),
            num_heads=config.get("NUM_HEADS", 4),
            num_actions=env.action_space(env_params).n,
            dropout_rate=config.get("DROPOUT", 0.1),
            use_cnn=config.get("USE_CNN", False),
            cnn_features=config.get("CNN_FEATURES", (32, 64)),
            cnn_mode=config.get("CNN_MODE", "replace"),
        )
        
        rng, _rng, dropout_rng = jax.random.split(rng, 3)
        init_obs = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init({'params': _rng, 'dropout': dropout_rng}, init_obs[None], training=False)
        
        # INIT OPTIMIZER
        use_muon = config.get("USE_MUON", False) and _MUON_AVAILABLE
        
        if use_muon:
            momentum = config.get("MUON_MOMENTUM", 0.95)
            nesterov = config.get("MUON_NESTEROV", True)
            
            # Use schedules for both Muon and aux learning rates
            tx = chain_with_muon(
                muon_lr=muon_schedule,
                aux_lr=aux_schedule,
                max_grad_norm=config["MAX_GRAD_NORM"],
                momentum=momentum,
                nesterov=nesterov,
            )
        elif config["ANNEAL_LR"]:
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
        
        rng, _rng = jax.random.split(rng)
        return network, train_state, env_state, obsv, _rng
    
    def make_update_fn(network):
        """Create the update function"""
        
        def update_step(runner_state, update_idx):
            """Single PPO update step"""
            train_state, env_state, last_obs, rng = runner_state
            
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
                        
                        # CALCULATE VALUE LOSS
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
                    (total_loss, (value_loss, actor_loss, entropy)), grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets, dropout_rng
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return (train_state, rng), (total_loss, value_loss, actor_loss, entropy)
                
                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                
                # Batching and shuffling
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
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
                (train_state, rng), loss_info = jax.lax.scan(
                    _update_minibatch, (train_state, rng), minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, loss_info
            
            # Update for multiple epochs
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]
            
            # Combine environment metrics with loss metrics
            # loss_info is a tuple of 4 arrays: (total_loss, value_loss, actor_loss, entropy)
            # Each has shape: [num_epochs, num_minibatches]
            # Take mean across epochs and minibatches for logging
            total_loss, value_loss, actor_loss, entropy = loss_info
            
            # Calculate current learning rate
            if config.get("USE_MUON", False):
                # For Muon, calculate both LRs using schedules
                current_lr = muon_schedule(update_idx)
                current_aux_lr = aux_schedule(update_idx)
            elif config["ANNEAL_LR"]:
                current_lr = linear_schedule(update_idx)
            else:
                current_lr = config["LR"]
            
            # Create combined metrics dict
            # Note: traj_batch.info is a dict from LogWrapper, but we can't modify it in JIT
            # So we return it as-is and add loss metrics in Python land
            env_metric = traj_batch.info
            loss_metric = {
                'total_loss': jnp.mean(total_loss),
                'value_loss': jnp.mean(value_loss),
                'actor_loss': jnp.mean(actor_loss),
                'entropy': jnp.mean(entropy),
                'learning_rate': current_lr,
            }
            
            # Add aux learning rate for Muon
            if config.get("USE_MUON", False):
                loss_metric['aux_learning_rate'] = current_aux_lr
            
            runner_state = (train_state, env_state, last_obs, rng)
            # Return both env and loss metrics as a tuple
            return runner_state, (env_metric, loss_metric)
        
        return jax.jit(update_step)
    
    return init_train_state, make_update_fn


def make_train_custom(config, env, env_params):
    """
    Create PPO training function for Snake (full training loop version)
    
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
            d_model=config.get("D_MODEL", 64),
            num_layers=config.get("NUM_LAYERS", 2),
            num_heads=config.get("NUM_HEADS", 4),
            num_actions=env.action_space(env_params).n,
            dropout_rate=config.get("DROPOUT", 0.1),
            use_cnn=config.get("USE_CNN", False),
            cnn_features=config.get("CNN_FEATURES", (32, 64)),
            cnn_mode=config.get("CNN_MODE", "replace"),
        )
        
        rng, _rng, dropout_rng = jax.random.split(rng, 3)
        init_obs = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init({'params': _rng, 'dropout': dropout_rng}, init_obs[None], training=False)
        
        # INIT OPTIMIZER
        use_muon = config.get("USE_MUON", False) and _MUON_AVAILABLE
        
        if use_muon:
            # Use Muon optimizer for weight matrices, Adam for auxiliary params
            muon_lr = config.get("MUON_LR", 0.02)
            aux_lr = config.get("AUX_ADAM_LR", config["LR"])
            momentum = config.get("MUON_MOMENTUM", 0.95)
            nesterov = config.get("MUON_NESTEROV", True)
            
            tx = chain_with_muon(
                muon_lr=muon_lr,
                aux_lr=aux_lr,
                max_grad_norm=config["MAX_GRAD_NORM"],
                momentum=momentum,
                nesterov=nesterov,
            )
            print(f"  Using Muon optimizer: muon_lr={muon_lr:.2e}, aux_lr={aux_lr:.2e}, momentum={momentum}")
        elif config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        
        if config.get("USE_MUON", False) and not _MUON_AVAILABLE:
            print("  WARNING: --use-muon set but muon_jax not available. Using Adam.")
        
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
            
            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric
        
        # Main training scan
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}
    
    # JIT compile the train function
    train_jit = jax.jit(train)
    return train_jit


def make_train_with_callback(config, env, env_params, progress_callback):
    """
    Create PPO training function for Snake with progress callback
    
    Args:
        config: Training configuration dict
        env: SnakeGymnaxWrapper instance
        env_params: Environment parameters
        progress_callback: Callback function(update_idx, metrics) called after each update
    
    Returns:
        train_fn: JIT-compiled training function with callback support
    """
    
    def linear_schedule(count):
        """Learning rate schedule"""
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac
    
    def muon_schedule(count):
        """Muon learning rate schedule"""
        if config.get("ANNEAL_LR", False):
            frac = (
                1.0
                - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
                / config["NUM_UPDATES"]
            )
            return config.get("MUON_LR", 0.02) * frac
        else:
            return config.get("MUON_LR", 0.02)
    
    def aux_schedule(count):
        """Auxiliary (Adam) learning rate schedule for Muon"""
        if config.get("ANNEAL_LR", False):
            frac = (
                1.0
                - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
                / config["NUM_UPDATES"]
            )
            return config.get("AUX_ADAM_LR", config["LR"]) * frac
        else:
            return config.get("AUX_ADAM_LR", config["LR"])
    
    def train(rng):
        # INIT NETWORK
        network = TransformerPolicy(
            d_model=config.get("D_MODEL", 64),
            num_layers=config.get("NUM_LAYERS", 2),
            num_heads=config.get("NUM_HEADS", 4),
            num_actions=env.action_space(env_params).n,
            dropout_rate=config.get("DROPOUT", 0.1),
            use_cnn=config.get("USE_CNN", False),
            cnn_features=config.get("CNN_FEATURES", (32, 64)),
            cnn_mode=config.get("CNN_MODE", "replace"),
        )
        
        rng, _rng, dropout_rng = jax.random.split(rng, 3)
        init_obs = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init({'params': _rng, 'dropout': dropout_rng}, init_obs[None], training=False)
        
        # INIT OPTIMIZER
        use_muon = config.get("USE_MUON", False) and _MUON_AVAILABLE
        
        if use_muon:
            # Use Muon optimizer for weight matrices, Adam for auxiliary params
            momentum = config.get("MUON_MOMENTUM", 0.95)
            nesterov = config.get("MUON_NESTEROV", True)
            
            # Use schedules for both Muon and aux learning rates
            tx = chain_with_muon(
                muon_lr=muon_schedule,
                aux_lr=aux_schedule,
                max_grad_norm=config["MAX_GRAD_NORM"],
                momentum=momentum,
                nesterov=nesterov,
            )
        elif config["ANNEAL_LR"]:
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
        def _update_step(runner_state, update_idx):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state
                
                # SELECT ACTION
                rng, _rng, dropout_rng = jax.random.split(rng, 3)
                logits, value = network.apply(
                    train_state.params, last_obs, 
                    training=True, 
                    rngs={'dropout': dropout_rng}
                )
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
            rng, dropout_rng = jax.random.split(rng)
            _, last_val = network.apply(
                train_state.params, last_obs,
                training=False,
                rngs={'dropout': dropout_rng}
            )
            
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
                def _update_minibatch(train_state_and_rng, batch_info):
                    train_state, rng = train_state_and_rng
                    traj_batch, advantages, targets = batch_info
                    
                    def _loss_fn(params, traj_batch, gae, targets, dropout_rng):
                        # RERUN NETWORK
                        logits, value = network.apply(
                            params, traj_batch.obs,
                            training=True,
                            rngs={'dropout': dropout_rng}
                        )
                        pi = distrax.Categorical(logits=logits)
                        log_prob = pi.log_prob(traj_batch.action)
                        
                        # CALCULATE VALUE LOSS
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
                    (total_loss, (value_loss, actor_loss, entropy)), grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets, dropout_rng
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return (train_state, rng), (total_loss, value_loss, actor_loss, entropy)
                
                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                
                # Batching and shuffling
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
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
                (train_state, rng), loss_info = jax.lax.scan(
                    _update_minibatch, (train_state, rng), minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, loss_info
            
            # Update for multiple epochs
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]
            
            # Combine environment metrics with loss metrics
            # loss_info is a tuple of 4 arrays: (total_loss, value_loss, actor_loss, entropy)
            # Each has shape: [num_epochs, num_minibatches]
            # Take mean across epochs and minibatches for logging
            total_loss, value_loss, actor_loss, entropy = loss_info
            
            # Calculate current learning rate
            if config.get("USE_MUON", False):
                # For Muon, calculate both LRs using schedules
                current_lr = muon_schedule(update_idx)
                current_aux_lr = aux_schedule(update_idx)
            elif config.get("ANNEAL_LR", False):
                current_lr = linear_schedule(update_idx)
            else:
                current_lr = config["LR"]
            
            # Return both environment info and loss metrics separately
            env_metric = traj_batch.info  # Dict from LogWrapper
            loss_metric = {
                'total_loss': jnp.mean(total_loss),
                'value_loss': jnp.mean(value_loss),
                'actor_loss': jnp.mean(actor_loss),
                'entropy': jnp.mean(entropy),
                'learning_rate': current_lr,
            }
            
            # Add aux learning rate for Muon
            if config.get("USE_MUON", False):
                loss_metric['aux_learning_rate'] = current_aux_lr
            
            # Call progress callback using io_callback
            # Extract only the metrics dict to pass to callback
            def _call_callback(idx, env_dict, loss_dict):
                progress_callback(int(idx), {**env_dict, **loss_dict})
            
            jax.experimental.io_callback(
                _call_callback,
                None,  # No return value
                update_idx,
                env_metric,
                loss_metric,
                ordered=True  # Ensure callbacks happen in order
            )
            
            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, (env_metric, loss_metric)
        
        # Main training scan
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["NUM_UPDATES"])
        )
        return {"runner_state": runner_state, "metrics": metric}
    
    # JIT compile the train function
    train_jit = jax.jit(train)
    return train_jit
