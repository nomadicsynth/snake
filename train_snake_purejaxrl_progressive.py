"""
Progressive training with callbacks for Snake using JAX PPO

This version provides updates during training for progress bars and wandb logging.
"""

# Set XLA flags to avoid ptxas compilation errors
import os
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
import argparse
from datetime import datetime
from pathlib import Path

from snake_jax.config import EnvConfig
from snake_jax.gymnax_wrapper import SnakeGymnaxWrapper
from snake_jax.network import TransformerPolicy
import time
from tqdm.auto import tqdm
import wandb
import pickle
from typing import Optional, Dict, Any
import optax
from flax.training.train_state import TrainState
import distrax


class ProgressiveTrainer:
    """Trainer that provides progress updates during training"""
    
    def __init__(self, config: Dict[str, Any], env, env_params, use_wandb: bool = False):
        self.config = config
        self.env = env
        self.env_params = env_params
        self.use_wandb = use_wandb
        
        # Create network
        self.network = TransformerPolicy(
            d_model=config.get("D_MODEL", 64),
            num_layers=config.get("NUM_LAYERS", 2),
            num_heads=config.get("NUM_HEADS", 4),
            num_actions=env.action_space(env_params).n,
            dropout_rate=config.get("DROPOUT", 0.1)
        )
        
        # JIT compile update function
        self._compile_functions()
    
    def _compile_functions(self):
        """JIT compile the training update function"""
        
        def linear_schedule(count):
            """Learning rate schedule"""
            frac = (
                1.0
                - (count // (self.config["NUM_MINIBATCHES"] * self.config["UPDATE_EPOCHS"]))
                / self.config["NUM_UPDATES"]
            )
            return self.config["LR"] * frac
        
        def single_update(train_state, env_state, obs, rng):
            """Single PPO update step"""
            
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state_inner, env_state_inner, last_obs, rng_inner = runner_state
                
                # SELECT ACTION
                rng_inner, _rng, dropout_rng = jax.random.split(rng_inner, 3)
                logits, value = self.network.apply(
                    train_state_inner.params, last_obs, training=True, rngs={'dropout': dropout_rng}
                )
                pi = distrax.Categorical(logits=logits)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                
                # STEP ENV
                rng_inner, _rng = jax.random.split(rng_inner)
                rng_step = jax.random.split(_rng, self.config["NUM_ENVS"])
                obsv, env_state_inner, reward, done, info = jax.vmap(
                    self.env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state_inner, action, self.env_params)
                
                from train_snake_purejaxrl_impl import Transition
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state_inner, env_state_inner, obsv, rng_inner)
                return runner_state, transition
            
            runner_state = (train_state, env_state, obs, rng)
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, self.config["NUM_STEPS"]
            )
            
            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = self.network.apply(train_state.params, last_obs, training=False)
            
            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + self.config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + self.config["GAMMA"] * self.config["GAE_LAMBDA"] * (1 - done) * gae
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
                    train_state_mb, rng_mb = update_info
                    traj_batch_mb, advantages_mb, targets_mb = batch_info
                    
                    def _loss_fn(params, traj_batch, gae, targets, dropout_rng):
                        # RERUN NETWORK
                        logits, value = self.network.apply(
                            params, traj_batch.obs, training=True, rngs={'dropout': dropout_rng}
                        )
                        pi = distrax.Categorical(logits=logits)
                        log_prob = pi.log_prob(traj_batch.action)
                        
                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-self.config["CLIP_EPS"], self.config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        
                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(
                            ratio, 1.0 - self.config["CLIP_EPS"], 1.0 + self.config["CLIP_EPS"]
                        ) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()
                        
                        total_loss = (
                            loss_actor
                            + self.config["VF_COEF"] * value_loss
                            - self.config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)
                    
                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    rng_mb, dropout_rng = jax.random.split(rng_mb)
                    (total_loss, (value_loss, actor_loss, entropy)), grads = grad_fn(
                        train_state_mb.params, traj_batch_mb, advantages_mb, targets_mb, dropout_rng
                    )
                    train_state_mb = train_state_mb.apply_gradients(grads=grads)
                    
                    loss_dict = {
                        'total_loss': total_loss,
                        'value_loss': value_loss,
                        'actor_loss': actor_loss,
                        'entropy': entropy,
                    }
                    
                    return (train_state_mb, rng_mb), loss_dict
                
                train_state_ep, traj_batch, advantages, targets, rng_ep = update_state
                rng_ep, _rng = jax.random.split(rng_ep)
                
                # Batching and shuffling
                batch_size = self.config["MINIBATCH_SIZE"] * self.config["NUM_MINIBATCHES"]
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
                        x, [self.config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                (train_state_ep, rng_ep), loss_info = jax.lax.scan(
                    _update_minibatch, (train_state_ep, rng_ep), minibatches
                )
                update_state = (train_state_ep, traj_batch, advantages, targets, rng_ep)
                return update_state, loss_info
            
            # Update for multiple epochs
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, self.config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]
            
            # Extract metrics
            metric = {
                'info': traj_batch.info,
                'rewards': traj_batch.reward,
                'loss': loss_info,
            }
            
            return train_state, env_state, last_obs, rng, metric
        
        # JIT compile
        self._update_fn = jax.jit(single_update)
    
    def train(self, rng, pbar: Optional[tqdm] = None):
        """Train with progress updates"""
        
        # Initialize network
        rng, _rng, dropout_rng = jax.random.split(rng, 3)
        init_obs = jnp.zeros(self.env.observation_space(self.env_params).shape)
        network_params = self.network.init(
            {'params': _rng, 'dropout': dropout_rng}, init_obs[None], training=False
        )
        
        # Initialize optimizer
        if self.config["ANNEAL_LR"]:
            def linear_schedule(count):
                frac = (
                    1.0 - (count // (self.config["NUM_MINIBATCHES"] * self.config["UPDATE_EPOCHS"]))
                    / self.config["NUM_UPDATES"]
                )
                return self.config["LR"] * frac
            
            tx = optax.chain(
                optax.clip_by_global_norm(self.config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(self.config["MAX_GRAD_NORM"]),
                optax.adam(self.config["LR"], eps=1e-5),
            )
        
        train_state = TrainState.create(
            apply_fn=self.network.apply,
            params=network_params,
            tx=tx,
        )
        
        # Initialize environment
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, self.config["NUM_ENVS"])
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(
            reset_rng, self.env_params
        )
        
        # Training loop
        for update_idx in range(self.config["NUM_UPDATES"]):
            rng, _rng = jax.random.split(rng)
            train_state, env_state, obs, rng, metrics = self._update_fn(
                train_state, env_state, obs, _rng
            )
            
            # Calculate steps completed
            steps = (update_idx + 1) * self.config["NUM_ENVS"] * self.config["NUM_STEPS"]
            
            # Update progress bar
            if pbar is not None:
                pbar.update(self.config["NUM_ENVS"] * self.config["NUM_STEPS"])
                
                # Extract episode info if available
                info = metrics.get('info', {})
                if 'returned_episode_returns' in info:
                    episode_mask = info.get('returned_episode', jnp.zeros_like(info['returned_episode_returns'])) > 0
                    if jnp.any(episode_mask):
                        returns = info['returned_episode_returns'][episode_mask]
                        mean_return = float(jnp.mean(returns))
                        pbar.set_postfix({
                            'return': f'{mean_return:.2f}',
                            'update': f'{update_idx + 1}/{self.config["NUM_UPDATES"]}'
                        })
            
            # Log to wandb periodically
            if self.use_wandb and (update_idx % 10 == 0 or update_idx == self.config["NUM_UPDATES"] - 1):
                log_dict = {'update': update_idx, 'steps': steps}
                
                # Add loss metrics (take mean over epochs and minibatches)
                loss_metrics = metrics.get('loss', {})
                for key, val in loss_metrics.items():
                    if isinstance(val, jnp.ndarray):
                        log_dict[f'train/{key}'] = float(jnp.mean(val))
                
                # Add episode metrics if available
                info = metrics.get('info', {})
                if 'returned_episode_returns' in info:
                    episode_mask = info.get('returned_episode', jnp.zeros_like(info['returned_episode_returns'])) > 0
                    if jnp.any(episode_mask):
                        returns = info['returned_episode_returns'][episode_mask]
                        lengths = info.get('returned_episode_lengths', jnp.zeros_like(info['returned_episode_returns']))[episode_mask]
                        
                        log_dict.update({
                            'episode/mean_return': float(jnp.mean(returns)),
                            'episode/max_return': float(jnp.max(returns)),
                            'episode/min_return': float(jnp.min(returns)),
                            'episode/mean_length': float(jnp.mean(lengths)) if len(lengths) > 0 else 0,
                        })
                
                # Add reward statistics
                rewards = metrics.get('rewards', None)
                if rewards is not None:
                    log_dict.update({
                        'train/mean_reward': float(jnp.mean(rewards)),
                        'train/max_reward': float(jnp.max(rewards)),
                    })
                
                wandb.log(log_dict, step=steps)
        
        return train_state, env_state, obs, rng


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Snake with JAX PPO (Progressive)")
    
    # Environment args
    parser.add_argument("--width", type=int, default=10, help="Board width")
    parser.add_argument("--height", type=int, default=10, help="Board height")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--apple-reward", type=float, default=10.0, help="Reward for eating apple")
    parser.add_argument("--death-penalty", type=float, default=-10.0, help="Penalty for dying")
    parser.add_argument("--step-penalty", type=float, default=-0.01, help="Penalty per step")
    
    # Training args
    parser.add_argument("--num-envs", type=int, default=2048, help="Number of parallel environments")
    parser.add_argument("--num-steps", type=int, default=128, help="Steps per rollout")
    parser.add_argument("--total-timesteps", type=int, default=5_000_000, help="Total training timesteps")
    parser.add_argument("--update-epochs", type=int, default=4, help="PPO update epochs")
    parser.add_argument("--num-minibatches", type=int, default=32, help="Number of minibatches")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max gradient norm")
    parser.add_argument("--anneal-lr", action="store_true", default=True, help="Use learning rate annealing")
    
    # Network args
    parser.add_argument("--d-model", type=int, default=64, help="Transformer model dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Logging args
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="snake-jax-ppo", help="WandB project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="WandB entity")
    parser.add_argument("--run-name", type=str, default=None, help="Run name")
    parser.add_argument("--save-dir", type=str, default="models", help="Directory to save models")
    
    # Misc args
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("🚀 GPU-NATIVE SNAKE TRAINING WITH PROGRESSIVE PPO")
    print("=" * 70)
    print()
    
    # Environment configuration
    env_config = EnvConfig(
        width=args.width,
        height=args.height,
        max_steps=args.max_steps,
        apple_reward=args.apple_reward,
        death_penalty=args.death_penalty,
        step_penalty=args.step_penalty
    )
    
    # Training hyperparameters
    config = {
        "NUM_ENVS": args.num_envs,
        "NUM_STEPS": args.num_steps,
        "TOTAL_TIMESTEPS": args.total_timesteps,
        "UPDATE_EPOCHS": args.update_epochs,
        "NUM_MINIBATCHES": args.num_minibatches,
        "GAMMA": args.gamma,
        "GAE_LAMBDA": args.gae_lambda,
        "CLIP_EPS": args.clip_eps,
        "ENT_COEF": args.ent_coef,
        "VF_COEF": args.vf_coef,
        "MAX_GRAD_NORM": args.max_grad_norm,
        "LR": args.lr,
        "ANNEAL_LR": args.anneal_lr,
        "D_MODEL": args.d_model,
        "NUM_LAYERS": args.num_layers,
        "NUM_HEADS": args.num_heads,
        "DROPOUT": args.dropout,
    }
    
    # Derived values
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    
    # Generate run name
    if args.run_name:
        run_name = args.run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"snake_jax_{args.width}x{args.height}_{timestamp}"
    
    # Initialize wandb
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                **config,
                **vars(env_config),
                "seed": args.seed,
            },
            tags=["jax", "ppo", "snake", "gpu-native", "progressive"],
        )
        print(f"📊 WandB initialized: {wandb.run.url}")
        print()
    
    print(f"Configuration:")
    print(f"  Run name: {run_name}")
    print(f"  Environment: {env_config.width}x{env_config.height} Snake")
    print(f"  Device: {jax.devices()[0]}")
    print(f"  Parallel envs: {config['NUM_ENVS']}")
    print(f"  Steps per rollout: {config['NUM_STEPS']}")
    print(f"  Total timesteps: {config['TOTAL_TIMESTEPS']:,}")
    print(f"  Number of updates: {config['NUM_UPDATES']:,}")
    print(f"  Minibatch size: {config['MINIBATCH_SIZE']}")
    print(f"  Learning rate: {config['LR']:.2e}")
    print(f"  Network: d_model={args.d_model}, layers={args.num_layers}, heads={args.num_heads}")
    print()
    
    # Create environment
    env = SnakeGymnaxWrapper(env_config)
    env_params = env.default_params
    
    print(f"Environment:")
    print(f"  Observation shape: {env.observation_space(env_params).shape}")
    print(f"  Action space: {env.action_space(env_params).n}")
    print()
    
    # Create trainer
    trainer = ProgressiveTrainer(config, env, env_params, use_wandb=args.wandb)
    
    # Initialize RNG
    rng = jax.random.PRNGKey(args.seed)
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    run_dir = save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("Starting training...")
    print("=" * 70)
    print()
    
    start_time = time.time()
    
    # Train with progress bar
    with tqdm(
        total=config['TOTAL_TIMESTEPS'],
        desc="🐍 Training Snake",
        unit="steps",
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
        colour='green',
        dynamic_ncols=True,
        ascii=False,
    ) as pbar:
        train_state, env_state, obs, rng = trainer.train(rng, pbar=pbar)
    
    total_time = time.time() - start_time
    
    print()
    print("=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print()
    print(f"⏱️  Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print()
    print(f"📊 Performance:")
    print(f"  Total steps: {config['TOTAL_TIMESTEPS']:,}")
    print(f"  Training FPS: {config['TOTAL_TIMESTEPS'] / total_time:,.0f}")
    print(f"  Time per update: {total_time / config['NUM_UPDATES']:.3f}s")
    print()
    
    # Log timing metrics to wandb
    if args.wandb:
        wandb.log({
            "final/total_time": total_time,
            "final/training_fps": config['TOTAL_TIMESTEPS'] / total_time,
            "final/time_per_update": total_time / config['NUM_UPDATES'],
        })
    
    print("🎉 GPU-native training complete!")
    print(f"   Compare to SB3: {total_time:.1f}s vs ~30-50 minutes")
    print(f"   Speedup: ~{(30*60)/total_time:.0f}x faster!")
    print()
    
    # Save the trained model
    model_path = run_dir / "final_model.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump({
            'params': train_state.params,
            'config': config,
            'env_config': env_config,
            'run_name': run_name,
        }, f)
    
    print(f"💾 Model saved to: {model_path}")
    
    if args.wandb:
        # Save model as wandb artifact
        artifact = wandb.Artifact(f"model-{run_name}", type="model")
        artifact.add_file(str(model_path))
        wandb.log_artifact(artifact)
        print(f"📦 Model uploaded to WandB")
        
        wandb.finish()
    
    print()


if __name__ == "__main__":
    main()
