"""
Train Snake using PureJaxRL's PPO implementation

This integrates our GPU-native Snake environment with PureJaxRL's 
battle-tested PPO algorithm with wandb tracking and progress bars.
"""

# Set XLA flags to avoid ptxas compilation errors
import os
import warnings

# Disable Triton GEMM and use more conservative compilation
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_gemm=false '
    '--xla_gpu_autotune_level=0 '
    '--xla_gpu_force_compilation_parallelism=1'
)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'

# Suppress pydantic warnings from dependencies (gymnax/brax/purejaxrl)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")

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
from purejaxrl.purejaxrl.wrappers import LogWrapper


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Snake with JAX PPO")
    
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
    
    # Muon optimizer args
    parser.add_argument("--use-muon", action="store_true", default=False, help="Use Muon optimizer for weight matrices")
    parser.add_argument("--muon-lr", type=float, default=0.02, help="Learning rate for Muon (weight matrices)")
    parser.add_argument("--aux-adam-lr", type=float, default=None, help="Learning rate for Adam (aux params, defaults to --lr)")
    parser.add_argument("--muon-momentum", type=float, default=0.95, help="Momentum for Muon optimizer")
    parser.add_argument("--muon-nesterov", action="store_true", default=True, help="Use Nesterov momentum in Muon")
    
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
    parser.add_argument("--save-freq", type=int, default=100, help="Save model every N updates")
    
    # Misc args
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("🚀 GPU-NATIVE SNAKE TRAINING WITH PUREJAXRL PPO")
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
    
    # Training hyperparameters (dict format for PureJaxRL)
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
        # Muon optimizer
        "USE_MUON": args.use_muon,
        "MUON_LR": args.muon_lr,
        "AUX_ADAM_LR": args.aux_adam_lr if args.aux_adam_lr is not None else args.lr,
        "MUON_MOMENTUM": args.muon_momentum,
        "MUON_NESTEROV": args.muon_nesterov,
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
                **env_config._asdict(),
                "seed": args.seed,
            },
            tags=["jax", "ppo", "snake", "gpu-native"],
            sync_tensorboard=False,  # We'll log manually
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
    if config['USE_MUON']:
        print(f"  Optimizer: Muon")
        print(f"    Muon LR (weights): {config['MUON_LR']:.2e}")
        print(f"    Adam LR (aux): {config['AUX_ADAM_LR']:.2e}")
        print(f"    Momentum: {config['MUON_MOMENTUM']}")
    else:
        print(f"  Optimizer: Adam")
        print(f"  Learning rate: {config['LR']:.2e}")
    print(f"  Network: d_model={args.d_model}, layers={args.num_layers}, heads={args.num_heads}")
    print()
    
    # Create environment with logging wrapper
    base_env = SnakeGymnaxWrapper(env_config)
    env = LogWrapper(base_env)  # Wrap with LogWrapper to track episode returns
    env_params = env.default_params
    
    print(f"Environment:")
    print(f"  Observation shape: {env.observation_space(env_params).shape}")
    print(f"  Action space: {env.action_space(env_params).n}")
    print()
    
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
    
    # Variables to track training state
    result = None
    runner_state = None
    network = None
    interrupted = False
    train_time = 0
    
    try:
        # Training loop with progress bar and metrics tracking
        print("🚀 Training...")
        train_start = time.time()
        
        # Import the step-by-step training function
        from train_snake_purejaxrl_impl import make_train_step
        init_fn, make_update_fn = make_train_step(config, env, env_params)
        
        # Initialize training state
        print("   Initializing network and environment...")
        network, train_state, env_state, obsv, rng = init_fn(rng)
        runner_state = (train_state, env_state, obsv, rng)
        
        # Create JIT-compiled update function
        update_fn = make_update_fn(network)
        
        print("   Compiling update function (first call will be slow)...")
        print()
        
        # Run training loop with progress bar
        all_metrics = []
        with tqdm(total=config['TOTAL_TIMESTEPS'], 
                  desc="Training", 
                  unit="steps",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                  colour='green',
                  dynamic_ncols=True) as pbar:
            
            for update_idx in range(config['NUM_UPDATES']):
                runner_state, metrics = update_fn(runner_state, update_idx)
                
                # Transfer metrics from GPU to CPU for logging
                metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
                all_metrics.append(metrics)
                
                # Update progress bar
                steps_completed = config['NUM_STEPS'] * config['NUM_ENVS']
                pbar.update(steps_completed)
                
                # Log to wandb periodically
                if args.wandb and update_idx % 10 == 0:
                    # Check if metrics has the expected structure
                    if 'returned_episode_returns' in metrics and 'returned_episode' in metrics:
                        # With LogWrapper, returned_episode is a boolean mask indicating which envs completed
                        # and returned_episode_returns contains the cumulative returns for completed episodes
                        # Shape: [num_steps, num_envs]
                        completed_mask = metrics['returned_episode']  # Boolean [num_steps, num_envs]
                        episode_returns = metrics['returned_episode_returns']  # [num_steps, num_envs]
                        
                        # Get returns for completed episodes only
                        # Filter where completed_mask is True
                        if jnp.any(completed_mask):
                            valid_returns = episode_returns[completed_mask]
                            mean_return = float(jnp.mean(valid_returns))
                            max_return = float(jnp.max(valid_returns))
                            min_return = float(jnp.min(valid_returns))
                            num_episodes = int(jnp.sum(completed_mask))
                            
                            wandb.log({
                                "train/mean_return": mean_return,
                                "train/max_return": max_return,
                                "train/min_return": min_return,
                                "train/num_episodes": num_episodes,
                                "train/update": update_idx,
                                "train/timesteps": (update_idx + 1) * steps_completed,
                            })
        
        # Combine results
        result = {
            'runner_state': runner_state,
            'metrics': jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *all_metrics)
        }
        
        train_time = time.time() - train_start
        
        print()
        print("=" * 70)
        print("✅ TRAINING COMPLETE!")
        print("=" * 70)
        print()
        print(f"⏱️  Timing:")
        print(f"  Training: {train_time:.2f}s")
        print()
        print(f"📊 Performance:")
        print(f"  Total steps: {config['TOTAL_TIMESTEPS']:,}")
        print(f"  Training FPS: {config['TOTAL_TIMESTEPS'] / train_time:,.0f}")
        print(f"  Time per update: {train_time / config['NUM_UPDATES']:.3f}s")
        print()
    
    except KeyboardInterrupt:
        train_time = time.time() - train_start
        interrupted = True
        print()
        print()
        print("=" * 70)
        print("⚠️  TRAINING INTERRUPTED BY USER")
        print("=" * 70)
        print()
        print(f"⏱️  Training time before interrupt: {train_time:.2f}s")
        print()
    
    finally:
        # Extract and log metrics if we have results
        metrics = {}
        if result is not None:
            metrics = result.get('metrics', {})
        
            # Calculate mean episode returns if available
            if isinstance(metrics, dict) and 'returned_episode_returns' in metrics:
                returns = metrics['returned_episode_returns']
                episodes = metrics.get('returned_episode', None)
                
                if episodes is not None:
                    # Filter to only episodes that actually happened
                    valid_mask = episodes > 0
                    if jnp.any(valid_mask):
                        valid_returns = returns[valid_mask]
                        mean_return = float(jnp.mean(valid_returns))
                        max_return = float(jnp.max(valid_returns))
                        min_return = float(jnp.min(valid_returns))
                        
                        print(f"📈 Episode Statistics:")
                        print(f"  Mean return: {mean_return:.2f}")
                        print(f"  Max return: {max_return:.2f}")
                        print(f"  Min return: {min_return:.2f}")
                        print()
                        
                        if args.wandb:
                            wandb.log({
                                "final/mean_return": mean_return,
                                "final/max_return": max_return,
                                "final/min_return": min_return,
                            })
            
            # Log timing metrics to wandb
            if args.wandb and train_time > 0:
                wandb.log({
                    "timing/train_time": train_time,
                    "timing/training_fps": config['TOTAL_TIMESTEPS'] / train_time,
                    "interrupted": interrupted,
                })
        
        if not interrupted:
            print("🎉 GPU-native training complete!")
            print()
        
        # Save the trained model (even if interrupted)
        # Check if we have a runner_state (which contains the train_state with params)
        if runner_state is not None:
            try:
                model_filename = "interrupted_model.pkl" if interrupted else "final_model.pkl"
                print(f"💾 Saving model to {model_filename}...")
                model_path = run_dir / model_filename
                
                # Extract train_state from runner_state
                # runner_state is (train_state, env_state, obs, rng)
                final_train_state = runner_state[0]
                
                # Save params and config
                with open(model_path, 'wb') as f:
                    pickle.dump({
                        'params': final_train_state.params,
                        'config': config,
                        'env_config': env_config,
                        'run_name': run_name,
                        'metrics': metrics,
                        'interrupted': interrupted,
                        'train_time': train_time,
                    }, f)
                
                print(f"💾 Model saved to: {model_path}")
                
                if args.wandb:
                    # Save model as wandb artifact
                    artifact_name = f"model-{run_name}-interrupted" if interrupted else f"model-{run_name}"
                    artifact = wandb.Artifact(artifact_name, type="model")
                    artifact.add_file(str(model_path))
                    wandb.log_artifact(artifact)
                    print(f"📦 Model uploaded to WandB")
            
            except Exception as e:
                print(f"⚠️  Error saving model: {e}")
                print("   Training results may be lost.")
        else:
            print("⚠️  No model to save (training did not start or failed early)")
        
        # Clean up wandb
        if args.wandb:
            try:
                wandb.finish()
            except Exception:
                pass
        
        print()


if __name__ == "__main__":
    main()
