"""
Play Snake with a pretrained model

Loads a pretrained model and visualizes it playing Snake.
"""

import jax
import jax.numpy as jnp
import pickle
import argparse
import time
from pathlib import Path

from snake_jax.config import EnvConfig
from snake_jax.env import SnakeEnv
from snake_jax.network import TransformerPolicy


def render_state(state, env_config):
    """Simple ASCII visualization of the game state"""
    grid = [['.' for _ in range(env_config.width)] for _ in range(env_config.height)]
    
    # Place food
    food_x, food_y = int(state.food_pos[0]), int(state.food_pos[1])
    if 0 <= food_y < env_config.height and 0 <= food_x < env_config.width:
        grid[food_y][food_x] = '🍎'
    
    # Place snake (in reverse order so head is drawn last)
    for i in range(int(state.snake_length) - 1, -1, -1):
        x, y = int(state.snake_body[i, 0]), int(state.snake_body[i, 1])
        if 0 <= y < env_config.height and 0 <= x < env_config.width:
            if i == 0:
                grid[y][x] = '🟢'  # Head
            else:
                grid[y][x] = '🟩'  # Body
    
    # Print grid
    print('\n' + '┌' + '──' * env_config.width + '┐')
    for row in grid:
        print('│' + ''.join(f'{cell} ' for cell in row) + '│')
    print('└' + '──' * env_config.width + '┘')
    print(f'Length: {state.snake_length} | Score: {state.score}')


def play_episode(env, network, params, rng, render=True, delay=0.1):
    """Play one episode and optionally render it"""
    
    # Reset environment
    rng, reset_rng = jax.random.split(rng)
    state = env.reset(reset_rng)
    
    done = False
    total_reward = 0
    steps = 0
    apples_eaten = 0
    
    print("\n" + "="*50)
    print("Starting new episode...")
    print("="*50)
    
    while not done and steps < env.config.max_steps:
        if render:
            render_state(state, env.config)
            time.sleep(delay)
        
        # Get action from policy
        obs = env._get_observation(state)
        obs_batch = obs[None, ...]  # Add batch dimension
        
        # Get logits and sample action
        logits, _ = network.apply(params, obs_batch, training=False)
        action = jnp.argmax(logits[0])  # Greedy action
        
        # Step environment (returns: state, obs, reward, done, info)
        state, obs, reward, done, _ = env.step(state, action)
        
        total_reward += float(reward)
        steps += 1
        
        # Count apples
        if reward > 5.0:  # Apple reward is 10.0
            apples_eaten += 1
            print(f"\n🍎 Apple eaten! Total: {apples_eaten}, Score: {total_reward:.1f}")
    
    print("\n" + "="*50)
    print(f"Episode finished!")
    print(f"  Steps: {steps}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Apples Eaten: {apples_eaten}")
    print(f"  Snake Length: {state.snake_length}")
    print("="*50)
    
    return total_reward, steps, apples_eaten


def main():
    parser = argparse.ArgumentParser(description="Play Snake with pretrained model")
    parser.add_argument("--model", type=str, required=True, help="Path to model pickle file")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to play")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between steps (seconds)")
    parser.add_argument("--no-render", action="store_true", help="Don't render (just compute stats)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print("="*70)
    print("PRETRAINED MODEL EVALUATION")
    print("="*70)
    print()
    
    # Load model
    print(f"Loading model from {args.model}...")
    with open(args.model, 'rb') as f:
        data = pickle.load(f)
    
    params = data['params']
    config = data['config']
    
    print(f"Model config:")
    print(f"  d_model: {config['d_model']}")
    print(f"  num_layers: {config['num_layers']}")
    print(f"  num_heads: {config['num_heads']}")
    print(f"  use_cnn: {config.get('use_cnn', False)}")
    if config.get('use_cnn'):
        print(f"  cnn_mode: {config.get('cnn_mode', 'N/A')}")
    if 'val_accuracy' in data:
        print(f"  Validation accuracy: {data['val_accuracy']:.2%}")
    print()
    
    # Create network
    network = TransformerPolicy(
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        num_actions=4,
        dropout_rate=config.get('dropout', 0.1),
        use_cnn=config.get('use_cnn', False),
        cnn_mode=config.get('cnn_mode', 'replace'),
    )
    
    # Create environment
    env_config = EnvConfig(
        width=10,
        height=10,
        max_steps=500,
        apple_reward=10.0,
        death_penalty=-10.0,
        step_penalty=-0.01,
    )
    env = SnakeEnv(env_config)
    
    print(f"Playing {args.episodes} episodes...")
    print(f"Environment: {env_config.width}x{env_config.height}")
    print()
    
    # Play episodes
    rng = jax.random.PRNGKey(args.seed)
    
    all_rewards = []
    all_steps = []
    all_apples = []
    
    for ep in range(args.episodes):
        print(f"\n{'='*70}")
        print(f"Episode {ep + 1}/{args.episodes}")
        print(f"{'='*70}")
        
        rng, episode_rng = jax.random.split(rng)
        reward, steps, apples = play_episode(
            env, network, params, episode_rng,
            render=not args.no_render,
            delay=args.delay
        )
        
        all_rewards.append(reward)
        all_steps.append(steps)
        all_apples.append(apples)
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Episodes played: {args.episodes}")
    print(f"\nRewards:")
    print(f"  Mean: {jnp.mean(jnp.array(all_rewards)):.2f}")
    print(f"  Std:  {jnp.std(jnp.array(all_rewards)):.2f}")
    print(f"  Min:  {jnp.min(jnp.array(all_rewards)):.2f}")
    print(f"  Max:  {jnp.max(jnp.array(all_rewards)):.2f}")
    print(f"\nApples eaten:")
    print(f"  Mean: {jnp.mean(jnp.array(all_apples)):.2f}")
    print(f"  Std:  {jnp.std(jnp.array(all_apples)):.2f}")
    print(f"  Min:  {jnp.min(jnp.array(all_apples)):.0f}")
    print(f"  Max:  {jnp.max(jnp.array(all_apples)):.0f}")
    print(f"\nSteps survived:")
    print(f"  Mean: {jnp.mean(jnp.array(all_steps)):.2f}")
    print(f"  Std:  {jnp.std(jnp.array(all_steps)):.2f}")
    print(f"  Min:  {jnp.min(jnp.array(all_steps)):.0f}")
    print(f"  Max:  {jnp.max(jnp.array(all_steps)):.0f}")
    print("="*70)
    print()


if __name__ == "__main__":
    main()
