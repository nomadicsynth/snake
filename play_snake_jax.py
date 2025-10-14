#!/usr/bin/env python
"""
Watch trained Snake agent play in real-time

This script loads a trained JAX PPO model and visualizes it playing Snake.
Optionally records video of the playback.
"""

import os
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_gemm=false '
    '--xla_gpu_autotune_level=0 '
    '--xla_gpu_force_compilation_parallelism=1'
)

import jax
import jax.numpy as jnp
import pickle
import time
import argparse
from pathlib import Path
from snake_jax.config import EnvConfig
from snake_jax.gymnax_wrapper import SnakeGymnaxWrapper
from snake_jax.network import TransformerPolicy
from flax.training.train_state import TrainState
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
import numpy as np


def create_frame(snake_state, env_config):
    """Create a visual frame from the Snake game state for video recording"""
    grid = jnp.zeros((env_config.height, env_config.width), dtype=int)
    
    # Place snake body
    snake_body = snake_state.snake_body[:snake_state.snake_length]
    for i, pos in enumerate(snake_body):
        y, x = pos
        if i == 0:
            grid = grid.at[y, x].set(3)  # Head
        else:
            grid = grid.at[y, x].set(2)  # Body
    
    # Place food
    food_y, food_x = snake_state.food_pos
    grid = grid.at[food_y, food_x].set(1)
    
    return np.array(grid)


def render_state(snake_state, env_config):
    """Render the Snake game state as ASCII art"""
    grid = jnp.zeros((env_config.height, env_config.width), dtype=int)
    
    # Place snake body
    snake_body = snake_state.snake_body[:snake_state.snake_length]
    for i, pos in enumerate(snake_body):
        y, x = pos
        if i == 0:
            grid = grid.at[y, x].set(3)  # Head
        else:
            grid = grid.at[y, x].set(2)  # Body
    
    # Place food
    food_y, food_x = snake_state.food_pos
    grid = grid.at[food_y, food_x].set(1)
    
    # Print grid
    print("\n" + "=" * (env_config.width * 2 + 2))
    for row in grid:
        line = "|"
        for cell in row:
            if cell == 3:  # Head
                line += "🟢"
            elif cell == 2:  # Body
                line += "🟩"
            elif cell == 1:  # Food
                line += "🍎"
            else:
                line += "  "
        line += "|"
        print(line)
    print("=" * (env_config.width * 2 + 2))


def save_video(frames, output_path, fps=10):
    """Save frames as a video file"""
    print(f"\n🎥 Saving video to {output_path}...")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')
    
    # Create custom colormap: black, red (food), green (body), bright green (head)
    cmap = colors.ListedColormap(['black', 'red', 'green', 'lime'])
    bounds = [0, 1, 2, 3, 4]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    # Initialize image
    im = ax.imshow(frames[0], cmap=cmap, norm=norm, interpolation='nearest')
    
    def update(frame_idx):
        im.set_array(frames[frame_idx])
        ax.set_title(f'Frame {frame_idx}/{len(frames)}', fontsize=16, pad=20)
        return [im]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), 
        interval=1000/fps, blit=True, repeat=True
    )
    
    # Save as video
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, bitrate=1800)
    anim.save(output_path, writer=writer)
    
    plt.close(fig)
    print(f"✅ Video saved successfully!")
    print(f"   Total frames: {len(frames)}")
    print(f"   Duration: {len(frames)/fps:.1f}s")


def play_episode(network, params, env, env_params, rng, render_delay=0.1, record_video=False):
    """Play one episode and render it"""
    
    # Reset environment
    rng, reset_rng = jax.random.split(rng)
    obs, state = env.reset(reset_rng, env_params)
    
    total_reward = 0
    steps = 0
    max_length = 1
    
    # For video recording
    frames = [] if record_video else None
    
    print("\n🎮 Starting new episode...")
    
    while not state.snake_state.done:
        # Render current state
        if not record_video:
            render_state(state.snake_state, env.config)
            print(f"Score: {state.snake_state.score} | Length: {state.snake_state.snake_length} | Steps: {steps}")
        else:
            # Capture frame for video
            frames.append(create_frame(state.snake_state, env.config))
        
        # Get action from policy
        rng, act_rng, dropout_rng = jax.random.split(rng, 3)
        obs_batch = jnp.expand_dims(obs, axis=0)  # Add batch dimension
        logits, value = network.apply(
            params, 
            obs_batch, 
            training=False,
            rngs={'dropout': dropout_rng}
        )
        
        # Sample action
        action = jax.random.categorical(act_rng, logits[0])
        
        # Step environment
        rng, step_rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(
            step_rng, state, action, env_params
        )
        
        total_reward += reward
        steps += 1
        max_length = max(max_length, int(state.snake_state.snake_length))
        
        # Sleep for visualization (only when not recording)
        if not record_video:
            time.sleep(render_delay)
        
        if done:
            break
    
    # Final render/frame
    if not record_video:
        render_state(state.snake_state, env.config)
    else:
        frames.append(create_frame(state.snake_state, env.config))
    
    print(f"\n🎬 Episode finished!")
    print(f"   Final Score: {state.snake_state.score}")
    print(f"   Max Length: {max_length}")
    print(f"   Total Steps: {steps}")
    print(f"   Total Reward: {total_reward:.2f}")
    
    return total_reward, steps, max_length, frames


def main():
    parser = argparse.ArgumentParser(description='Watch trained Snake agent play')
    parser.add_argument('--model', type=str, default='models/snake_jax_ppo.pkl',
                        help='Path to the trained model file (default: models/snake_jax_ppo.pkl)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to play (default: 5)')
    parser.add_argument('--delay', type=float, default=0.15,
                        help='Delay between frames in seconds when not recording (default: 0.15)')
    parser.add_argument('--record', action='store_true',
                        help='Record video of the playback')
    parser.add_argument('--output', type=str, default='snake_playback.mp4',
                        help='Output video file path (default: snake_playback.mp4)')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second for video recording (default: 10)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("🐍 WATCH TRAINED SNAKE AGENT PLAY")
    print("=" * 70)
    print()
    
    # Load trained model
    model_path = args.model
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("   Please train a model first using train_snake_purejaxrl.py")
        return
    
    print(f"📂 Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    params = checkpoint['params']
    config = checkpoint['config']
    env_config = checkpoint['env_config']
    
    print(f"✅ Model loaded!")
    print(f"   Environment: {env_config.width}x{env_config.height}")
    print(f"   Device: {jax.devices()[0]}")
    print()
    
    # Create environment and network
    env = SnakeGymnaxWrapper(env_config)
    env_params = env.default_params
    
    obs_shape = env.observation_space(env_params).shape
    num_actions = env.action_space(env_params).n
    
    network = TransformerPolicy(
        num_actions=num_actions,
        d_model=64,
        num_layers=2,
        num_heads=4,
        dropout_rate=0.1
    )
    
    # Initialize RNG
    rng = jax.random.PRNGKey(0)
    
    # Play episodes
    num_episodes = args.episodes
    render_delay = args.delay
    record_video = args.record
    
    print(f"🎮 Playing {num_episodes} episodes...")
    if not record_video:
        print(f"   Render delay: {render_delay}s per frame")
    else:
        print(f"   Recording video to: {args.output}")
        print(f"   Video FPS: {args.fps}")
    print()
    
    all_rewards = []
    all_lengths = []
    all_frames = []
    
    for episode in range(num_episodes):
        print(f"\n{'='*70}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*70}")
        
        rng, episode_rng = jax.random.split(rng)
        reward, steps, max_length, frames = play_episode(
            network, params, env, env_params, episode_rng, render_delay, record_video
        )
        
        all_rewards.append(reward)
        all_lengths.append(max_length)
        if record_video:
            all_frames.extend(frames)
        
        if episode < num_episodes - 1 and not record_video:
            print("\nPress Ctrl+C to stop, or waiting 2s for next episode...")
            try:
                time.sleep(2)
            except KeyboardInterrupt:
                print("\n\n👋 Stopped by user")
                break
    
    # Save video if recording
    if record_video and all_frames:
        save_video(all_frames, args.output, fps=args.fps)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"Episodes played: {len(all_rewards)}")
    print(f"Average reward: {sum(all_rewards)/len(all_rewards):.2f}")
    print(f"Average max length: {sum(all_lengths)/len(all_lengths):.1f}")
    print(f"Best max length: {max(all_lengths)}")
    if record_video:
        print(f"Video saved to: {args.output}")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
