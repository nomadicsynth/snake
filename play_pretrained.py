"""
Play Snake with a pretrained model

Loads a pretrained model and visualizes it playing Snake.
Supports both standard and RSM (Reasoning Snake Model) modes.
"""

import jax
import jax.numpy as jnp
import pickle
import argparse
import time
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
import imageio

from environments.snake_jax.config import EnvConfig
from environments.snake_jax.env import SnakeEnv
from environments.snake_jax.network import TransformerPolicy
from pretrain_utils import get_positions_from_state
from reasoning_dsl import generate_reasoning_text, reasoning_to_embeddings


def generate_reasoning_autoregressive(
    network, params, obs, max_tokens=128, temperature=0.0
):
    """
    Autoregressively generate reasoning tokens.
    
    The model generates tokens one by one, attending to grid + previous reasoning tokens.
    Generation stops when an action token (0-3) is predicted.
    
    Args:
        network: TransformerPolicy network
        params: Model parameters
        obs: Grid observation (height, width, 3)
        max_tokens: Maximum reasoning tokens to generate
        temperature: Sampling temperature (0.0 = greedy)
    
    Returns:
        reasoning_tokens: Generated reasoning token IDs
        action: Final predicted action (0-3)
        reasoning_text: Decoded reasoning string
    """
    obs_batch = obs[None, ...]  # Add batch dimension
    
    # Start with empty reasoning
    reasoning_tokens = jnp.zeros((1, 0), dtype=jnp.int32)
    
    # Special token markers
    ACTION_OFFSET = 4  # Actions are 0-3, reasoning tokens start at 4
    
    generated_tokens = []
    
    for t in range(max_tokens):
        # Forward pass with current reasoning tokens
        logits, _ = network.apply(
            params, obs_batch, 
            training=False, 
            reasoning_tokens=reasoning_tokens if reasoning_tokens.shape[1] > 0 else None
        )
        # logits shape: (1, num_actions + vocab_size) = (1, 4 + 128) = (1, 132)
        
        # Sample/pick next token
        if temperature == 0.0:
            next_token_id = jnp.argmax(logits[0])
        else:
            # Temperature sampling
            probs = jax.nn.softmax(logits[0] / temperature)
            rng = jax.random.PRNGKey(t)  # Simple approach; better to pass RNG
            next_token_id = jax.random.categorical(rng, jnp.log(probs))
        
        next_token_id = int(next_token_id)
        
        # Check if it's an action token (0-3)
        if next_token_id < 4:
            # Model decided to predict action - stop reasoning
            action = next_token_id
            break
        
        # It's a reasoning token - add to sequence
        reasoning_token = next_token_id - ACTION_OFFSET  # Convert back to ASCII range
        generated_tokens.append(reasoning_token)
        
        # Append to reasoning sequence
        new_token = jnp.array([[reasoning_token]], dtype=jnp.int32)
        if reasoning_tokens.shape[1] == 0:
            reasoning_tokens = new_token
        else:
            reasoning_tokens = jnp.concatenate([reasoning_tokens, new_token], axis=1)
    else:
        # Max tokens reached - force action prediction
        # Do one more forward pass to get action
        logits, _ = network.apply(
            params, obs_batch, 
            training=False, 
            reasoning_tokens=reasoning_tokens if reasoning_tokens.shape[1] > 0 else None
        )
        action = int(jnp.argmax(logits[0, :4]))  # Only consider action logits
    
    # Decode reasoning text
    if generated_tokens:
        reasoning_text = ''.join(chr(min(max(t, 0), 127)) for t in generated_tokens)
    else:
        reasoning_text = ""
    
    return reasoning_tokens, action, reasoning_text


def render_state(state, env_config, mode="graphical", return_frame=False, moves=None):
    """Render the game state in ASCII or graphical mode (default: graphical)
    If return_frame is True, returns the RGB numpy array for video saving."""
    if mode == "ascii":
        grid = [['  ' for _ in range(env_config.width)] for _ in range(env_config.height)]
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
        print("\n" + "┌" + "───" * env_config.width + "┐")
        for row in grid:
            print('│' + ''.join(f'{cell} ' for cell in row) + '│')
        print("└" + "───" * env_config.width + "┘")
        print(f'Length: {state.snake_length} | Score: {state.score}')
        return None
    else:
        # Graphical rendering using matplotlib
        grid = np.zeros((env_config.height, env_config.width, 3), dtype=np.float32)
        # Background
        grid[:] = [0.9, 0.9, 0.9]
        # Food
        food_x, food_y = int(state.food_pos[0]), int(state.food_pos[1])
        if 0 <= food_y < env_config.height and 0 <= food_x < env_config.width:
            grid[food_y, food_x] = [1.0, 0.0, 0.0]  # Red
        # Snake body
        for i in range(int(state.snake_length) - 1, -1, -1):
            x, y = int(state.snake_body[i, 0]), int(state.snake_body[i, 1])
            if 0 <= y < env_config.height and 0 <= x < env_config.width:
                if i == 0:
                    grid[y, x] = [0.0, 0.8, 0.0]  # Head: green
                else:
                    grid[y, x] = [0.0, 0.5, 0.0]  # Body: darker green
        plt.clf()
        plt.imshow(grid, interpolation='nearest')
        title_str = f"Length: {state.snake_length} | Score: {state.score}"
        if moves is not None:
            title_str += f" | Moves: {moves}"
        plt.title(title_str)
        plt.axis('off')
        plt.pause(0.001)
        if return_frame:
            # Use FigureCanvasAgg for off-screen rendering
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            fig = plt.gcf()
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            width, height = fig.get_size_inches() * fig.get_dpi()
            width, height = int(width), int(height)
            buf = canvas.buffer_rgba()
            frame = np.asarray(buf)[:, :, :3].copy()  # Drop alpha channel
            return frame
        return None


def play_episode(env, network, params, rng, render=True, delay=0.1, render_mode="graphical", 
                 save_video=False, video_path=None, use_reasoning=False, show_reasoning=False):
    """Play one episode and optionally render it and save video"""
    # Reset environment
    rng, reset_rng = jax.random.split(rng)
    state = env.reset(reset_rng)
    done = False
    total_reward = 0
    steps = 0
    apples_eaten = 0
    frames = []
    fig = None
    if render and render_mode == "graphical":
        plt.ion()
        fig = plt.figure(figsize=(6, 6), dpi=100)
    print("\n" + "="*50)
    print("Starting new episode...")
    if use_reasoning:
        print("🧠 RSM mode active - generating reasoning...")
    print("="*50)
    while not done and steps < env.config.max_steps:
        frame = None
        if render:
            frame = render_state(state, env.config, mode=render_mode, return_frame=save_video, moves=steps)
            time.sleep(delay)
        elif save_video:
            frame = render_state(state, env.config, mode=render_mode, return_frame=True, moves=steps)
        if save_video and frame is not None:
            frames.append(frame)
        
        # Get action from policy
        obs = env._get_observation(state)
        
        # Use autoregressive reasoning generation for RSM models
        if use_reasoning:
            # Generate reasoning autoregressively
            reasoning_tokens, action, reasoning_text = generate_reasoning_autoregressive(
                network, params, obs, max_tokens=128, temperature=0.0
            )
            
            if show_reasoning:
                print(f"  🧠 Generated reasoning: {reasoning_text}")
                print(f"     Action: {['UP', 'RIGHT', 'DOWN', 'LEFT'][action]}")
        else:
            # Standard forward pass without reasoning
            obs_batch = obs[None, ...]  # Add batch dimension
            logits, _ = network.apply(params, obs_batch, training=False)
            action = int(jnp.argmax(logits[0, :4]))  # Extract action from logits
        
        # Step environment (returns: state, obs, reward, done, info)
        state, obs, reward, done, _ = env.step(state, action)
        total_reward += float(reward)
        steps += 1
        # Count apples
        if reward > 5.0:  # Apple reward is 10.0
            apples_eaten += 1
            print(f"\n🍎 Apple eaten! Total: {apples_eaten}, Score: {total_reward:.1f}")
    if render and render_mode == "graphical":
        plt.ioff()
        plt.close()
    if save_video and frames and video_path:
        print(f"Saving video to {video_path}...")
        # Ensure all frames are the same size and divisible by macro_block_size=16
        import cv2
        target_shape = frames[0].shape
        # Make width and height divisible by 16
        def pad_to_16(x):
            return ((x + 15) // 16) * 16
        padded_height = pad_to_16(target_shape[0])
        padded_width = pad_to_16(target_shape[1])
        padded_shape = (padded_height, padded_width, target_shape[2])
        def pad_frame(frame):
            h, w, c = frame.shape
            pad_h = padded_height - h
            pad_w = padded_width - w
            return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=255)
        resized_frames = [cv2.resize(frame, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA) if frame.shape != target_shape else frame for frame in frames]
        padded_frames = [pad_frame(frame) for frame in resized_frames]
        imageio.mimsave(video_path, padded_frames, fps=int(1/max(delay, 0.01)), macro_block_size=16)
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
    parser.add_argument("--env_width", type=int, default=20, help="Environment width")
    parser.add_argument("--env_height", type=int, default=20, help="Environment height")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to play")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between steps (seconds)")
    parser.add_argument("--no-render", action="store_true", help="Don't render (just compute stats)")
    parser.add_argument("--render-mode", choices=["graphical", "ascii"], default="graphical", help="Rendering mode (graphical or ascii)")
    parser.add_argument("--save-video", action="store_true", help="Save a video of the episode (graphical mode only)")
    parser.add_argument("--video-dir", type=str, default=None, help="Directory to save video files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--show-reasoning", action="store_true", help="Print reasoning text (RSM models only)")
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
    
    use_reasoning = config.get('use_reasoning', False)
    
    print(f"Model config:")
    print(f"  d_model: {config['d_model']}")
    print(f"  num_layers: {config['num_layers']}")
    print(f"  num_heads: {config['num_heads']}")
    print(f"  use_cnn: {config.get('use_cnn', False)}")
    if config.get('use_cnn'):
        print(f"  cnn_mode: {config.get('cnn_mode', 'N/A')}")
    if use_reasoning:
        print(f"  🧠 RSM mode: ENABLED")
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
        use_reasoning=use_reasoning,
    )
    
    # Create environment
    env_config = EnvConfig(
        width=args.env_width,
        height=args.env_height,
        max_steps=500,
        apple_reward=10.0,
        death_penalty=-10.0,
        step_penalty=-0.01,
    )
    env = SnakeEnv(env_config)
    
    print(f"Playing {args.episodes} episodes...")
    print(f"Environment: {env_config.width}x{env_config.height}")
    if use_reasoning:
        print(f"🧠 Reasoning mode active")
        if args.show_reasoning:
            print(f"   (Will print reasoning text)")
    print()
    
    # Play episodes
    rng = jax.random.PRNGKey(args.seed)
    
    all_rewards = []
    all_steps = []
    all_apples = []
    
    import os
    for ep in range(args.episodes):
        print(f"\n{'='*70}")
        print(f"Episode {ep + 1}/{args.episodes}")
        print(f"{'='*70}")
        rng, episode_rng = jax.random.split(rng)
        video_path = None
        if args.save_video and args.render_mode == "graphical":
            video_dir = args.video_dir or os.path.dirname(args.model)
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, f"snake_episode_{ep+1}.mp4")
        reward, steps, apples = play_episode(
            env, network, params, episode_rng,
            render=not args.no_render,
            delay=args.delay,
            render_mode=args.render_mode,
            save_video=args.save_video and args.render_mode == "graphical",
            video_path=video_path,
            use_reasoning=use_reasoning,
            show_reasoning=args.show_reasoning
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
