"""
Play Snake with a pretrained HuggingFace model

Loads a pretrained model saved by train_hf.py and visualizes it playing Snake.
Supports both standard and RSM (Reasoning Snake Model) modes.
"""

import argparse
import time
import os
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio

from environments.snake import SnakeGame
from model.model_pytorch import SnakeTransformerConfig, TransformerPolicy
from render_utils import render_ascii, render_state_for_video


def generate_reasoning_autoregressive(
    model, obs, max_tokens=128, temperature=0.0, device='cpu'
):
    """
    Autoregressively generate reasoning tokens.
    
    The model generates tokens one by one, attending to grid + previous reasoning tokens.
    Generation stops when an action token (0-3) is predicted.
    
    Args:
        model: TransformerPolicy model
        obs: Grid observation (height, width, 3) as numpy array
        max_tokens: Maximum reasoning tokens to generate
        temperature: Sampling temperature (0.0 = greedy)
        device: Device to run on
    
    Returns:
        reasoning_tokens: Generated reasoning token IDs (tensor)
        action: Final predicted action (0-3)
        reasoning_text: Decoded reasoning string
    """
    model.eval()
    with torch.no_grad():
        # Convert obs to tensor and add batch dimension
        # obs is (H, W, 3), need (1, 3, H, W) for model
        obs_tensor = torch.from_numpy(obs).float()
        obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        obs_tensor = obs_tensor.to(device)
        
        # Start with empty reasoning
        reasoning_tokens = None
        generated_tokens = []
        
        # Special token markers
        ACTION_OFFSET = 4  # Actions are 0-3, reasoning tokens start at 4
        
        for t in range(max_tokens):
            # Forward pass with current reasoning tokens
            if reasoning_tokens is not None:
                reasoning_tokens_tensor = reasoning_tokens.to(device)
            else:
                reasoning_tokens_tensor = None
            
            logits, _ = model(obs_tensor, reasoning_tokens=reasoning_tokens_tensor)
            # logits shape: (1, num_actions + vocab_size) = (1, 4 + 512) = (1, 516)
            
            # Sample/pick next token
            if temperature == 0.0:
                next_token_id = torch.argmax(logits[0]).item()
            else:
                # Temperature sampling
                probs = torch.softmax(logits[0] / temperature, dim=0)
                next_token_id = torch.multinomial(probs, 1).item()
            
            # Check if it's an action token (0-3)
            if next_token_id < 4:
                # Model decided to predict action - stop reasoning
                action = next_token_id
                break
            
            # It's a reasoning token - add to sequence
            reasoning_token = next_token_id - ACTION_OFFSET  # Convert back to vocab range
            generated_tokens.append(reasoning_token)
            
            # Append to reasoning sequence
            new_token = torch.tensor([[reasoning_token]], dtype=torch.long, device=device)
            if reasoning_tokens is None:
                reasoning_tokens = new_token
            else:
                reasoning_tokens = torch.cat([reasoning_tokens, new_token], dim=1)
        else:
            # Max tokens reached - force action prediction
            # Do one more forward pass to get action
            if reasoning_tokens is not None:
                reasoning_tokens_tensor = reasoning_tokens.to(device)
            else:
                reasoning_tokens_tensor = None
            logits, _ = model(obs_tensor, reasoning_tokens=reasoning_tokens_tensor)
            action = torch.argmax(logits[0, :4]).item()  # Only consider action logits
        
        # Decode reasoning text
        if generated_tokens:
            reasoning_text = ''.join(chr(min(max(t, 0), 127)) for t in generated_tokens)
        else:
            reasoning_text = ""
        
        return reasoning_tokens, action, reasoning_text


def render_state(state, env, mode="graphical", return_frame=False, moves=None, score=None):
    """Render the game state in ASCII or graphical mode (default: graphical)
    If return_frame is True, returns the RGB numpy array for video saving."""
    if mode == "ascii":
        render_ascii(state, action=0, action_names=False)  # action not needed for rendering
        print(f'Length: {len(env.snake)} | Score: {env.score}')
        if moves is not None:
            print(f'Moves: {moves}')
        return None
    else:
        # Graphical rendering using matplotlib
        if return_frame:
            frame = render_state_for_video(state, moves=moves, score=score)
            return frame
        else:
            # Interactive rendering
            grid = np.zeros((env.height, env.width, 3), dtype=np.float32)
            # Background
            grid[:] = [0.9, 0.9, 0.9]
            
            # Food
            for food_x, food_y in env.foods:
                if 0 <= food_y < env.height and 0 <= food_x < env.width:
                    grid[food_y, food_x] = [1.0, 0.0, 0.0]  # Red
            
            # Snake body
            for i, (snake_x, snake_y) in enumerate(env.snake):
                if 0 <= snake_y < env.height and 0 <= snake_x < env.width:
                    if i == 0:
                        grid[snake_y, snake_x] = [0.0, 0.8, 0.0]  # Head: green
                    else:
                        grid[snake_y, snake_x] = [0.0, 0.5, 0.0]  # Body: darker green
            
            plt.clf()
            plt.imshow(grid, interpolation='nearest')
            title_str = f"Length: {len(env.snake)} | Score: {env.score}"
            if moves is not None:
                title_str += f" | Moves: {moves}"
            plt.title(title_str)
            plt.axis('off')
            plt.pause(0.001)
            return None


def play_episode(env, model, device, render=True, delay=0.1, render_mode="graphical", 
                 save_video=False, video_path=None, use_reasoning=False, show_reasoning=False):
    """Play one episode and optionally render it and save video"""
    # Reset environment
    state = env.reset()
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
    
    max_steps = 500  # Safety limit
    while not done and steps < max_steps:
        frame = None
        if render:
            frame = render_state(state, env, mode=render_mode, return_frame=save_video, moves=steps, score=env.score)
            time.sleep(delay)
        elif save_video:
            frame = render_state(state, env, mode=render_mode, return_frame=True, moves=steps, score=env.score)
        if save_video and frame is not None:
            frames.append(frame)
        
        # Get action from policy
        # State format now matches model format (R=food, G=snake, B=empty)
        # Use autoregressive reasoning generation for RSM models
        if use_reasoning:
            # Generate reasoning autoregressively
            reasoning_tokens, action, reasoning_text = generate_reasoning_autoregressive(
                model, state, max_tokens=128, temperature=0.0, device=device
            )
            
            if show_reasoning:
                print(f"  🧠 Generated reasoning: {reasoning_text}")
                print(f"     Action: {['UP', 'RIGHT', 'DOWN', 'LEFT'][action]}")
        else:
            # Standard forward pass without reasoning
            model.eval()
            with torch.no_grad():
                # Convert state to tensor: (H, W, 3) -> (1, 3, H, W)
                obs_tensor = torch.from_numpy(state).float()
                obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
                obs_tensor = obs_tensor.to(device)
                
                logits, _ = model(obs_tensor)
                action = torch.argmax(logits[0, :4]).item()  # Extract action from logits
        
        # Step environment
        state, reward, done = env.step(action)
        total_reward += float(reward)
        steps += 1
        # Count apples
        if env.ate_last_step:
            apples_eaten += 1
            print(f"\n🍎 Apple eaten! Total: {apples_eaten}, Score: {total_reward:.1f}")
    
    if render and render_mode == "graphical":
        plt.ioff()
        plt.close(fig)
    
    if save_video and frames and video_path:
        print(f"Saving video to {video_path}...")
        # Ensure all frames are the same size and divisible by macro_block_size=16
        target_shape = frames[0].shape
        # Make width and height divisible by 16
        def pad_to_16(x):
            return ((x + 15) // 16) * 16
        
        padded_height = pad_to_16(target_shape[0])
        padded_width = pad_to_16(target_shape[1])
        
        def pad_frame(frame):
            h, w, c = frame.shape
            pad_h = padded_height - h
            pad_w = padded_width - w
            return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=255)
        
        # Resize frames to target shape if needed, then pad
        # Use simple nearest neighbor resizing with numpy
        resized_frames = []
        for frame in frames:
            if frame.shape != target_shape:
                # Simple resize using numpy indexing (nearest neighbor)
                h, w = target_shape[:2]
                h_old, w_old = frame.shape[:2]
                y_indices = np.round(np.linspace(0, h_old - 1, h)).astype(int)
                x_indices = np.round(np.linspace(0, w_old - 1, w)).astype(int)
                frame = frame[np.ix_(y_indices, x_indices)]
            resized_frames.append(frame)
        
        padded_frames = [pad_frame(frame) for frame in resized_frames]
        imageio.mimsave(video_path, padded_frames, fps=int(1/max(delay, 0.01)), macro_block_size=16)
    
    print("\n" + "="*50)
    print(f"Episode finished!")
    print(f"  Steps: {steps}")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  Apples Eaten: {apples_eaten}")
    print(f"  Snake Length: {len(env.snake)}")
    print("="*50)
    return total_reward, steps, apples_eaten


def load_model(model_path, device='cpu'):
    """
    Load a HuggingFace model from a checkpoint directory.
    
    Args:
        model_path: Path to model directory (should contain config.json and model files)
        device: Device to load model on
    
    Returns:
        model: Loaded TransformerPolicy model
        config: Model configuration
    """
    model_path = Path(model_path)
    
    # Try loading as HuggingFace model first
    try:
        config = SnakeTransformerConfig.from_pretrained(str(model_path))
        model = TransformerPolicy.from_pretrained(str(model_path))
        model = model.to(device)
        model.eval()
        return model, config
    except Exception as e:
        # Fallback: try loading config.json and state dict manually
        print(f"Could not load as HuggingFace model: {e}")
        print("Trying to load config and state dict manually...")
        
        import json
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = SnakeTransformerConfig(**config_dict)
        model = TransformerPolicy(config)
        
        # Try to load state dict
        state_dict_path = model_path / "pytorch_model.bin"
        if not state_dict_path.exists():
            # Try alternative names
            state_dict_path = model_path / "model.safetensors"
            if not state_dict_path.exists():
                raise FileNotFoundError(f"Model weights not found in {model_path}")
        
        if state_dict_path.suffix == '.bin':
            state_dict = torch.load(state_dict_path, map_location=device)
        else:
            # For safetensors, would need safetensors library
            raise NotImplementedError("Safetensors format not yet supported")
        
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return model, config


def main():
    parser = argparse.ArgumentParser(description="Play Snake with pretrained HuggingFace model")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--env-width", type=int, default=32, help="Environment width")
    parser.add_argument("--env-height", type=int, default=32, help="Environment height")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to play")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between steps (seconds)")
    parser.add_argument("--no-render", action="store_true", help="Don't render (just compute stats)")
    parser.add_argument("--render-mode", choices=["graphical", "ascii"], default="graphical", help="Rendering mode (graphical or ascii)")
    parser.add_argument("--save-video", action="store_true", help="Save a video of the episode (graphical mode only)")
    parser.add_argument("--video-dir", type=str, default=None, help="Directory to save video files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--show-reasoning", action="store_true", help="Print reasoning text (RSM models only)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)")
    args = parser.parse_args()
    
    print("="*70)
    print("PRETRAINED MODEL EVALUATION")
    print("="*70)
    print()
    
    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model, config = load_model(args.model, device=device)
    
    use_reasoning = config.use_reasoning
    
    print(f"Model config:")
    print(f"  d_model: {config.d_model}")
    print(f"  num_layers: {config.num_layers}")
    print(f"  num_heads: {config.num_heads}")
    print(f"  use_cnn: {config.use_cnn}")
    if config.use_cnn:
        print(f"  cnn_mode: {config.cnn_mode}")
    if use_reasoning:
        print(f"  🧠 RSM mode: ENABLED")
    print()
    
    # Create environment
    env = SnakeGame(
        width=args.env_width,
        height=args.env_height,
        step_penalty=-0.01,
        wall_collision=True,
        apple_reward=10.0,
        death_penalty=-10.0,
    )
    
    print(f"Playing {args.episodes} episodes...")
    print(f"Environment: {args.env_width}x{args.env_height}")
    if use_reasoning:
        print(f"🧠 Reasoning mode active")
        if args.show_reasoning:
            print(f"   (Will print reasoning text)")
    print()
    
    # Set random seed
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Play episodes
    all_rewards = []
    all_steps = []
    all_apples = []
    
    for ep in range(args.episodes):
        print(f"\n{'='*70}")
        print(f"Episode {ep + 1}/{args.episodes}")
        print(f"{'='*70}")
        video_path = None
        if args.save_video and args.render_mode == "graphical":
            video_dir = args.video_dir or os.path.dirname(args.model)
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, f"snake_episode_{ep+1}.mp4")
        reward, steps, apples = play_episode(
            env, model, device,
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
    print(f"  Mean: {np.mean(all_rewards):.2f}")
    print(f"  Std:  {np.std(all_rewards):.2f}")
    print(f"  Min:  {np.min(all_rewards):.2f}")
    print(f"  Max:  {np.max(all_rewards):.2f}")
    print(f"\nApples eaten:")
    print(f"  Mean: {np.mean(all_apples):.2f}")
    print(f"  Std:  {np.std(all_apples):.2f}")
    print(f"  Min:  {int(np.min(all_apples))}")
    print(f"  Max:  {int(np.max(all_apples))}")
    print(f"\nSteps survived:")
    print(f"  Mean: {np.mean(all_steps):.2f}")
    print(f"  Std:  {np.std(all_steps):.2f}")
    print(f"  Min:  {int(np.min(all_steps))}")
    print(f"  Max:  {int(np.max(all_steps))}")
    print("="*70)
    print()


if __name__ == "__main__":
    main()
