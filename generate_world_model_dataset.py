#!/usr/bin/env python3
"""
Generate Snake World Model dataset with (state, action, next_state) tuples

This extends the standard dataset generation to include next_state,
which is needed for training the EqM world model.
"""

import argparse
from pathlib import Path
import numpy as np
from datasets import Dataset, DatasetDict
from pretrain_dataset import generate_pretraining_dataset
from pretrain_utils import state_from_positions
import sys
from environments.snake import SnakeEnv


def apply_action_get_next_state(state, action, width=32, height=32):
    """
    Apply action to state and get next state
    
    Args:
        state: (H, W, 3) numpy array
        action: int (0=up, 1=right, 2=down, 3=left)
        width, height: grid dimensions
    
    Returns:
        next_state: (H, W, 3) numpy array after taking action
    """
    # Create a temporary environment and set its state
    env = SnakeEnv(width=width, height=height)
    
    # Extract snake and food positions from state
    # state[:, :, 1] is the green channel (snake body)
    # state[:, :, 0] is the red channel (food)
    
    snake_grid = state[:, :, 1]  # Green channel
    food_grid = state[:, :, 0]   # Red channel
    
    # Find snake positions (values indicate order: head=1.0, decreasing for body)
    snake_positions = []
    snake_coords = np.argwhere(snake_grid > 0)
    
    if len(snake_coords) == 0:
        # Invalid state, return same state
        return state
    
    # Sort by intensity (head is brightest)
    snake_values = [snake_grid[y, x] for y, x in snake_coords]
    sorted_indices = np.argsort(snake_values)[::-1]
    snake_positions = [(int(snake_coords[i][1]), int(snake_coords[i][0])) 
                       for i in sorted_indices]
    
    # Find food position
    food_coords = np.argwhere(food_grid > 0)
    if len(food_coords) > 0:
        food_position = (int(food_coords[0][1]), int(food_coords[0][0]))
    else:
        # No food, use random position
        food_position = (width // 2, height // 2)
    
    # Set environment state
    env.snake = snake_positions
    env.food = food_position
    
    # Take action
    _, _, done, _ = env.step(action)
    
    # Get next state
    if done:
        # If game ended, return terminal state (empty grid)
        next_state = np.zeros_like(state)
    else:
        # Convert environment state to grid representation
        next_state = state_from_positions(
            env.snake,
            env.food,
            width=width,
            height=height
        )
    
    return next_state


def convert_to_world_model_dataset(samples, width=32, height=32):
    """
    Convert list of samples to world model dataset with next_state
    
    Args:
        samples: List of dicts with 'state', 'action'
        width, height: grid dimensions
    
    Returns:
        HuggingFace Dataset with 'state', 'action', 'next_state'
    """
    states = []
    actions = []
    next_states = []
    
    print("Computing next states...")
    for i, sample in enumerate(samples):
        if i % 10000 == 0:
            print(f"  Processed {i}/{len(samples)} samples...")
        
        state = sample['state']
        action = sample['action']
        
        # Compute next state
        next_state = apply_action_get_next_state(
            state, 
            action, 
            width=width, 
            height=height
        )
        
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
    
    print(f"  Processed {len(samples)}/{len(samples)} samples.")
    
    # Convert to numpy arrays
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int64)
    next_states = np.array(next_states, dtype=np.float32)
    
    data_dict = {
        'state': states,
        'action': actions,
        'next_state': next_states,
    }
    
    return Dataset.from_dict(data_dict)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Snake World Model dataset with next_state"
    )
    
    parser.add_argument('--num-samples', type=int, default=10000, 
                        help='Number of unique states to generate (before augmentation)')
    parser.add_argument('--width', type=int, default=32, help='Grid width')
    parser.add_argument('--height', type=int, default=32, help='Grid height')
    parser.add_argument('--min-length', type=int, default=3, help='Minimum snake length')
    parser.add_argument('--max-length', type=int, default=30, help='Maximum snake length')
    parser.add_argument('--use-astar', action='store_true', default=True, 
                        help='Use A* for expert labels (default: True)')
    parser.add_argument('--no-astar', dest='use_astar', action='store_false', 
                        help='Use heuristics instead of A*')
    parser.add_argument('--temperature', type=float, default=0.5, 
                        help='Temperature for soft labels')
    parser.add_argument('--augment', action='store_true', default=True, 
                        help='Apply 8x geometric augmentation (default: True)')
    parser.add_argument('--no-augment', dest='augment', action='store_false', 
                        help='Skip augmentation')
    parser.add_argument('--output', type=str, default='snake_world_dataset_hf', 
                        help='Output directory path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--val-split', type=float, default=0.1, 
                        help='Validation split fraction')

    args = parser.parse_args()
    
    print("=" * 70)
    print("SNAKE WORLD MODEL DATASET GENERATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Samples: {args.num_samples:,}")
    print(f"  Grid: {args.width}x{args.height}")
    print(f"  Snake length: {args.min_length}-{args.max_length}")
    print(f"  Expert: {'A*' if args.use_astar else 'Heuristic'}")
    print(f"  Augmentation: {args.augment}")
    print(f"  Validation split: {args.val_split:.1%}")
    print(f"  Output: outputs/datasets/{args.output}/")
    print()
    
    # Generate base dataset (without next_state)
    print("Generating base dataset...")
    samples = generate_pretraining_dataset(
        num_samples=args.num_samples,
        width=args.width,
        height=args.height,
        min_length=args.min_length,
        max_length=args.max_length,
        use_astar=args.use_astar,
        temperature=args.temperature,
        augment=args.augment,
        seed=args.seed,
    )
    
    print(f"✅ Generated {len(samples):,} samples")
    print()
    
    # Split into train/val
    np.random.seed(args.seed)
    indices = np.random.permutation(len(samples))
    val_size = int(len(samples) * args.val_split)
    
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]
    
    print(f"Split: {len(train_samples):,} train, {len(val_samples):,} val")
    print()
    
    # Convert to HuggingFace datasets with next_state
    print("Converting to HuggingFace format with next_state...")
    train_dataset = convert_to_world_model_dataset(
        train_samples, 
        width=args.width, 
        height=args.height
    )
    val_dataset = convert_to_world_model_dataset(
        val_samples, 
        width=args.width, 
        height=args.height
    )
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
    })
    
    # Save to disk
    output_dir = Path('outputs') / 'datasets' / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to {output_dir}...")
    dataset_dict.save_to_disk(str(output_dir))
    
    print()
    print("=" * 70)
    print("✅ DATASET GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\nDataset saved to: {output_dir}")
    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Validation: {len(val_dataset):,} samples")
    print(f"\nDataset features:")
    print(f"  state: {train_dataset['state'].shape}")
    print(f"  action: {train_dataset['action'].shape}")
    print(f"  next_state: {train_dataset['next_state'].shape}")
    print()
    print(f"To train the world model, run:")
    print(f"  python train_snake_world.py --dataset {output_dir} --epochs 10")
    print()


if __name__ == "__main__":
    main()
