"""
Validate the expert dataset by replaying trajectories

This will show us if the A* trajectories are actually good.
"""

import pickle
import jax.numpy as jnp
import time
from collections import defaultdict

from snake_jax.config import EnvConfig
from snake_jax.env import SnakeEnv


def render_state(state, env_config):
    """Simple ASCII visualization"""
    grid = [['.' for _ in range(env_config.width)] for _ in range(env_config.height)]
    
    # Place food
    food_x, food_y = int(state.food_pos[0]), int(state.food_pos[1])
    if 0 <= food_y < env_config.height and 0 <= food_x < env_config.width:
        grid[food_y][food_x] = '🍎'
    
    # Place snake
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


def replay_trajectory_from_samples(samples, env, delay=0.1):
    """
    Try to reconstruct and replay a trajectory from dataset samples.
    
    Note: Dataset samples are individual state-action pairs, not full trajectories.
    We'll just show the states to see what the expert saw.
    """
    print(f"\nShowing {len(samples)} expert state-action pairs:")
    print("="*50)
    
    apples_eaten = 0
    prev_length = None
    
    for i, sample in enumerate(samples):
        state_grid = sample['state']  # This is the observation grid (H, W, 3) in RGB
        action = sample['action']
        metadata = sample['metadata']
        
        # RGB encoding: R=food (red), G=snake (green), B=unused (black empty)
        red_channel = state_grid[:, :, 0]    # Food
        green_channel = state_grid[:, :, 1]  # Snake
        
        print(f"\nStep {i+1}/{len(samples)}:")
        print(f"  Action: {action} (0=up, 1=right, 2=down, 3=left)")
        print(f"  Snake length: {metadata['snake_length']}")
        print(f"  Distance to food: {metadata['distance_to_food']}")
        
        if prev_length is not None and metadata['snake_length'] > prev_length:
            print(f"  🍎 APPLE EATEN!")
            apples_eaten += 1
        
        prev_length = metadata['snake_length']
        
        # Print the grid
        print()
        print('┌' + '──' * state_grid.shape[1] + '┐')
        for y in range(state_grid.shape[0]):
            row = '│'
            for x in range(state_grid.shape[1]):
                if red_channel[y, x] > 0:      # Food (red)
                    row += '🍎 '
                elif green_channel[y, x] > 0:  # Snake (green)
                    row += '🟢 '
                else:                          # Empty (black)
                    row += '. '
            row += '│'
            print(row)
        print('└' + '──' * state_grid.shape[1] + '┘')
        
        time.sleep(delay)
    
    print(f"\n" + "="*50)
    print(f"Apples eaten in this sequence: {apples_eaten}")
    print("="*50)


def main():
    print("="*70)
    print("EXPERT DATASET VALIDATION")
    print("="*70)
    print()
    
    # Load dataset
    print("Loading dataset...")
    with open('snake_pretrain_dataset.pkl', 'rb') as f:
        data = pickle.load(f)
    
    print(f"Total samples: {len(data)}")
    print()
    
    # Group samples by snake length to find trajectory sequences
    # (This is a heuristic - samples aren't marked by trajectory)
    by_length = defaultdict(list)
    for sample in data:
        length = sample['metadata']['snake_length']
        by_length[length].append(sample)
    
    print("Samples by snake length:")
    for length in sorted(by_length.keys()):
        print(f"  Length {length}: {len(by_length[length])} samples")
    print()
    
    # Show a sequence that grows (apple-eating sequence)
    print("Looking for an apple-eating sequence...")
    
    # Find samples with consecutive lengths
    sequence = []
    for length in range(3, 15):  # Snake starts at 3, look up to length 15
        if length in by_length and by_length[length]:
            # Take first sample of this length
            sequence.append(by_length[length][0])
            if len(sequence) >= 10:
                break
    
    if sequence:
        print(f"Found sequence with {len(sequence)} samples showing growth:")
        replay_trajectory_from_samples(sequence, None, delay=0.3)
    else:
        # Just show first 20 samples
        print("Showing first 20 samples from dataset:")
        replay_trajectory_from_samples(data[:20], None, delay=0.2)
    
    # Statistics
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    
    lengths = [s['metadata']['snake_length'] for s in data]
    distances = [s['metadata']['distance_to_food'] for s in data]
    
    print(f"Snake lengths: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.1f}")
    print(f"Distances to food: min={min(distances)}, max={max(distances)}, mean={sum(distances)/len(distances):.1f}")
    print()


if __name__ == "__main__":
    main()
