"""
Dataset generation for Snake pretraining.
Creates synthetic game states with expert labels.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import random
from tqdm import tqdm
import pickle
import torch
from torch.utils.data import Dataset

from pretrain_utils import (
    get_action_distribution,
    get_expert_action_astar,
    get_safe_actions,
    state_from_positions,
    augment_state_action,
    AUGMENTATIONS,
    get_positions_from_state
)
from reasoning_dsl import generate_reasoning_text, reasoning_to_embeddings


def is_valid_snake(
    snake_positions: List[Tuple[int, int]],
    width: int,
    height: int
) -> bool:
    """Check if snake configuration is valid."""
    # Check bounds
    for pos in snake_positions:
        if not (0 <= pos[0] < height and 0 <= pos[1] < width):
            return False
    
    # Check no duplicates (self-collision)
    if len(snake_positions) != len(set(snake_positions)):
        return False
    
    # Check connectivity (each segment adjacent to next)
    for i in range(len(snake_positions) - 1):
        curr = snake_positions[i]
        next_pos = snake_positions[i + 1]
        dist = abs(curr[0] - next_pos[0]) + abs(curr[1] - next_pos[1])
        if dist != 1:
            return False
    
    return True


def generate_random_snake(
    width: int,
    height: int,
    target_length: int,
    max_attempts: int = 100
) -> Optional[List[Tuple[int, int]]]:
    """
    Generate a random valid snake of given length.
    Returns list of positions [head, body..., tail] or None if failed.
    """
    for _ in range(max_attempts):
        # Random starting position
        head = (random.randint(0, height - 1), random.randint(0, width - 1))
        snake = [head]
        
        # Grow snake randomly
        for _ in range(target_length - 1):
            # Get valid neighbors
            current = snake[-1]
            neighbors = []
            
            for dx, dy in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                next_pos = (current[0] + dx, current[1] + dy)
                
                # Check bounds and not already occupied
                if (0 <= next_pos[0] < height and 
                    0 <= next_pos[1] < width and 
                    next_pos not in snake):
                    neighbors.append(next_pos)
            
            if not neighbors:
                break  # Can't grow anymore
            
            # Add random neighbor
            snake.append(random.choice(neighbors))
        
        # Check if we reached target length
        if len(snake) == target_length and is_valid_snake(snake, width, height):
            return snake
    
    return None


def generate_random_state(
    width: int,
    height: int,
    min_length: int = 3,
    max_length: int = 30
) -> Optional[Dict]:
    """
    Generate a random valid game state.
    
    Returns:
        Dict with 'snake_positions' and 'food_pos', or None if failed
    """
    # Random snake length (weighted toward shorter snakes)
    weights = [1.0 / (i + 1) for i in range(max_length - min_length + 1)]
    weights = [w / sum(weights) for w in weights]
    length = random.choices(
        range(min_length, max_length + 1),
        weights=weights
    )[0]
    
    # Generate snake
    snake = generate_random_snake(width, height, length)
    if snake is None:
        return None
    
    # Place food in non-occupied cell
    occupied = set(snake)
    free_cells = []
    for x in range(height):
        for y in range(width):
            if (x, y) not in occupied:
                free_cells.append((x, y))
    
    if not free_cells:
        return None
    
    food_pos = random.choice(free_cells)
    
    return {
        'snake_positions': snake,
        'food_pos': food_pos
    }


def generate_pretraining_dataset(
    num_samples: int,
    width: int = 20,
    height: int = 20,
    min_length: int = 3,
    max_length: int = 30,
    use_astar: bool = True,
    temperature: float = 0.5,
    augment: bool = True,
    failure_ratio: float = 0.0,
    epsilon_greedy_ratio: float = 0.0,
    epsilon: float = 0.3,
    seed: Optional[int] = None,
    add_reasoning: bool = False,
    reasoning_depth: int = 1,
    reasoning_format: str = 'compact'
) -> List[Dict]:
    """
    Generate pretraining dataset with expert labels and optional failure samples.
    
    Args:
        num_samples: Number of unique states to generate
        width, height: Grid dimensions
        min_length, max_length: Snake length range
        use_astar: Use A* for labels (vs heuristic)
        temperature: Softmax temperature for soft labels
        augment: Apply 8x geometric augmentation
        failure_ratio: Ratio of random/failure actions to include (0.0-1.0)
        epsilon_greedy_ratio: Ratio of epsilon-greedy samples to include (0.0-1.0)
        epsilon: Epsilon value for epsilon-greedy sampling
        seed: Random seed
        add_reasoning: Add CoT-style reasoning text before action (RSM mode)
        reasoning_depth: Lookahead depth for reasoning (1-3)
        reasoning_format: 'compact' or 'verbose'
    
    Returns:
        List of dicts with 'state', 'action', 'action_probs', 'metadata', and optionally 'reasoning'
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    dataset = []
    attempts = 0
    max_attempts = num_samples * 10
    
    pbar = tqdm(total=num_samples, desc="Generating dataset")
    
    while len(dataset) < num_samples and attempts < max_attempts:
        attempts += 1
        
        # Generate random state
        state_dict = generate_random_state(width, height, min_length, max_length)
        if state_dict is None:
            continue
        
        snake_positions = state_dict['snake_positions']
        food_pos = state_dict['food_pos']
        
        # Check if there are safe actions
        safe_actions = get_safe_actions(snake_positions, width, height)
        if not safe_actions:
            continue  # Skip terminal states
        
        # Get expert label
        if use_astar:
            expert_action = get_expert_action_astar(
                snake_positions, food_pos, width, height
            )
            if expert_action is None:
                # No path to food, use safest heuristic action
                action_probs = get_action_distribution(
                    snake_positions, food_pos, width, height,
                    temperature=temperature, use_astar=False
                )
                expert_action = int(np.argmax(action_probs))
            else:
                # Hard label for A* solution
                action_probs = np.zeros(4, dtype=np.float32)
                action_probs[expert_action] = 1.0
        else:
            # Soft labels from heuristic
            action_probs = get_action_distribution(
                snake_positions, food_pos, width, height,
                temperature=temperature, use_astar=False
            )
            expert_action = int(np.argmax(action_probs))
        
        # Convert to state array
        state = state_from_positions(snake_positions, food_pos, width, height)
        
        # Generate reasoning text if requested
        reasoning_text = None
        reasoning_tokens = None
        if add_reasoning:
            reasoning_text = generate_reasoning_text(
                snake_positions, food_pos, width, height, expert_action,
                lookahead_depth=reasoning_depth, format=reasoning_format
            )
            reasoning_tokens = reasoning_to_embeddings(reasoning_text, d_model=64, max_length=128)
        
        # Create base sample
        sample = {
            'state': state,
            'action': expert_action,
            'action_probs': action_probs,
            'metadata': {
                'snake_length': len(snake_positions),
                'distance_to_food': abs(snake_positions[0][0] - food_pos[0]) + 
                                   abs(snake_positions[0][1] - food_pos[1]),
                'num_safe_actions': len(safe_actions)
            }
        }
        
        if add_reasoning:
            sample['reasoning'] = reasoning_text
            sample['reasoning_tokens'] = reasoning_tokens
        
        dataset.append(sample)
        pbar.update(1)
    
    pbar.close()
    
    if len(dataset) < num_samples:
        print(f"Warning: Only generated {len(dataset)}/{num_samples} samples")
    
    # Add failure samples (random actions)
    if failure_ratio > 0:
        num_failures = int(len(dataset) * failure_ratio)
        print(f"Adding {num_failures} failure samples (random actions)...")
        
        failure_samples = []
        for i in tqdm(range(num_failures), desc="Generating failures"):
            # Pick a random sample from the dataset
            base_sample = random.choice(dataset)
            
            # Extract snake positions and food from state
            state = base_sample['state']
            snake_positions, food_pos = get_positions_from_state(state)
            
            safe_actions = get_safe_actions(snake_positions, width, height)
            if not safe_actions:
                continue
            
            # Pick a random action (could be safe or unsafe)
            random_action = random.randint(0, 3)
            
            # Create uniform probability distribution
            failure_probs = np.ones(4, dtype=np.float32) / 4.0
            
            failure_sample = {
                'state': state.copy(),
                'action': random_action,
                'action_probs': failure_probs,
                'metadata': base_sample['metadata'].copy()
            }
            
            # Generate reasoning for failure samples too (shows wrong reasoning)
            if add_reasoning:
                reasoning_text = generate_reasoning_text(
                    snake_positions, food_pos, width, height, random_action,
                    lookahead_depth=reasoning_depth, format=reasoning_format
                )
                reasoning_tokens = reasoning_to_embeddings(reasoning_text, d_model=64, max_length=128)
                failure_sample['reasoning'] = reasoning_text
                failure_sample['reasoning_tokens'] = reasoning_tokens
            
            failure_samples.append(failure_sample)
        
        dataset.extend(failure_samples)
        print(f"Dataset size after failures: {len(dataset)}")
    
    # Add epsilon-greedy samples
    if epsilon_greedy_ratio > 0:
        num_epsilon = int(len(dataset) * epsilon_greedy_ratio)
        print(f"Adding {num_epsilon} epsilon-greedy samples (ε={epsilon})...")
        
        epsilon_samples = []
        for i in tqdm(range(num_epsilon), desc="Generating epsilon-greedy"):
            # Pick a random sample from the dataset
            base_sample = random.choice(dataset)
            
            # With probability epsilon, use random action; otherwise use expert
            if random.random() < epsilon:
                # Random action
                random_action = random.randint(0, 3)
                epsilon_probs = np.ones(4, dtype=np.float32) / 4.0
            else:
                # Use expert action
                random_action = base_sample['action']
                epsilon_probs = base_sample['action_probs'].copy()
            
            epsilon_sample = {
                'state': base_sample['state'].copy(),
                'action': random_action,
                'action_probs': epsilon_probs,
                'metadata': base_sample['metadata'].copy()
            }
            
            # Generate reasoning for epsilon-greedy samples
            if add_reasoning:
                state = base_sample['state']
                snake_positions, food_pos = get_positions_from_state(state)
                reasoning_text = generate_reasoning_text(
                    snake_positions, food_pos, width, height, random_action,
                    lookahead_depth=reasoning_depth, format=reasoning_format
                )
                reasoning_tokens = reasoning_to_embeddings(reasoning_text, d_model=64, max_length=128)
                epsilon_sample['reasoning'] = reasoning_text
                epsilon_sample['reasoning_tokens'] = reasoning_tokens
            
            epsilon_samples.append(epsilon_sample)
        
        dataset.extend(epsilon_samples)
        print(f"Dataset size after epsilon-greedy: {len(dataset)}")
    
    # Shuffle to mix expert/failure/epsilon samples
    if failure_ratio > 0 or epsilon_greedy_ratio > 0:
        random.shuffle(dataset)
    
    # Apply augmentation if requested
    if augment:
        print("Applying geometric augmentation (8x)...")
        augmented_dataset = []
        
        for sample in tqdm(dataset, desc="Augmenting"):
            for aug_type in AUGMENTATIONS:
                aug_state, aug_action = augment_state_action(
                    sample['state'],
                    sample['action'],
                    aug_type
                )
                
                # Transform action probabilities as well
                if aug_type == 'identity':
                    aug_probs = sample['action_probs']
                else:
                    # Create new probability distribution for augmented actions
                    aug_probs = np.zeros(4, dtype=np.float32)
                    for orig_action in range(4):
                        _, transformed_action = augment_state_action(
                            sample['state'], orig_action, aug_type
                        )
                        aug_probs[transformed_action] += sample['action_probs'][orig_action]
                
                augmented_dataset.append({
                    'state': aug_state,
                    'action': aug_action,
                    'action_probs': aug_probs,
                    'metadata': sample['metadata'].copy()
                })
        
        dataset = augmented_dataset
        print(f"Dataset size after augmentation: {len(dataset)}")
    
    return dataset


def save_dataset(dataset: List[Dict], filepath: str):
    """Save dataset to disk."""
    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Saved dataset to {filepath}")


def load_dataset(filepath: str) -> List[Dict]:
    """Load dataset from disk."""
    with open(filepath, 'rb') as f:
        dataset = pickle.load(f)
    print(f"Loaded {len(dataset)} samples from {filepath}")
    return dataset


class SnakePretrainDataset(Dataset):
    """PyTorch Dataset for Snake pretraining."""
    
    def __init__(self, data: List[Dict], use_soft_labels: bool = False):
        """
        Args:
            data: List of sample dicts from generate_pretraining_dataset
            use_soft_labels: If True, return action_probs; else return hard action
        """
        self.data = data
        self.use_soft_labels = use_soft_labels
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        
        state = torch.from_numpy(sample['state']).float()
        action = torch.tensor(sample['action'], dtype=torch.long)
        
        if self.use_soft_labels:
            action_probs = torch.from_numpy(sample['action_probs']).float()
            return {
                'state': state,
                'action': action,
                'action_probs': action_probs
            }
        else:
            return {
                'state': state,
                'action': action
            }


def analyze_dataset(dataset: List[Dict]):
    """Print statistics about the dataset."""
    print("\n=== Dataset Statistics ===")
    print(f"Total samples: {len(dataset)}")
    
    # Action distribution
    actions = [s['action'] for s in dataset]
    action_names = ['Up', 'Right', 'Down', 'Left']
    print("\nAction distribution:")
    for i, name in enumerate(action_names):
        count = sum(1 for a in actions if a == i)
        print(f"  {name}: {count} ({100*count/len(actions):.1f}%)")
    
    # Snake length distribution
    lengths = [s['metadata']['snake_length'] for s in dataset]
    print(f"\nSnake length: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.1f}, median={np.median(lengths):.1f}")
    
    # Distance to food
    distances = [s['metadata']['distance_to_food'] for s in dataset]
    print(f"\nDistance to food: min={min(distances)}, max={max(distances)}, "
          f"mean={np.mean(distances):.1f}, median={np.median(distances):.1f}")
    
    # Safe actions
    safe_counts = [s['metadata']['num_safe_actions'] for s in dataset]
    print(f"\nSafe actions: min={min(safe_counts)}, max={max(safe_counts)}, "
          f"mean={np.mean(safe_counts):.1f}")
    
    print()


if __name__ == "__main__":
    # Test dataset generation
    print("Testing dataset generation...")
    
    dataset = generate_pretraining_dataset(
        num_samples=1000,
        width=20,
        height=20,
        use_astar=True,
        augment=True,
        seed=42
    )
    
    analyze_dataset(dataset)
    
    # Save test dataset
    save_dataset(dataset, "test_pretrain_dataset.pkl")
    
    # Test PyTorch dataset
    print("Testing PyTorch Dataset...")
    torch_dataset = SnakePretrainDataset(dataset[:100], use_soft_labels=True)
    sample = torch_dataset[0]
    print(f"State shape: {sample['state'].shape}")
    print(f"Action: {sample['action']}")
    print(f"Action probs: {sample['action_probs']}")
