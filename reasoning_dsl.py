"""
Domain-Specific Language (DSL) for Snake reasoning trajectories.

Generates compact text descriptions of lookahead reasoning for each action.
This enables "Reasoning Snake Model" (RSM) - CoT-style reasoning before action prediction.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from pretrain_utils import manhattan_distance, count_reachable_cells


# Direction mapping
DIRECTION_NAMES = {0: 'U', 1: 'R', 2: 'D', 3: 'L'}
DIRECTION_FULL = {0: 'UP', 1: 'RIGHT', 2: 'DOWN', 3: 'LEFT'}
ACTION_DELTAS = {
    0: (-1, 0),  # Up
    1: (0, 1),   # Right
    2: (1, 0),   # Down
    3: (0, -1),  # Left
}


def check_action_outcome(
    head: Tuple[int, int],
    action: int,
    snake_positions: List[Tuple[int, int]],
    food_pos: Tuple[int, int],
    width: int,
    height: int,
) -> Dict:
    """
    Check the immediate outcome of taking an action.
    
    Returns dict with:
        - outcome: 'wall', 'death', 'safe', 'apple'
        - distance: Manhattan distance to food after move
        - freedom: Number of reachable cells after move (0 if death/wall)
    """
    dx, dy = ACTION_DELTAS[action]
    next_pos = (head[0] + dx, head[1] + dy)
    
    # Check bounds
    if not (0 <= next_pos[0] < height and 0 <= next_pos[1] < width):
        return {'outcome': 'wall', 'distance': 999, 'freedom': 0}
    
    # Check if it's the food
    if next_pos == food_pos:
        # Simulate eating apple - snake grows
        next_snake = [next_pos] + snake_positions  # Snake grows
        freedom = count_reachable_cells(next_pos, next_snake, width, height)
        return {'outcome': 'apple', 'distance': 0, 'freedom': freedom}
    
    # Check self-collision (tail will move, so tail position is OK)
    occupied = set(snake_positions)
    if next_pos in occupied and next_pos != snake_positions[-1]:
        return {'outcome': 'death', 'distance': 999, 'freedom': 0}
    
    # Safe move
    next_snake = [next_pos] + snake_positions[:-1]  # Tail moves
    distance = manhattan_distance(next_pos, food_pos)
    freedom = count_reachable_cells(next_pos, next_snake, width, height)
    
    return {'outcome': 'safe', 'distance': distance, 'freedom': freedom}


def multi_step_lookahead(
    head: Tuple[int, int],
    action: int,
    snake_positions: List[Tuple[int, int]],
    food_pos: Tuple[int, int],
    width: int,
    height: int,
    depth: int = 1,
) -> List[Dict]:
    """
    Perform multi-step lookahead for a given action.
    Continues in the same direction for 'depth' steps.
    
    Returns list of outcome dicts, one per step.
    """
    outcomes = []
    current_pos = head
    current_snake = list(snake_positions)
    
    for step in range(depth):
        result = check_action_outcome(
            current_pos, action, current_snake, food_pos, width, height
        )
        outcomes.append(result)
        
        # Stop if we hit wall/death/apple
        if result['outcome'] in ['wall', 'death', 'apple']:
            break
        
        # Update for next step (continue in same direction)
        dx, dy = ACTION_DELTAS[action]
        current_pos = (current_pos[0] + dx, current_pos[1] + dy)
        current_snake = [current_pos] + current_snake[:-1]
    
    return outcomes


def generate_reasoning_text(
    snake_positions: List[Tuple[int, int]],
    food_pos: Tuple[int, int],
    width: int,
    height: int,
    expert_action: int,
    lookahead_depth: int = 1,
    format: str = 'compact'
) -> str:
    """
    Generate reasoning text in DSL format.
    
    Args:
        snake_positions: List of (row, col) positions [head, body..., tail]
        food_pos: Food (row, col)
        width, height: Grid dimensions
        expert_action: The chosen action (0-3)
        lookahead_depth: How many steps to look ahead (1-3)
        format: 'compact' or 'verbose'
    
    Returns:
        String like "THINK: U:wall D:safe(d=5,f=12) L:safe(d=7,f=8) R:death | BEST:D | ACT:1"
    """
    head = snake_positions[0]
    
    # Analyze all four directions
    action_analyses = []
    
    for action in range(4):
        outcomes = multi_step_lookahead(
            head, action, snake_positions, food_pos, width, height, depth=lookahead_depth
        )
        
        dir_name = DIRECTION_NAMES[action]
        
        if lookahead_depth == 1:
            # Single step analysis
            outcome = outcomes[0]
            if outcome['outcome'] == 'wall':
                action_analyses.append(f"{dir_name}:wall")
            elif outcome['outcome'] == 'death':
                action_analyses.append(f"{dir_name}:death")
            elif outcome['outcome'] == 'apple':
                action_analyses.append(f"{dir_name}:apple(f={outcome['freedom']})")
            else:  # safe
                action_analyses.append(f"{dir_name}:safe(d={outcome['distance']},f={outcome['freedom']})")
        else:
            # Multi-step analysis
            steps_desc = []
            for i, outcome in enumerate(outcomes):
                if outcome['outcome'] == 'wall':
                    steps_desc.append(f"{i+1}:wall")
                    break
                elif outcome['outcome'] == 'death':
                    steps_desc.append(f"{i+1}:death")
                    break
                elif outcome['outcome'] == 'apple':
                    steps_desc.append(f"{i+1}:apple")
                    break
                else:
                    steps_desc.append(f"{i+1}:safe")
            
            # Final outcome with distance/freedom
            final = outcomes[-1]
            steps_str = "->".join(steps_desc)
            action_analyses.append(f"{dir_name}:[{steps_str}](d={final['distance']},f={final['freedom']})")
    
    # Format the reasoning text
    if format == 'compact':
        actions_str = " ".join(action_analyses)
        best_str = DIRECTION_NAMES[expert_action]
        return f"THINK: {actions_str} | BEST:{best_str} | ACT:{expert_action}"
    else:  # verbose
        actions_str = "\n  ".join(action_analyses)
        best_str = DIRECTION_FULL[expert_action]
        return f"THINK:\n  {actions_str}\nBEST: {best_str}\nACT: {expert_action}"


def tokenize_reasoning(reasoning_text: str, vocab_size: int = 128) -> np.ndarray:
    """
    Convert reasoning text to token IDs.
    
    Uses a simple character-based tokenization with ASCII mapping.
    For a production system, you'd want a proper tokenizer/vocabulary.
    
    Args:
        reasoning_text: The DSL reasoning string
        vocab_size: Size of vocabulary (default 128 for ASCII)
    
    Returns:
        tokens: Array of token IDs
    """
    # Simple ASCII-based tokenization
    tokens = [min(ord(c), vocab_size - 1) for c in reasoning_text]
    return np.array(tokens, dtype=np.int32)


def reasoning_to_embeddings(
    reasoning_text: str,
    d_model: int,
    max_length: int = 128,
    vocab_size: int = 128
) -> np.ndarray:
    """
    Convert reasoning text to a sequence of embedding indices.
    Pads or truncates to max_length.
    
    Args:
        reasoning_text: The DSL reasoning string
        d_model: Model dimension (not used in tokenization, but for compatibility)
        max_length: Maximum sequence length
        vocab_size: Vocabulary size
    
    Returns:
        tokens: (max_length,) array of token IDs, padded with 0s
    """
    tokens = tokenize_reasoning(reasoning_text, vocab_size)
    
    # Pad or truncate
    if len(tokens) < max_length:
        tokens = np.pad(tokens, (0, max_length - len(tokens)), constant_values=0)
    else:
        tokens = tokens[:max_length]
    
    return tokens


# Example usage and testing
if __name__ == "__main__":
    # Example game state
    snake_positions = [(10, 10), (10, 9), (10, 8), (9, 8)]
    food_pos = (8, 12)
    width, height = 20, 20
    expert_action = 1  # Right
    
    # Generate reasoning text
    reasoning_compact = generate_reasoning_text(
        snake_positions, food_pos, width, height, expert_action,
        lookahead_depth=1, format='compact'
    )
    print("Compact format (depth=1):")
    print(reasoning_compact)
    print()
    
    reasoning_multi = generate_reasoning_text(
        snake_positions, food_pos, width, height, expert_action,
        lookahead_depth=3, format='compact'
    )
    print("Compact format (depth=3):")
    print(reasoning_multi)
    print()
    
    reasoning_verbose = generate_reasoning_text(
        snake_positions, food_pos, width, height, expert_action,
        lookahead_depth=1, format='verbose'
    )
    print("Verbose format (depth=1):")
    print(reasoning_verbose)
    print()
    
    # Tokenization
    tokens = reasoning_to_embeddings(reasoning_compact, d_model=64, max_length=128)
    print(f"Tokenized (first 50 tokens): {tokens[:50]}")
    print(f"Total tokens: {len(tokens)}, Non-zero: {np.count_nonzero(tokens)}")
