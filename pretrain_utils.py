"""
Utilities for generating expert demonstrations and labels for Snake pretraining.
"""

from collections import deque
import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def get_neighbors(pos: Tuple[int, int], width: int, height: int) -> List[Tuple[int, int]]:
    """Get valid neighboring positions (4-connected)."""
    x, y = pos
    neighbors = []
    
    # Up, Right, Down, Left
    for dx, dy in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < height and 0 <= ny < width:
            neighbors.append((nx, ny))
    
    return neighbors


def action_from_move(current: Tuple[int, int], next_pos: Tuple[int, int]) -> int:
    """Convert position change to action index."""
    dx = next_pos[0] - current[0]
    dy = next_pos[1] - current[1]
    
    # Up=0, Right=1, Down=2, Left=3
    if dx == -1 and dy == 0:
        return 0  # Up
    elif dx == 0 and dy == 1:
        return 1  # Right
    elif dx == 1 and dy == 0:
        return 2  # Down
    elif dx == 0 and dy == -1:
        return 3  # Left
    else:
        raise ValueError(f"Invalid move from {current} to {next_pos}")


def astar_to_food(
    snake_positions: List[Tuple[int, int]], 
    food_pos: Tuple[int, int],
    width: int,
    height: int
) -> Optional[List[Tuple[int, int]]]:
    """
    A* pathfinding from snake head to food.
    Returns path as list of positions, or None if no path exists.
    """
    head = snake_positions[0]
    occupied = set(snake_positions[1:])  # Don't count head
    
    # A* algorithm
    frontier = [(0, head)]
    came_from = {head: None}
    cost_so_far = {head: 0}
    
    while frontier:
        _, current = heapq.heappop(frontier)
        
        if current == food_pos:
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return list(reversed(path))
        
        for next_pos in get_neighbors(current, width, height):
            # Skip if occupied by snake body
            if next_pos in occupied:
                continue
            
            new_cost = cost_so_far[current] + 1
            
            if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                cost_so_far[next_pos] = new_cost
                priority = new_cost + manhattan_distance(next_pos, food_pos)
                heapq.heappush(frontier, (priority, next_pos))
                came_from[next_pos] = current
    
    return None  # No path found


def get_safe_actions(
    snake_positions: List[Tuple[int, int]],
    width: int,
    height: int
) -> List[int]:
    """Return list of actions that don't immediately kill the snake."""
    head = snake_positions[0]
    occupied = set(snake_positions)
    safe_actions = []
    
    for action in range(4):
        # Calculate next position
        dx, dy = [(-1, 0), (0, 1), (1, 0), (0, -1)][action]
        next_pos = (head[0] + dx, head[1] + dy)
        
        # Check bounds
        if not (0 <= next_pos[0] < height and 0 <= next_pos[1] < width):
            continue
        
        # Check self-collision (allow tail position as it will move)
        if next_pos in occupied and next_pos != snake_positions[-1]:
            continue
        
        safe_actions.append(action)
    
    return safe_actions


def count_reachable_cells(
    start: Tuple[int, int],
    snake_positions: List[Tuple[int, int]],
    width: int,
    height: int
) -> int:
    """Count how many cells are reachable from start position (flood fill)."""
    occupied = set(snake_positions)
    visited = {start}
    queue = deque([start])
    
    while queue:
        current = queue.popleft()
        
        for next_pos in get_neighbors(current, width, height):
            if next_pos not in visited and next_pos not in occupied:
                visited.add(next_pos)
                queue.append(next_pos)
    
    return len(visited)


def get_expert_action_astar(
    snake_positions: List[Tuple[int, int]],
    food_pos: Tuple[int, int],
    width: int,
    height: int
) -> Optional[int]:
    """
    Get expert action using A* pathfinding.
    Returns None if no safe path to food exists.
    """
    path = astar_to_food(snake_positions, food_pos, width, height)
    
    if path and len(path) > 1:
        return action_from_move(path[0], path[1])
    
    return None


def get_expert_action_safe(
    snake_positions: List[Tuple[int, int]],
    food_pos: Tuple[int, int],
    width: int,
    height: int
) -> Optional[int]:
    """
    Get expert action with safety heuristics.
    Prefers actions that:
    1. Don't die
    2. Move toward food
    3. Maximize future freedom (reachable cells)
    """
    safe_actions = get_safe_actions(snake_positions, width, height)
    
    if not safe_actions:
        return None
    
    head = snake_positions[0]
    
    # Score each safe action
    action_scores = []
    
    for action in safe_actions:
        dx, dy = [(-1, 0), (0, 1), (1, 0), (0, -1)][action]
        next_pos = (head[0] + dx, head[1] + dy)
        
        # Distance to food (lower is better)
        dist_to_food = manhattan_distance(next_pos, food_pos)
        
        # Future freedom (higher is better)
        # Simulate snake at next position
        next_snake = [next_pos] + snake_positions[:-1]
        freedom = count_reachable_cells(next_pos, next_snake, width, height)
        
        # Combined score (weights tuned empirically)
        score = -dist_to_food + 0.3 * freedom
        action_scores.append((score, action))
    
    # Return action with highest score
    action_scores.sort(reverse=True)
    return action_scores[0][1]


def get_action_distribution(
    snake_positions: List[Tuple[int, int]],
    food_pos: Tuple[int, int],
    width: int,
    height: int,
    temperature: float = 0.5,
    use_astar: bool = True
) -> np.ndarray:
    """
    Generate soft label distribution over actions.
    
    Args:
        snake_positions: List of (x, y) positions, head first
        food_pos: Food position
        width, height: Grid dimensions
        temperature: Softmax temperature (lower = more peaked)
        use_astar: Whether to use A* or heuristic scoring
    
    Returns:
        Array of shape (4,) with action probabilities
    """
    head = snake_positions[0]
    safe_actions = get_safe_actions(snake_positions, width, height)
    
    # Initialize scores (very negative for unsafe actions)
    scores = np.full(4, -100.0)
    
    if not safe_actions:
        # No safe actions - uniform over all (will die anyway)
        return np.ones(4) / 4.0
    
    # Try A* first if requested
    if use_astar:
        expert_action = get_expert_action_astar(snake_positions, food_pos, width, height)
        if expert_action is not None:
            # Hard label for A* solution
            probs = np.zeros(4)
            probs[expert_action] = 1.0
            return probs
    
    # Fall back to heuristic scoring
    for action in safe_actions:
        dx, dy = [(-1, 0), (0, 1), (1, 0), (0, -1)][action]
        next_pos = (head[0] + dx, head[1] + dy)
        
        # Distance to food (negative reward)
        dist_to_food = manhattan_distance(next_pos, food_pos)
        
        # Freedom (positive reward)
        next_snake = [next_pos] + snake_positions[:-1]
        freedom = count_reachable_cells(next_pos, next_snake, width, height)
        
        # Combined score
        scores[action] = -dist_to_food + 0.3 * freedom
    
    # Convert to probabilities with temperature
    scores = scores / temperature
    exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
    probs = exp_scores / np.sum(exp_scores)
    
    return probs


def state_from_positions(
    snake_positions: List[Tuple[int, int]],
    food_pos: Tuple[int, int],
    width: int,
    height: int
) -> np.ndarray:
    """
    Convert game positions to state array using RGB encoding.
    Returns array of shape (height, width, 3) with RGB channels:
    - Empty cells: (0, 0, 0) - Black
    - Snake: (0, 255, 0) - Green
    - Food: (255, 0, 0) - Red
    
    Values are normalized to [0, 1] range for neural network input.
    """
    state = np.zeros((height, width, 3), dtype=np.float32)
    
    # Mark snake with head-to-tail green gradient (head brightest = 1.0)
    length = len(snake_positions)
    if length == 1:
        y, x = snake_positions[0]
        state[y, x, 0] = 0.0
        state[y, x, 1] = 1.0
        state[y, x, 2] = 0.0
    elif length > 1:
        # Linearly decrease green from head (index 0) toward tail (index length-1)
        # Tail floor kept > 0 for reliable recovery
        tail_floor = 0.2
        for idx, (y, x) in enumerate(snake_positions):
            t = idx / (length - 1)
            g = (1.0 - t) * (1.0 - tail_floor) + tail_floor
            state[y, x, 0] = 0.0
            state[y, x, 1] = float(g)
            state[y, x, 2] = 0.0
    
    # Food in red
    state[food_pos[0], food_pos[1], 0] = 1.0   # R
    state[food_pos[0], food_pos[1], 1] = 0.0   # G
    state[food_pos[0], food_pos[1], 2] = 0.0   # B
    
    return state


def augment_state_action(
    state: np.ndarray,
    action: int,
    augmentation: str
) -> Tuple[np.ndarray, int]:
    """
    Apply geometric augmentation to state and corresponding action.
    
    Args:
        state: (H, W, 3) state array
        action: Action index (0=Up, 1=Right, 2=Down, 3=Left)
        augmentation: One of ['identity', 'rot90', 'rot180', 'rot270', 
                              'flip_h', 'flip_v', 'flip_h_rot90', 'flip_v_rot90']
    
    Returns:
        Augmented state and transformed action
    """
    aug_state = state.copy()
    aug_action = action
    
    if augmentation == 'identity':
        pass
    
    elif augmentation == 'rot90':
        aug_state = np.rot90(aug_state, k=1, axes=(0, 1))
        # Action mapping: Up->Left, Right->Up, Down->Right, Left->Down
        aug_action = (action - 1) % 4
    
    elif augmentation == 'rot180':
        aug_state = np.rot90(aug_state, k=2, axes=(0, 1))
        # Action mapping: Up->Down, Right->Left, Down->Up, Left->Right
        aug_action = (action + 2) % 4
    
    elif augmentation == 'rot270':
        aug_state = np.rot90(aug_state, k=3, axes=(0, 1))
        # Action mapping: Up->Right, Right->Down, Down->Left, Left->Up
        aug_action = (action + 1) % 4
    
    elif augmentation == 'flip_h':
        aug_state = np.flip(aug_state, axis=1)
        # Horizontal flip: Left<->Right
        if action == 1:  # Right
            aug_action = 3
        elif action == 3:  # Left
            aug_action = 1
    
    elif augmentation == 'flip_v':
        aug_state = np.flip(aug_state, axis=0)
        # Vertical flip: Up<->Down
        if action == 0:  # Up
            aug_action = 2
        elif action == 2:  # Down
            aug_action = 0
    
    elif augmentation == 'flip_h_rot90':
        aug_state = np.flip(aug_state, axis=1)
        aug_state = np.rot90(aug_state, k=1, axes=(0, 1))
        # Combined transformation
        if action == 0:  # Up
            aug_action = 3  # Left
        elif action == 1:  # Right
            aug_action = 2  # Down
        elif action == 2:  # Down
            aug_action = 1  # Right
        elif action == 3:  # Left
            aug_action = 0  # Up
    
    elif augmentation == 'flip_v_rot90':
        aug_state = np.flip(aug_state, axis=0)
        aug_state = np.rot90(aug_state, k=1, axes=(0, 1))
        # Combined transformation
        if action == 0:  # Up
            aug_action = 1  # Right
        elif action == 1:  # Right
            aug_action = 0  # Up
        elif action == 2:  # Down
            aug_action = 3  # Left
        elif action == 3:  # Left
            aug_action = 2  # Down
    
    return aug_state, aug_action


# All augmentation types for 8-fold symmetry
AUGMENTATIONS = [
    'identity', 'rot90', 'rot180', 'rot270',
    'flip_h', 'flip_v', 'flip_h_rot90', 'flip_v_rot90'
]


def get_positions_from_state(
    state: np.ndarray
) -> Tuple[List[Tuple[int, int]], Tuple[int, int]]:
    """
    Extract snake positions and food position from state array.
    Inverse of state_from_positions.
    
    Args:
        state: (H, W, 3) state array
    
    Returns:
        snake_positions: List of (row, col) tuples
        food_pos: (row, col) tuple
    """
    height, width, _ = state.shape
    
    # Extract food position (red channel)
    food_positions = np.argwhere(state[:, :, 0] > 0)
    if len(food_positions) == 0:
        raise ValueError("No food found in state")
    food_pos = tuple(food_positions[0])
    
    # Extract snake positions (green channel has values)
    # Higher green value = closer to head
    snake_cells = []
    for y in range(height):
        for x in range(width):
            if state[y, x, 1] > 0:
                snake_cells.append(((y, x), state[y, x, 1]))
    
    # Sort by green value (descending) to get head first
    snake_cells.sort(key=lambda x: x[1], reverse=True)
    snake_positions = [pos for pos, _ in snake_cells]
    
    return snake_positions, food_pos
