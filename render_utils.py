"""
Rendering utilities for Snake game states.

Provides ASCII and graphical rendering functions for visualizing game states.
"""

import numpy as np
import matplotlib.pyplot as plt
from pretrain_utils import get_positions_from_state, is_padded_cell, PADDING_VALUE


def render_ascii(state: np.ndarray, action: int, action_names: bool = False, reasoning: str = None):
    """
    Render game state in ASCII format.
    
    Args:
        state: (H, W, 3) state array
        action: Action integer (0-3)
        action_names: If True, show action name instead of number
        reasoning: Optional reasoning string to display
    """
    try:
        snake_positions, food_pos = get_positions_from_state(state)
    except Exception as e:
        print(f"Error parsing state: {e}")
        return
    
    height, width, _ = state.shape
    
    # Create grid
    grid = [['. ' for _ in range(width)] for _ in range(height)]
    
    # Mark padded cells
    for y in range(height):
        for x in range(width):
            if is_padded_cell(state, y, x):
                grid[y][x] = '⬛'  # Black square for padded cells
    
    # Place food
    food_y, food_x = food_pos
    if 0 <= food_y < height and 0 <= food_x < width:
        grid[food_y][food_x] = '🍎'
    
    # Place snake (head is first in list)
    for i, (snake_y, snake_x) in enumerate(snake_positions):
        if 0 <= snake_y < height and 0 <= snake_x < width:
            if i == 0:
                grid[snake_y][snake_x] = '🟢'  # Head
            else:
                grid[snake_y][snake_x] = '🟩'  # Body
    
    # Action mapping
    action_map = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}
    action_str = action_map[action] if action_names else str(action)
    
    # Print grid with borders
    border = "───" * width
    print(f"\n┌{border}┐")
    for row in grid:
        print('│' + ''.join(f' {cell}' for cell in row) + '│')
    print(f"└{border}┘")
    print(f"Action: {action_str} ({action_map[action]})")
    print(f"Snake length: {len(snake_positions)}")
    if reasoning is not None:
        print(f"Reasoning: {reasoning}")
    print()


def render_graphical(state: np.ndarray, action: int, action_names: bool = False, reasoning: str = None, sample_number: int = None):
    """
    Render game state using matplotlib.
    
    Args:
        state: (H, W, 3) state array
        action: Action integer (0-3)
        action_names: If True, show action name instead of number
        reasoning: Optional reasoning string
        sample_number: Optional sample number for title
    """
    try:
        snake_positions, food_pos = get_positions_from_state(state)
    except Exception as e:
        print(f"Error parsing state: {e}")
        return
    
    height, width, _ = state.shape
    
    # Create RGB grid
    grid = np.zeros((height, width, 3), dtype=np.float32)
    grid[:] = [0.1, 0.1, 0.1]  # Dark background
    
    # Mark padded cells with a distinct color (dark gray/blue)
    for y in range(height):
        for x in range(width):
            if is_padded_cell(state, y, x):
                grid[y, x] = [0.2, 0.2, 0.4]  # Dark blue-gray for padded cells
    
    # Place food (red)
    food_y, food_x = food_pos
    if 0 <= food_y < height and 0 <= food_x < width:
        grid[food_y, food_x] = [1.0, 0.0, 0.0]  # Red
    
    # Place snake (green, head brighter)
    for i, (snake_y, snake_x) in enumerate(snake_positions):
        if 0 <= snake_y < height and 0 <= snake_x < width:
            if i == 0:
                grid[snake_y, snake_x] = [0.0, 1.0, 0.0]  # Bright green head
            else:
                grid[snake_y, snake_x] = [0.0, 0.6, 0.0]  # Darker green body
    
    # Action mapping
    action_map = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}
    action_str = f"{str(action)} ({action_map[action]})"
    reasoning_str = f"Reasoning: {reasoning}" if reasoning is not None else ""
    
    # Display
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot()
    ax.imshow(grid, interpolation='nearest')
    title_parts = []
    if sample_number is not None:
        title_parts.append(f"Sample {sample_number}")
    title_parts.append(f"Action: {action_str}")
    title_parts.append(f"Snake length: {len(snake_positions)}")
    fig.suptitle(" | ".join(title_parts), fontsize=10)
    if reasoning is not None:
        ax.set_title(reasoning_str, fontsize=8, loc='center', pad=10)
    ax.axis('off')
    fig.tight_layout()
    plt.show()


def render_state_for_video(state: np.ndarray, moves: int = None, score: int = None) -> np.ndarray:
    """
    Render game state as RGB array for video saving.
    
    Args:
        state: (H, W, 3) state array
        moves: Optional move count to display
        score: Optional score to display
        
    Returns:
        frame: RGB numpy array (H, W, 3) with values in [0, 255]
    """
    try:
        snake_positions, food_pos = get_positions_from_state(state)
    except Exception as e:
        print(f"Error parsing state: {e}")
        return None
    
    height, width, _ = state.shape
    
    # Create RGB grid
    grid = np.zeros((height, width, 3), dtype=np.float32)
    grid[:] = [0.9, 0.9, 0.9]  # Light background
    
    # Mark padded cells with a distinct color (light gray/blue)
    for y in range(height):
        for x in range(width):
            if is_padded_cell(state, y, x):
                grid[y, x] = [0.7, 0.7, 0.8]  # Light blue-gray for padded cells
    
    # Place food (red)
    food_y, food_x = food_pos
    if 0 <= food_y < height and 0 <= food_x < width:
        grid[food_y, food_x] = [1.0, 0.0, 0.0]  # Red
    
    # Place snake (green, head brighter)
    for i, (snake_y, snake_x) in enumerate(snake_positions):
        if 0 <= snake_y < height and 0 <= snake_x < width:
            if i == 0:
                grid[snake_y, snake_x] = [0.0, 0.8, 0.0]  # Head: green
            else:
                grid[snake_y, snake_x] = [0.0, 0.5, 0.0]  # Body: darker green
    
    # Convert to matplotlib figure for text overlay
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111)
    ax.imshow(grid, interpolation='nearest')
    
    # Add title with stats
    title_parts = [f"Length: {len(snake_positions)}"]
    if score is not None:
        title_parts.append(f"Score: {score}")
    if moves is not None:
        title_parts.append(f"Moves: {moves}")
    ax.set_title(" | ".join(title_parts))
    ax.axis('off')
    
    # Render to numpy array
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width_px, height_px = fig.get_size_inches() * fig.get_dpi()
    width_px, height_px = int(width_px), int(height_px)
    buf = canvas.buffer_rgba()
    frame = np.asarray(buf)[:, :, :3].copy()  # Drop alpha channel
    frame = (frame * 255).astype(np.uint8)  # Convert to [0, 255]
    
    plt.close(fig)
    
    return frame

