"""
Snake Game Environment

A pure Python implementation of the Snake game environment.
Matches the JAX environment behavior for consistency.
"""

import random
import numpy as np


# Default configuration
DEFAULT_WIDTH = 10
DEFAULT_HEIGHT = 10
DEFAULT_MAX_STEPS = 500


class SnakeGame:
    """
    Snake game environment.
    
    The environment uses RGB encoding for state representation:
    - Empty cells: (0, 0, 0) - Black
    - Snake: (0, 1, 0) - Green
    - Food: (1, 0, 0) - Red
    
    Actions:
    - 0: Up
    - 1: Right
    - 2: Down
    - 3: Left
    """
    
    def __init__(
        self,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        step_penalty: float = -0.01,
        wall_collision: bool = True,
        apple_reward: float = 10.0,
        death_penalty: float = -10.0,
        num_apples: int = 1,
        initial_snake_length: int = 3,
        max_steps: int = DEFAULT_MAX_STEPS,
    ):
        """
        Initialize Snake game environment.
        
        Args:
            width: Grid width
            height: Grid height
            step_penalty: Reward per step (negative to encourage efficiency)
            wall_collision: If True, hitting walls ends episode
            apple_reward: Reward for eating an apple
            death_penalty: Penalty for dying
            num_apples: Number of apples on the board simultaneously
            initial_snake_length: Initial length of snake
            max_steps: Maximum steps per episode
        """
        self.width = width
        self.height = height
        self.step_penalty = step_penalty
        self.wall_collision = wall_collision
        self.apple_reward = apple_reward
        self.death_penalty = death_penalty
        self.initial_snake_length = initial_snake_length
        self.max_steps = max_steps
        # Support multiple concurrent apples. Keep backward-compatible `self.food`
        # as the current target (nearest apple), and store all apples in `self.foods`.
        self.num_apples = max(1, int(num_apples))
        self.foods = []  # list[(x,y)]
        self.food = None  # nearest apple for compatibility
        self.ate_last_step = False  # precise, per-step flag for apple consumption
        self.step_count = 0
        self.reset()

    def reset(self):
        """
        Reset environment to initial state.
        
        Returns:
            state: Initial state array (H, W, 3)
        """
        # Initialize snake with initial_snake_length segments, moving right
        center_x = self.width // 2
        center_y = self.height // 2
        self.snake = []
        for i in range(self.initial_snake_length):
            self.snake.append((center_x - i, center_y))
        self.direction = (1, 0)  # Start moving right
        self.foods = []
        self._ensure_apples()
        self._update_current_food_target()
        self.score = 0
        self.game_over = False
        self.ate_last_step = False
        self.step_count = 0
        return self._get_state()

    def _place_food(self):
        """Place a single apple in a free cell and return its position."""
        while True:
            food = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if food not in self.snake and food not in self.foods:
                return food

    def _ensure_apples(self):
        """Ensure there are exactly `self.num_apples` apples on the board."""
        # Add apples if fewer than desired
        while len(self.foods) < self.num_apples:
            self.foods.append(self._place_food())
        # Remove extra apples if more than desired
        while len(self.foods) > self.num_apples:
            self.foods.pop()

    def _update_current_food_target(self):
        """Set `self.food` to the nearest apple (for backward compatibility)."""
        if not self.foods:
            self.food = None
            return
        head_x, head_y = self.snake[0]
        def manhattan(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        self.food = min(self.foods, key=lambda f: manhattan((head_x, head_y), f))

    def _move_snake(self, direction):
        """
        Move snake in given direction and update game state.
        
        Args:
            direction: Direction tuple (dx, dy)
        
        Returns:
            reward: Reward for this step
        """
        # Prevent 180-degree turns (moving backwards into self)
        # If new direction is opposite of current, keep current direction
        # disabled for now because it feels like cheating. shouldn't the model be
        # expected to learn not to move backwards?
        # if direction == (-self.direction[0], -self.direction[1]):
        #     direction = self.direction
        
        head_x, head_y = self.snake[0]
        dir_x, dir_y = direction
        new_x = head_x + dir_x
        new_y = head_y + dir_y

        # Wall collision or wrap-around
        if self.wall_collision:
            if new_x < 0 or new_x >= self.width or new_y < 0 or new_y >= self.height:
                self.game_over = True
                return float(self.death_penalty)
            new_head = (new_x, new_y)
        else:
            new_head = (new_x % self.width, new_y % self.height)

        # Check self collision (exclude tail position since it will move)
        # Only check if new_head matches body positions (not tail)
        occupied = set(self.snake[:-1])  # Exclude tail
        if new_head in occupied:
            self.game_over = True
            return float(self.death_penalty)  # Penalty for hitting itself
        
        # Update direction
        self.direction = direction

        self.snake.insert(0, new_head)

        reward = self.step_penalty  # small penalty each step to discourage inefficient/aimless behavior
        self.ate_last_step = False
        # Eating logic with multiple apples
        if new_head in self.foods:
            self.score += 1
            reward = float(self.apple_reward)
            # keep growth: do not pop tail
            # remove the eaten apple and spawn a new one to maintain count
            try:
                self.foods.remove(new_head)
            except ValueError:
                pass
            self._ensure_apples()
            self.ate_last_step = True
        else:
            # normal move: pop tail
            self.snake.pop()

        # Update current target apple for compatibility
        self._update_current_food_target()
        
        # Check max steps
        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.game_over = True

        return reward

    def _get_state(self):
        """
        Generate observation from state using RGB encoding (matching pretrain_utils).
        
        Returns:
            Grid (H, W, 3) where channels are RGB:
            - Empty cells: (0, 0, 0) - Black
            - Snake: (0, g, 0) - Green with head-to-tail gradient (head=1.0, tail=0.2)
            - Food: (1, 0, 0) - Red
            
            Values are in [0, 1] range (normalized).
        """
        state = np.zeros((self.height, self.width, 3), dtype=np.float32)
        
        # Mark food positions in red (R=1, G=0, B=0)
        for food_x, food_y in self.foods:
            if 0 <= food_y < self.height and 0 <= food_x < self.width:
                state[food_y, food_x, 0] = 1.0  # Red channel
        
        # Mark snake body with head-to-tail green gradient (head brightest = 1.0)
        length = len(self.snake)
        if length == 1:
            # Single segment: full brightness
            snake_x, snake_y = self.snake[0]
            if 0 <= snake_y < self.height and 0 <= snake_x < self.width:
                state[snake_y, snake_x, 1] = 1.0  # Green channel
        elif length > 1:
            # Linearly decrease green from head (index 0) toward tail (index length-1)
            # Tail floor kept > 0 for reliable recovery
            tail_floor = 0.2
            for idx, (snake_x, snake_y) in enumerate(self.snake):
                if 0 <= snake_y < self.height and 0 <= snake_x < self.width:
                    t = idx / (length - 1)  # Normalized position (0 at head, 1 at tail)
                    g = (1.0 - t) * (1.0 - tail_floor) + tail_floor
                    state[snake_y, snake_x, 1] = float(g)  # Green channel
        
        # Empty cells remain (0, 0, 0) - Black
        return state

    def step(self, action):
        """
        Step the environment.
        
        Args:
            action: Action to take (0=up, 1=right, 2=down, 3=left)
        
        Returns:
            next_state: Next state (H, W, 3)
            reward: Scalar reward
            done: Whether episode is terminal
        """
        # Map action to direction (matching JAX environment)
        # 0=up, 1=right, 2=down, 3=left
        action_to_direction = {
            0: (0, -1),   # up
            1: (1, 0),    # right
            2: (0, 1),    # down
            3: (-1, 0)    # left
        }
        direction = action_to_direction[action]
        reward = self._move_snake(direction)
        next_state = self._get_state()
        done = self.game_over
        return next_state, reward, done

    def _display(self):
        """Display the current game state in ASCII format (for debugging)."""
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) in self.snake:
                    print("O", end=" ")
                elif (x, y) in self.foods:
                    print("X", end=" ")
                else:
                    print(".", end=" ")
            print()

    # --- Runtime configuration APIs ---
    def set_num_apples(self, n: int):
        """Update the number of concurrent apples and adjust the board immediately."""
        self.num_apples = max(1, int(n))
        self._ensure_apples()
        self._update_current_food_target()
