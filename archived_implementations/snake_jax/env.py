"""
Pure JAX Snake Environment

Fully vectorizable, GPU-accelerated Snake game.
All operations are pure functions for JIT compilation.
"""

from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
from jax import random
from functools import partial

from snake_jax.config import EnvConfig


class SnakeState(NamedTuple):
    """Immutable state for a single Snake environment"""
    
    # Snake body positions: (max_length, 2) where each row is (x, y)
    # Only the first `snake_length` positions are valid
    snake_body: jnp.ndarray  # (max_length, 2) int32
    snake_length: int  # Current length of snake
    direction: jnp.ndarray   # (2,) int32 - current direction (dx, dy)
    
    # Food position
    food_pos: jnp.ndarray    # (2,) int32 - (x, y)
    
    # Episode state
    score: int
    done: bool
    step_count: int

    # RNG state for food placement
    rng: jax.random.PRNGKey


class SnakeEnv:
    """GPU-accelerated Snake environment"""
    
    def __init__(self, config: EnvConfig = None):
        self.config = config or EnvConfig()
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: jax.random.PRNGKey) -> SnakeState:
        """Reset environment to initial state"""
        config = self.config
        
        # Initialize snake in the center, moving right
        center_x = config.width // 2
        center_y = config.height // 2
        
        # Create snake body array (padded to max_length)
        snake_body = jnp.zeros((config.max_snake_length, 2), dtype=jnp.int32)
        
        # Initial snake: 3 segments in a line
        for i in range(config.initial_snake_length):
            snake_body = snake_body.at[i].set(jnp.array([center_x - i, center_y]))
        
        # Initial direction: right
        direction = jnp.array([1, 0], dtype=jnp.int32)
        
        # Place initial food
        rng, food_rng = random.split(rng)
        food_pos = self._place_food(food_rng, snake_body, config.initial_snake_length)
        
        return SnakeState(
            snake_body=snake_body,
            snake_length=jnp.int32(config.initial_snake_length),
            direction=direction,
            food_pos=food_pos,
            score=jnp.int32(0),
            done=jnp.bool_(False),
            step_count=jnp.int32(0),
            rng=rng
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: SnakeState, action: int) -> Tuple[SnakeState, jnp.ndarray, bool, dict]:
        """
        Step the environment
        
        Args:
            state: Current state
            action: Action to take (0=up, 1=right, 2=down, 3=left)
        
        Returns:
            new_state: Next state
            obs: Observation (H, W, 3)
            reward: Scalar reward
            done: Whether episode is terminal
            info: Dict (empty for JAX compatibility)
        """
        config = self.config
        
        # If already done, return current state with zero reward
        def already_done_fn(state):
            obs = self._get_observation(state)
            return state, obs, jnp.float32(0.0), jnp.bool_(True), {}
        
        # Normal step
        def step_fn(state):
            # Update direction based on action
            new_direction = self._action_to_direction(action)
            
            # Prevent 180-degree turns (moving backwards into self)
            # If new direction is opposite of current, keep current direction
            is_opposite = jnp.all(new_direction == -state.direction)
            direction = jnp.where(is_opposite, state.direction, new_direction)
            
            # Calculate new head position
            head = state.snake_body[0]
            new_head = head + direction
            
            # Check wall collision
            hit_wall = jnp.logical_or(
                jnp.logical_or(new_head[0] < 0, new_head[0] >= config.width),
                jnp.logical_or(new_head[1] < 0, new_head[1] >= config.height)
            )
            
            # Check self collision (check if new_head matches any body position)
            # Check all positions but mask out invalid ones
            matches = jnp.all(state.snake_body == new_head, axis=1)
            valid_mask = jnp.arange(config.max_snake_length) < state.snake_length
            hit_self = jnp.any(matches & valid_mask)
            
            # Determine if snake ate food
            ate_food = jnp.all(new_head == state.food_pos)
            
            # Check if episode should end
            died = jnp.logical_or(hit_wall, hit_self)
            max_steps_reached = state.step_count >= config.max_steps - 1
            done = jnp.logical_or(died, max_steps_reached)
            
            # Calculate reward
            reward = jnp.where(
                died,
                jnp.float32(config.death_penalty),
                jnp.where(
                    ate_food,
                    jnp.float32(config.apple_reward),
                    jnp.float32(config.step_penalty)
                )
            )
            
            # Update snake body
            # If ate food: insert new head, keep all segments
            # If didn't eat: insert new head, remove tail
            new_length = jnp.where(ate_food, state.snake_length + 1, state.snake_length)
            new_length = jnp.minimum(new_length, config.max_snake_length)
            
            # Shift snake body and insert new head
            new_snake_body = jnp.roll(state.snake_body, 1, axis=0)
            new_snake_body = new_snake_body.at[0].set(new_head)
            
            # If didn't eat food, we keep the shifted version (tail removed automatically)
            # If ate food, we already incremented length
            
            # Place new food if eaten
            rng, food_rng = random.split(state.rng)
            new_food_pos = jnp.where(
                ate_food,
                self._place_food(food_rng, new_snake_body, new_length),
                state.food_pos
            )
            
            # Update score
            new_score = jnp.where(ate_food, state.score + 1, state.score)
            
            # Create new state
            new_state = SnakeState(
                snake_body=new_snake_body,
                snake_length=new_length,
                direction=direction,
                food_pos=new_food_pos,
                score=new_score,
                done=done,
                step_count=state.step_count + 1,
                rng=rng
            )
            
            obs = self._get_observation(new_state)
            return new_state, obs, reward, done, {}
        
        # Use jax.lax.cond for branching
        return jax.lax.cond(
            state.done,
            already_done_fn,
            step_fn,
            state
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _action_to_direction(self, action: int) -> jnp.ndarray:
        """Convert action to direction vector"""
        # 0=up, 1=right, 2=down, 3=left
        directions = jnp.array([
            [0, -1],   # up
            [1, 0],    # right
            [0, 1],    # down
            [-1, 0]    # left
        ], dtype=jnp.int32)
        return directions[action]
    
    @partial(jax.jit, static_argnums=(0,))
    def _place_food(self, rng: jax.random.PRNGKey, snake_body: jnp.ndarray, 
                    snake_length: int) -> jnp.ndarray:
        """
        Place food in a random valid position
        
        Note: This is a simplified version that may occasionally place food on snake.
        For true correctness, we'd need a more complex rejection sampling approach.
        For RL training with large grids, collision probability is low enough.
        """
        config = self.config
        
        # Simple random placement (good enough for most cases)
        food_x = random.randint(rng, (), 0, config.width)
        rng, rng2 = random.split(rng)
        food_y = random.randint(rng2, (), 0, config.height)
        
        return jnp.array([food_x, food_y], dtype=jnp.int32)
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: SnakeState) -> jnp.ndarray:
        """
        Generate observation from state using RGB encoding.
        
        Returns:
            Grid (H, W, 3) where channels are RGB:
            - Empty cells: (0, 0, 0) - Black
            - Snake: (0, 1, 0) - Green
            - Food: (1, 0, 0) - Red
            
            Values are in [0, 1] range (normalized).
        """
        config = self.config
        
        # Initialize empty grid (all black)
        obs = jnp.zeros((config.height, config.width, 3), dtype=jnp.float32)
        
        # Mark food position in red (R=1, G=0, B=0)
        food_x, food_y = state.food_pos
        obs = obs.at[food_y, food_x, 0].set(1.0)  # Red channel
        
        # Mark snake body in green (R=0, G=1, B=0)
        # We need to mark all valid snake positions
        def mark_snake_segment(i, obs):
            # Only mark if this is a valid segment
            segment = state.snake_body[i]
            x, y = segment
            
            # Mark this segment as green
            obs_new = obs.at[y, x, 1].set(1.0)  # Green channel
            
            # Only apply if i < snake_length
            return jnp.where(i < state.snake_length, obs_new, obs)
        
        # Use fori_loop to mark all snake segments
        obs = jax.lax.fori_loop(0, config.max_snake_length, mark_snake_segment, obs)
        
        return obs
    
    def observation_space_shape(self) -> Tuple[int, int, int]:
        """Return observation space shape"""
        return (self.config.height, self.config.width, 3)
    
    def action_space_size(self) -> int:
        """Return number of actions"""
        return self.config.num_actions


# Vectorized versions for parallel environments
def make_reset_vmap(env: SnakeEnv, num_envs: int):
    """Create vectorized reset function"""
    @jax.jit
    def reset_vmap(rng: jax.random.PRNGKey):
        rngs = random.split(rng, num_envs)
        return jax.vmap(env.reset)(rngs)
    return reset_vmap


def make_step_vmap(env: SnakeEnv):
    """Create vectorized step function"""
    @jax.jit
    def step_vmap(states, actions):
        return jax.vmap(env.step)(states, actions)
    return step_vmap
