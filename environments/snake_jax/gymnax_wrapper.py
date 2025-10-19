"""
Gymnax-style wrapper for our JAX Snake environment

Makes our SnakeEnv compatible with PureJaxRL's expectations.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional
import chex
from flax import struct

from snake_jax.env import SnakeEnv, SnakeState
from snake_jax.config import EnvConfig


@struct.dataclass
class EnvState:
    """Wrapper around SnakeState for Gymnax compatibility"""
    snake_state: SnakeState
    time: int


@struct.dataclass
class EnvParams:
    """Gymnax-style parameters"""
    max_steps_in_episode: int = 500


class SnakeGymnaxWrapper:
    """
    Gymnax-compatible wrapper for Snake environment
    
    Provides the interface expected by PureJaxRL:
    - reset(rng, params) -> obs, state
    - step(rng, state, action, params) -> obs, state, reward, done, info
    - observation_space(params) -> space
    - action_space(params) -> space
    """
    
    def __init__(self, config: EnvConfig = None):
        self.config = config or EnvConfig()
        self.env = SnakeEnv(self.config)
        
        # Store shapes for space definitions
        self._obs_shape = (self.config.height, self.config.width, 3)
        self._action_dim = self.config.num_actions
    
    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_steps_in_episode=self.config.max_steps)
    
    def reset(
        self, 
        rng: chex.PRNGKey,
        params: Optional[EnvParams] = None
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment"""
        if params is None:
            params = self.default_params
        
        # Reset snake
        snake_state = self.env.reset(rng)
        
        # Get observation
        obs = self.env._get_observation(snake_state)
        
        # Wrap in EnvState
        state = EnvState(
            snake_state=snake_state,
            time=0
        )
        
        return obs, state
    
    def step(
        self,
        rng: chex.PRNGKey,
        state: EnvState,
        action: chex.Array,
        params: Optional[EnvParams] = None
    ) -> Tuple[chex.Array, EnvState, chex.Array, chex.Array, dict]:
        """Step environment"""
        if params is None:
            params = self.default_params
        
        # Step snake
        new_snake_state, obs, reward, done, info = self.env.step(
            state.snake_state, action
        )
        
        # Check time limit
        new_time = state.time + 1
        timeout = new_time >= params.max_steps_in_episode
        done = jnp.logical_or(done, timeout)
        
        # Update state
        new_state = EnvState(
            snake_state=new_snake_state,
            time=new_time
        )
        
        # Add episode metrics if done
        info = info or {}
        info['returned_episode'] = done
        info['returned_episode_returns'] = jnp.where(done, reward, 0.0)
        info['returned_episode_lengths'] = jnp.where(done, new_time, 0)
        
        return obs, new_state, reward, done, info
    
    def observation_space(self, params: Optional[EnvParams] = None):
        """Return observation space"""
        # Simple object with shape attribute (Gymnax style)
        class Space:
            def __init__(self, shape):
                self.shape = shape
        return Space(self._obs_shape)
    
    def action_space(self, params: Optional[EnvParams] = None):
        """Return action space"""
        # Simple object with n attribute (discrete space)
        class Space:
            def __init__(self, n):
                self.n = n
        return Space(self._action_dim)
    
    @property
    def name(self) -> str:
        return f"Snake-{self.config.width}x{self.config.height}"
    
    def __repr__(self):
        return f"SnakeGymnaxWrapper({self.config.width}x{self.config.height})"
