"""
GPU-Native Snake RL Environment in JAX

Pure functional implementation for massive parallelization.
"""

from snake_jax.env import SnakeEnv, SnakeState
from snake_jax.config import EnvConfig

__all__ = ["SnakeEnv", "SnakeState", "EnvConfig"]
