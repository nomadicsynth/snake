"""
Configuration for Snake JAX environment
"""

from typing import NamedTuple


class EnvConfig(NamedTuple):
    """Configuration for Snake environment"""
    
    # Grid dimensions
    width: int = 10
    height: int = 10
    
    # Snake parameters
    max_snake_length: int = 100  # Maximum possible snake length (width * height)
    initial_snake_length: int = 3
    
    # Reward shaping
    apple_reward: float = 10.0
    death_penalty: float = -10.0
    step_penalty: float = -0.01
    
    # Episode constraints
    max_steps: int = 500
    
    # Actions: 0=up, 1=right, 2=down, 3=left
    num_actions: int = 4
    
    def __post_init__(self):
        """Validate configuration"""
        # Use object.__setattr__ because NamedTuple is immutable
        if self.max_snake_length < self.width * self.height:
            object.__setattr__(self, 'max_snake_length', self.width * self.height)
