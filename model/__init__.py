"""
Snake RL Models

Contains various model architectures for the Snake game:
- TransformerPolicy: Standard transformer-based policy (with optional RSM)
- SnakeWorldEqM: Equilibrium Matching world model for joint state-action prediction
"""

from .model_pytorch import (
    TransformerPolicy,
    SnakeTransformerConfig,
    PositionalEncoding2D,
    CNNEncoder,
)

from .model_eqm import (
    SnakeWorldEqM,
    SnakeWorldEqMConfig,
    EqMGradientField,
)

__all__ = [
    # Standard transformer policy
    "TransformerPolicy",
    "SnakeTransformerConfig",
    "PositionalEncoding2D",
    "CNNEncoder",
    
    # EqM world model
    "SnakeWorldEqM",
    "SnakeWorldEqMConfig",
    "EqMGradientField",
]
