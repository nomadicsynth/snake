"""
Transformer Policy Network in Flax for Snake RL
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for grid-based observations"""
    d_model: int
    
    @nn.compact
    def __call__(self, height: int, width: int) -> jnp.ndarray:
        """
        Generate 2D positional encoding
        
        Returns:
            pos_embed: (height * width, d_model)
        """
        # Create positional encodings
        pos_h = jnp.arange(height)[:, None]  # (H, 1)
        pos_w = jnp.arange(width)[None, :]   # (1, W)
        
        # Expand to match d_model
        d_4 = self.d_model // 4
        
        # Frequency bands
        freqs = jnp.exp(jnp.arange(0, d_4) * -(jnp.log(10000.0) / d_4))
        
        # Height encodings
        pos_h_expanded = pos_h[:, :, None] * freqs[None, None, :]  # (H, 1, d_4)
        sin_h = jnp.sin(pos_h_expanded)
        cos_h = jnp.cos(pos_h_expanded)
        
        # Width encodings
        pos_w_expanded = pos_w[:, :, None] * freqs[None, None, :]  # (1, W, d_4)
        sin_w = jnp.sin(pos_w_expanded)
        cos_w = jnp.cos(pos_w_expanded)
        
        # Concatenate (H, W, d_model)
        pos_encoding = jnp.concatenate([
            sin_h * jnp.ones_like(sin_w),  # Broadcast
            cos_h * jnp.ones_like(cos_w),
            jnp.ones_like(sin_h) * sin_w,
            jnp.ones_like(cos_h) * cos_w,
        ], axis=-1)
        
        # Reshape to (H*W, d_model)
        pos_encoding = pos_encoding.reshape(height * width, self.d_model)
        
        return pos_encoding


class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    d_model: int
    num_heads: int
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x: (batch, seq_len, d_model)
        """
        # Multi-head self-attention
        y = nn.LayerNorm()(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout_rate if training else 0.0
        )(y, y, deterministic=not training)
        x = x + y
        
        # Feed-forward network
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.d_model * 4)(y)
        y = nn.gelu(y)
        y = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(y)
        y = nn.Dense(self.d_model)(y)
        y = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(y)
        x = x + y
        
        return x


class TransformerPolicy(nn.Module):
    """
    Transformer-based Actor-Critic policy for Snake
    
    Processes grid observations with transformer encoder.
    """
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 4
    num_actions: int = 4
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        """
        Args:
            x: (batch, height, width, 3) - grid observation
        
        Returns:
            logits: (batch, num_actions) - policy logits
            value: (batch,) - state value estimate
        """
        batch_size, height, width, channels = x.shape
        seq_len = height * width
        
        # Flatten spatial dims to tokens
        x = x.reshape(batch_size, seq_len, channels)  # (batch, H*W, 3)
        
        # Project input to d_model
        x = nn.Dense(self.d_model, name='input_proj')(x)
        
        # Add positional encoding
        pos_encoding = PositionalEncoding2D(self.d_model)(height, width)
        x = x + pos_encoding[None, :, :]  # Broadcast over batch
        
        # Transformer encoder layers
        for i in range(self.num_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                name=f'transformer_block_{i}'
            )(x, training=training)
        
        # Pool across sequence dimension
        x = nn.LayerNorm()(x)
        x = x.mean(axis=1)  # (batch, d_model)
        
        # Actor head (policy)
        logits = nn.Dense(self.d_model, name='policy_hidden')(x)
        logits = nn.gelu(logits)
        logits = nn.Dense(self.num_actions, name='policy_logits')(logits)
        
        # Critic head (value)
        value = nn.Dense(self.d_model, name='value_hidden')(x)
        value = nn.gelu(value)
        value = nn.Dense(1, name='value_out')(value)
        value = value.squeeze(-1)  # (batch,)
        
        return logits, value
    
    def init_params(self, rng, obs_shape):
        """Initialize network parameters"""
        dummy_obs = jnp.zeros((1, *obs_shape), dtype=jnp.float32)
        return self.init(rng, dummy_obs, training=False)
