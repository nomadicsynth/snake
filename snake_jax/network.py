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
    def __call__(self, x, training: bool = False, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional attention mask (batch, 1, seq_len, seq_len) or (1, 1, seq_len, seq_len)
        Returns:
            x: (batch, seq_len, d_model)
        """
        # Multi-head self-attention
        y = nn.LayerNorm()(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout_rate if training else 0.0
        )(y, y, mask=mask, deterministic=not training)
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


class CNNEncoder(nn.Module):
    """Convolutional encoder for spatial features"""
    features: Sequence[int] = (32, 64)
    kernel_sizes: Sequence[int] = (3, 3)
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        """
        Args:
            x: (batch, height, width, channels) - grid observation
        
        Returns:
            features: (batch, height, width, features[-1]) - extracted features
        """
        for i, (feat, kernel) in enumerate(zip(self.features, self.kernel_sizes)):
            x = nn.Conv(
                features=feat,
                kernel_size=(kernel, kernel),
                padding='SAME',
                name=f'conv_{i}'
            )(x)
            x = nn.gelu(x)
        
        return x


class TransformerPolicy(nn.Module):
    """
    Transformer-based Actor-Critic policy for Snake
    
    Processes grid observations with transformer encoder.
    Optionally uses CNN for feature extraction.
    Optionally processes reasoning tokens (RSM mode) with autoregressive generation.
    
    For RSM models, reasoning tokens are APPENDED after grid tokens to match
    the autoregressive generation order: Grid → CNN → Reasoning → Action
    
    The action head acts as an LM head during reasoning token generation.
    """
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 4
    num_actions: int = 4
    dropout_rate: float = 0.1
    use_cnn: bool = False
    cnn_features: Sequence[int] = (32, 64)
    cnn_mode: str = 'replace'  # 'replace' or 'append'
    use_reasoning: bool = False  # RSM mode
    reasoning_vocab_size: int = 128  # ASCII-based vocabulary
    max_reasoning_length: int = 128  # Maximum reasoning sequence length
    
    def make_causal_mask(self, seq_len: int, reasoning_start_idx: int) -> jnp.ndarray:
        """
        Create causal mask for autoregressive reasoning generation.
        
        Grid tokens can attend to all grid tokens (bidirectional).
        Reasoning tokens can only attend to grid tokens and previous reasoning tokens (causal).
        
        Args:
            seq_len: Total sequence length (grid + reasoning)
            reasoning_start_idx: Index where reasoning tokens start
            
        Returns:
            mask: (1, 1, seq_len, seq_len) attention mask
        """
        # Start with full attention (all True)
        mask = jnp.ones((seq_len, seq_len), dtype=jnp.bool_)
        
        # Apply causal mask only to reasoning tokens
        if reasoning_start_idx < seq_len:
            # Create causal mask for reasoning section
            reasoning_len = seq_len - reasoning_start_idx
            causal_mask = jnp.tril(jnp.ones((reasoning_len, reasoning_len), dtype=jnp.bool_))
            
            # Reasoning tokens can attend to all grid tokens
            grid_attention = jnp.ones((reasoning_len, reasoning_start_idx), dtype=jnp.bool_)
            
            # Combine: reasoning can see [all grid | causal reasoning]
            reasoning_row = jnp.concatenate([grid_attention, causal_mask], axis=1)
            
            # Update the mask for reasoning token rows
            mask = mask.at[reasoning_start_idx:, :].set(reasoning_row)
        
        # Add batch and head dimensions: (1, 1, seq_len, seq_len)
        return mask[None, None, :, :]
    
    @nn.compact
    def __call__(self, x, training: bool = False, reasoning_tokens=None):
        """
        Args:
            x: (batch, height, width, 3) - grid observation
            reasoning_tokens: Optional (batch, seq_len) - reasoning token IDs for RSM mode
        
        Returns:
            logits: (batch, num_actions) - policy logits (or vocab logits in RSM mode)
            value: (batch,) - state value estimate
        """
        batch_size, height, width, channels = x.shape
        seq_len = height * width
        
        # Optional CNN processing
        if self.use_cnn:
            cnn = CNNEncoder(
                features=self.cnn_features,
                kernel_sizes=(3,) * len(self.cnn_features),
                name='cnn_encoder'
            )
            cnn_features = cnn(x, training=training)  # (batch, H, W, cnn_features[-1])
            
            if self.cnn_mode == 'replace':
                # Use CNN features as the only input to transformer
                x = cnn_features
                channels = self.cnn_features[-1]
            elif self.cnn_mode == 'append':
                # Append CNN features to original input
                x = jnp.concatenate([x, cnn_features], axis=-1)  # (batch, H, W, 3 + cnn_features[-1])
                channels = channels + self.cnn_features[-1]
            else:
                raise ValueError(f"Invalid cnn_mode: {self.cnn_mode}. Must be 'replace' or 'append'")
        
        # Flatten spatial dims to tokens
        grid_tokens = x.reshape(batch_size, seq_len, channels)  # (batch, H*W, channels)
        
        # Project grid tokens to d_model
        grid_tokens = nn.Dense(self.d_model, name='input_proj')(grid_tokens)
        
        # Add positional encoding for grid tokens
        pos_encoding = PositionalEncoding2D(self.d_model)(height, width)
        grid_tokens = grid_tokens + pos_encoding[None, :, :]  # Broadcast over batch
        
        # Track where reasoning starts for causal masking
        reasoning_start_idx = seq_len
        
        # Handle reasoning tokens if provided (RSM mode)
        if self.use_reasoning and reasoning_tokens is not None:
            # Embed reasoning tokens
            reasoning_embed = nn.Embed(
                num_embeddings=self.reasoning_vocab_size,
                features=self.d_model,
                name='reasoning_embed'
            )(reasoning_tokens)  # (batch, reasoning_seq_len, d_model)
            
            # Add learned positional encodings for reasoning tokens
            # Use fixed max length and slice as needed
            reasoning_seq_len = reasoning_tokens.shape[1]
            reasoning_pos = self.param(
                'reasoning_pos',
                nn.initializers.normal(stddev=0.02),
                (self.max_reasoning_length, self.d_model)
            )
            # Slice to actual sequence length
            reasoning_embed = reasoning_embed + reasoning_pos[None, :reasoning_seq_len, :]
            
            # Append reasoning tokens after grid tokens
            # This matches autoregressive generation: grid → CNN → reasoning → action
            x = jnp.concatenate([grid_tokens, reasoning_embed], axis=1)  # (batch, H*W + reasoning_seq_len, d_model)
        else:
            x = grid_tokens
        
        # Create attention mask (causal for reasoning tokens)
        total_seq_len = x.shape[1]
        mask = None
        if self.use_reasoning and reasoning_tokens is not None:
            mask = self.make_causal_mask(total_seq_len, reasoning_start_idx)
        
        # Transformer encoder layers
        for i in range(self.num_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                name=f'transformer_block_{i}'
            )(x, training=training, mask=mask)
        
        # Layer norm before heads
        x = nn.LayerNorm()(x)
        
        # For RSM mode during generation, we need per-token logits
        # Otherwise, pool for single action prediction
        if self.use_reasoning and reasoning_tokens is not None:
            # Use the last token's representation for action/next-token prediction
            last_token = x[:, -1, :]  # (batch, d_model)
        else:
            # Pool across sequence dimension for non-RSM mode
            last_token = x.mean(axis=1)  # (batch, d_model)
        
        # LM head (used for both reasoning token generation and action prediction)
        # In RSM mode, this predicts next reasoning token OR final action
        lm_logits = nn.Dense(self.d_model, name='lm_hidden')(last_token)
        lm_logits = nn.gelu(lm_logits)
        
        # Output logits - vocab size for reasoning, num_actions for final decision
        if self.use_reasoning:
            # Predict over extended vocabulary: [action_0, action_1, action_2, action_3, token_0, ..., token_127]
            # This allows the model to predict either actions or reasoning tokens
            logits = nn.Dense(self.num_actions + self.reasoning_vocab_size, name='lm_logits')(lm_logits)
        else:
            # Standard action prediction
            logits = nn.Dense(self.num_actions, name='policy_logits')(lm_logits)
        
        # Critic head (value) - always use pooled representation
        if self.use_reasoning and reasoning_tokens is not None:
            # Pool the entire sequence for value estimation
            pooled = x.mean(axis=1)
        else:
            pooled = last_token
            
        value = nn.Dense(self.d_model, name='value_hidden')(pooled)
        value = nn.gelu(value)
        value = nn.Dense(1, name='value_out')(value)
        value = value.squeeze(-1)  # (batch,)
        
        return logits, value
    
    def init_params(self, rng, obs_shape, reasoning_seq_len=None):
        """Initialize network parameters"""
        dummy_obs = jnp.zeros((1, *obs_shape), dtype=jnp.float32)
        if self.use_reasoning and reasoning_seq_len is not None:
            dummy_reasoning = jnp.zeros((1, reasoning_seq_len), dtype=jnp.int32)
            return self.init(rng, dummy_obs, training=False, reasoning_tokens=dummy_reasoning)
        return self.init(rng, dummy_obs, training=False)
