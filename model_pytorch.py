"""
PyTorch Transformer Policy for Snake RL
HuggingFace compatible implementation
"""

import torch
import torch.nn as nn
import math
from typing import Optional
from transformers import PretrainedConfig, PreTrainedModel


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for grid-based observations"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
    def forward(self, height: int, width: int, device=None) -> torch.Tensor:
        """
        Generate 2D positional encoding
        
        Returns:
            pos_embed: (height * width, d_model)
        """
        if device is None:
            device = torch.device('cpu')
            
        # Create positional encodings
        pos_h = torch.arange(height, device=device).unsqueeze(1).float()  # (H, 1)
        pos_w = torch.arange(width, device=device).unsqueeze(0).float()   # (1, W)
        
        # Frequency bands
        d_4 = self.d_model // 4
        freqs = torch.exp(torch.arange(0, d_4, device=device).float() * -(math.log(10000.0) / d_4))
        
        # Height encodings
        pos_h_expanded = pos_h.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)  # (H, 1, d_4)
        sin_h = torch.sin(pos_h_expanded)
        cos_h = torch.cos(pos_h_expanded)
        
        # Width encodings
        pos_w_expanded = pos_w.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)  # (1, W, d_4)
        sin_w = torch.sin(pos_w_expanded)
        cos_w = torch.cos(pos_w_expanded)
        
        # Concatenate (H, W, d_model)
        pos_encoding = torch.cat([
            sin_h * torch.ones_like(sin_w),
            cos_h * torch.ones_like(cos_w),
            torch.ones_like(sin_h) * sin_w,
            torch.ones_like(cos_h) * cos_w,
        ], dim=-1)
        
        # Reshape to (H*W, d_model)
        pos_encoding = pos_encoding.reshape(height * width, self.d_model)
        
        return pos_encoding


class CNNEncoder(nn.Module):
    """Convolutional encoder for spatial features"""

    def __init__(self, in_channels: int = 3, features=(32, 64), kernel_sizes=(3, 3)):
        super().__init__()
        layers = []
        prev_ch = in_channels

        for feat, ks in zip(features, kernel_sizes):
            layers.extend([
                nn.Conv2d(prev_ch, feat, kernel_size=ks, padding=ks//2),
                nn.ReLU(),
            ])
            prev_ch = feat

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            features: (batch, features, height, width)
        """
        return self.conv(x)


class SnakeTransformerConfig(PretrainedConfig):
    """Configuration for Snake Transformer"""

    model_type = "snake_transformer"

    def __init__(
        self,
        d_model: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        num_actions: int = 4,
        dropout_rate: float = 0.1,
        use_cnn: bool = False,
        cnn_mode: str = "replace",
        use_reasoning: bool = False,
        reasoning_vocab_size: int = 512,
        max_reasoning_length: int = 128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_actions = num_actions
        self.dropout_rate = dropout_rate
        self.use_cnn = use_cnn
        self.cnn_mode = cnn_mode
        self.use_reasoning = use_reasoning
        self.reasoning_vocab_size = reasoning_vocab_size
        self.max_reasoning_length = max_reasoning_length


class TransformerPolicy(PreTrainedModel):
    """Transformer-based policy for Snake game"""

    config_class = SnakeTransformerConfig

    def __init__(self, config: SnakeTransformerConfig):
        super().__init__(config)

        self.d_model = config.d_model
        self.num_actions = config.num_actions
        self.use_cnn = config.use_cnn
        self.cnn_mode = config.cnn_mode
        self.use_reasoning = config.use_reasoning

        # Grid encoding
        self.pos_encoding = PositionalEncoding2D(config.d_model)

        # Input projection (3 channels: empty, snake, food)
        if config.use_cnn:
            self.cnn = CNNEncoder(in_channels=3, features=(32, 64))
            cnn_out_channels = 64
            self.cnn_proj = nn.Linear(cnn_out_channels, config.d_model)

            if config.cnn_mode == "append":
                # Will concatenate CNN features with grid tokens
                pass
            # else mode == "replace": CNN replaces grid embeddings
        else:
            self.input_proj = nn.Linear(3, config.d_model)

        # Reasoning token embeddings (if RSM mode)
        if config.use_reasoning:
            self.reasoning_embed = nn.Embedding(config.reasoning_vocab_size, config.d_model)
            self.reasoning_pos = nn.Embedding(config.max_reasoning_length, config.d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout_rate,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Output heads
        self.norm = nn.LayerNorm(config.d_model)

        if config.use_reasoning:
            # RSM: predict both actions and next reasoning tokens
            self.action_head = nn.Linear(config.d_model, config.num_actions)
            self.reasoning_head = nn.Linear(config.d_model, config.reasoning_vocab_size)
        else:
            self.action_head = nn.Linear(config.d_model, config.num_actions)

    def forward(
        self,
        obs: torch.Tensor,
        reasoning_tokens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            obs: (batch, channels, height, width) game state
            reasoning_tokens: (batch, seq_len) reasoning token IDs (RSM mode)
            attention_mask: (batch, seq_len) attention mask
            
        Returns:
            logits: (batch, num_actions) or (batch, num_actions + vocab_size) for RSM
            hidden: (batch, seq_len, d_model) transformer output
        """
        batch_size, channels, height, width = obs.shape
        device = obs.device

        # Encode grid
        if self.use_cnn:
            # CNN encoding
            cnn_features = self.cnn(obs)  # (batch, cnn_channels, H, W)
            cnn_features = cnn_features.permute(0, 2, 3, 1)  # (batch, H, W, cnn_channels)
            cnn_tokens = cnn_features.reshape(batch_size, height * width, -1)  # (batch, H*W, cnn_channels)
            cnn_tokens = self.cnn_proj(cnn_tokens)  # (batch, H*W, d_model)

            if self.cnn_mode == "replace":
                # Use CNN features directly
                grid_tokens = cnn_tokens
            else:  # append
                # Also compute linear embeddings to append
                obs_flat = obs.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
                linear_tokens = self.input_proj(obs_flat)
                grid_tokens = torch.cat([linear_tokens, cnn_tokens], dim=1)  # (batch, 2*H*W, d_model)
        else:
            # Linear projection
            obs_flat = obs.permute(0, 2, 3, 1)  # (batch, H, W, channels)
            obs_flat = obs_flat.reshape(batch_size, height * width, channels)
            grid_tokens = self.input_proj(obs_flat)  # (batch, H*W, d_model)

        # Add positional encoding to grid tokens
        pos_enc = self.pos_encoding(height, width, device=device)  # (H*W, d_model)

        if self.cnn_mode == "append" and self.use_cnn:
            # Split positional encoding for linear and CNN tokens
            pos_enc_full = torch.cat([pos_enc, pos_enc], dim=0)  # (2*H*W, d_model)
            grid_tokens = grid_tokens + pos_enc_full.unsqueeze(0)
        else:
            grid_tokens = grid_tokens + pos_enc.unsqueeze(0)  # (batch, H*W, d_model)

        # Add reasoning tokens if provided (RSM mode)
        if self.use_reasoning and reasoning_tokens is not None:
            seq_len = reasoning_tokens.shape[1]

            # Embed reasoning tokens
            reasoning_embeds = self.reasoning_embed(reasoning_tokens)  # (batch, seq_len, d_model)

            # Add positional encoding for reasoning
            reasoning_pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            reasoning_pos_enc = self.reasoning_pos(reasoning_pos_ids)
            reasoning_embeds = reasoning_embeds + reasoning_pos_enc

            # Concatenate grid and reasoning tokens
            tokens = torch.cat([reasoning_embeds, grid_tokens], dim=1)  # (batch, seq_len + H*W, d_model)
        else:
            tokens = grid_tokens

        # Transformer encoding
        hidden = self.transformer(tokens, src_key_padding_mask=attention_mask)

        # Pool for action prediction (mean over all tokens)
        pooled = hidden.mean(dim=1)  # (batch, d_model)
        pooled = self.norm(pooled)

        # Action logits
        action_logits = self.action_head(pooled)  # (batch, num_actions)

        if self.use_reasoning:
            # Also predict reasoning tokens (for autoregressive training)
            reasoning_logits = self.reasoning_head(pooled)  # (batch, vocab_size)
            logits = torch.cat([action_logits, reasoning_logits], dim=-1)
        else:
            logits = action_logits

        return logits, hidden
