"""
Transformer model for Snake pretraining with teacher-forcing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from positional_encoding import PositionalEncoding2D


class TransformerPolicyPretrainer(nn.Module):
    """
    Transformer-based policy network for supervised pretraining.
    Uses same architecture as TransformerExtractor but with action classification head.
    """
    
    def __init__(
        self,
        height: int = 20,
        width: int = 20,
        in_channels: int = 3,
        d_model: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        action_dim: int = 4
    ):
        super().__init__()
        
        self.height = height
        self.width = width
        self.d_model = d_model
        
        # Input projection: (B, S, 3) -> (B, S, d_model)
        self.input_proj = nn.Linear(in_channels, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(d_model, height, width)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm architecture (more stable)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        
        # Action classification head
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, action_dim)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            obs: Observations of shape (B, H, W, 3)
        
        Returns:
            Action logits of shape (B, 4)
        """
        b, h, w, c = obs.shape
        
        # Reshape to tokens: (B, H, W, 3) -> (B, H*W, 3)
        tokens = obs.view(b, h * w, c)
        
        # Project to d_model: (B, S, 3) -> (B, S, d)
        x = self.input_proj(tokens)
        
        # Add positional encoding: (B, S, d)
        x = self.pos_encoding(x)
        
        # Transformer: (B, S, d) -> (B, S, d)
        x = self.transformer(x)
        
        # Global average pooling: (B, S, d) -> (B, d)
        x = x.mean(dim=1)
        
        # Normalize
        x = self.norm(x)
        
        # Action logits: (B, d) -> (B, 4)
        logits = self.action_head(x)
        
        return logits
    
    def get_features(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Extract features without action head (for transfer learning).
        
        Args:
            obs: Observations of shape (B, H, W, 3)
        
        Returns:
            Features of shape (B, d_model)
        """
        b, h, w, c = obs.shape
        tokens = obs.view(b, h * w, c)
        x = self.input_proj(tokens)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.norm(x)
        return x
    
    def predict_action(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Predict action from observation.
        
        Args:
            obs: Observation of shape (B, H, W, 3) or (H, W, 3)
            deterministic: If True, return argmax; else sample from distribution
        
        Returns:
            Action indices of shape (B,) or scalar
        """
        single = obs.dim() == 3
        if single:
            obs = obs.unsqueeze(0)
        
        with torch.no_grad():
            logits = self.forward(obs)
            
            if deterministic:
                actions = logits.argmax(dim=1)
            else:
                probs = F.softmax(logits, dim=1)
                actions = torch.multinomial(probs, num_samples=1).squeeze(1)
        
        if single:
            return actions.item()
        return actions


class TransformerValuePretrainer(nn.Module):
    """
    Transformer-based value network for auxiliary pretraining task.
    Predicts expected return or distance to food.
    """
    
    def __init__(
        self,
        height: int = 20,
        width: int = 20,
        in_channels: int = 3,
        d_model: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.height = height
        self.width = width
        self.d_model = d_model
        
        # Shared encoder (same as policy)
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_encoding = PositionalEncoding2D(d_model, height, width)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(d_model)
        
        # Value head (predicts scalar value)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            obs: Observations of shape (B, H, W, 3)
        
        Returns:
            Value predictions of shape (B, 1)
        """
        b, h, w, c = obs.shape
        tokens = obs.view(b, h * w, c)
        x = self.input_proj(tokens)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.norm(x)
        value = self.value_head(x)
        return value


class MultiTaskPretrainer(nn.Module):
    """
    Multi-task pretraining combining policy and value prediction.
    Shares encoder, separate heads.
    """
    
    def __init__(
        self,
        height: int = 20,
        width: int = 20,
        in_channels: int = 3,
        d_model: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        action_dim: int = 4
    ):
        super().__init__()
        
        self.height = height
        self.width = width
        self.d_model = d_model
        
        # Shared encoder
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_encoding = PositionalEncoding2D(d_model, height, width)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(d_model)
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, action_dim)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
        # Auxiliary heads (optional)
        self.snake_length_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        
        self.food_distance_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, obs: torch.Tensor):
        """
        Forward pass with all heads.
        
        Args:
            obs: Observations of shape (B, H, W, 3)
        
        Returns:
            Dict with 'policy_logits', 'value', 'snake_length', 'food_distance'
        """
        b, h, w, c = obs.shape
        
        # Shared encoding
        tokens = obs.view(b, h * w, c)
        x = self.input_proj(tokens)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        features = x.mean(dim=1)
        features = self.norm(features)
        
        # All predictions
        return {
            'policy_logits': self.policy_head(features),
            'value': self.value_head(features),
            'snake_length': self.snake_length_head(features),
            'food_distance': self.food_distance_head(features)
        }
    
    def get_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract features for transfer learning."""
        b, h, w, c = obs.shape
        tokens = obs.view(b, h * w, c)
        x = self.input_proj(tokens)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.norm(x)
        return x


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    print("Testing TransformerPolicyPretrainer...")
    
    model = TransformerPolicyPretrainer(
        height=20, width=20,
        d_model=64, num_layers=2, num_heads=4
    )
    
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 8
    obs = torch.randn(batch_size, 20, 20, 3)
    logits = model(obs)
    print(f"Input shape: {obs.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Test prediction
    action = model.predict_action(obs[0])
    print(f"Predicted action: {action}")
    
    print("\nTesting MultiTaskPretrainer...")
    multi_model = MultiTaskPretrainer(
        height=20, width=20,
        d_model=64, num_layers=2, num_heads=4
    )
    print(f"Parameters: {count_parameters(multi_model):,}")
    
    outputs = multi_model(obs)
    print(f"Policy logits shape: {outputs['policy_logits'].shape}")
    print(f"Value shape: {outputs['value'].shape}")
    print(f"Snake length shape: {outputs['snake_length'].shape}")
    print(f"Food distance shape: {outputs['food_distance'].shape}")
