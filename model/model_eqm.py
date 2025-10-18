"""
Equilibrium Matching (EqM) World Model for Snake RL

Based on "Equilibrium Matching: Generative Modeling with Implicit Energy-Based Models"
by Runqian Wang and Yilun Du.

This model learns a time-invariant gradient field that optimizes jointly over:
- Next state (in latent space)
- Action to reach that state

The gradient field is conditioned on the current state (encoded by transformer),
and generates (next_state, action) pairs via gradient descent optimization.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict
from transformers import PretrainedConfig, PreTrainedModel

from .model_pytorch import PositionalEncoding2D, CNNEncoder


class SnakeWorldEqMConfig(PretrainedConfig):
    """Configuration for Snake World EqM model"""

    model_type = "snake_world_eqm"

    def __init__(
        self,
        # Encoder config
        d_model: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        use_cnn: bool = False,
        cnn_mode: str = "replace",
        
        # Action space
        num_actions: int = 4,
        
        # EqM config
        latent_dim: int = 64,  # dimension of latent next-state representation
        eqm_hidden_dim: int = 128,  # hidden dimension for EqM gradient network
        eqm_num_layers: int = 3,  # depth of EqM gradient network
        
        # EqM gradient magnitude function
        gradient_schedule: str = "linear",  # "linear", "truncated", or "piecewise"
        gradient_truncate_a: float = 0.3,  # for truncated/piecewise schedules
        gradient_piecewise_b: float = 2.0,  # for piecewise schedule
        gradient_multiplier: float = 1.0,  # lambda: overall gradient scale
        
        # EqM sampling config
        eqm_sampling_steps: int = 10,
        eqm_step_size: float = 0.1,
        eqm_nag_momentum: float = 0.9,  # Nesterov momentum
        eqm_adaptive_threshold: Optional[float] = None,  # if set, use adaptive compute
        
        # Optional: explicit energy learning
        use_explicit_energy: bool = False,
        energy_type: str = "dot_product",  # "dot_product" or "squared_l2"
        
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # Encoder
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_cnn = use_cnn
        self.cnn_mode = cnn_mode
        
        # Action space
        self.num_actions = num_actions
        
        # EqM
        self.latent_dim = latent_dim
        self.eqm_hidden_dim = eqm_hidden_dim
        self.eqm_num_layers = eqm_num_layers
        
        # Gradient schedule
        self.gradient_schedule = gradient_schedule
        self.gradient_truncate_a = gradient_truncate_a
        self.gradient_piecewise_b = gradient_piecewise_b
        self.gradient_multiplier = gradient_multiplier
        
        # Sampling
        self.eqm_sampling_steps = eqm_sampling_steps
        self.eqm_step_size = eqm_step_size
        self.eqm_nag_momentum = eqm_nag_momentum
        self.eqm_adaptive_threshold = eqm_adaptive_threshold
        
        # Explicit energy
        self.use_explicit_energy = use_explicit_energy
        self.energy_type = energy_type


class EqMGradientField(nn.Module):
    """
    EqM Gradient Field Network
    
    Predicts the gradient of the energy landscape with respect to (next_state, action).
    Conditioned on the current state's hidden representation.
    """
    
    def __init__(
        self,
        condition_dim: int,  # dimension of conditioning (hidden state from encoder)
        latent_dim: int,  # dimension of latent next state
        action_dim: int,  # dimension of action (continuous logits)
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # Input: [condition, latent_next_state, action_logits]
        input_dim = condition_dim + latent_dim + action_dim
        
        # Build MLP for gradient prediction
        layers = []
        prev_dim = input_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        
        # Separate heads for state and action gradients
        self.state_grad_head = nn.Linear(hidden_dim, latent_dim)
        self.action_grad_head = nn.Linear(hidden_dim, action_dim)
    
    def forward(
        self,
        condition: torch.Tensor,  # (batch, condition_dim)
        latent_state: torch.Tensor,  # (batch, latent_dim)
        action_logits: torch.Tensor,  # (batch, action_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict gradients for both latent state and action.
        
        Returns:
            grad_state: (batch, latent_dim) - gradient w.r.t. latent next state
            grad_action: (batch, action_dim) - gradient w.r.t. action logits
        """
        # Concatenate inputs
        x = torch.cat([condition, latent_state, action_logits], dim=-1)
        
        # Shared processing
        features = self.network(x)
        
        # Predict gradients
        grad_state = self.state_grad_head(features)
        grad_action = self.action_grad_head(features)
        
        return grad_state, grad_action


class SnakeWorldEqM(PreTrainedModel):
    """
    Snake World Model using Equilibrium Matching
    
    Architecture:
    1. Encoder: Processes current grid state into hidden representation
    2. EqM Decoder: Gradient field that optimizes (next_state, action) jointly
    3. Optional: Explicit energy function
    
    Training:
    - Uses EqM objective: match gradients at interpolated states
    - Learns energy landscape where (valid_next_state, correct_action) are minima
    
    Inference:
    - Initialize next_state and action (from noise or current state)
    - Run gradient descent to find optimal (next_state, action)
    - Uses Nesterov momentum and optional adaptive compute
    """
    
    config_class = SnakeWorldEqMConfig
    
    def __init__(self, config: SnakeWorldEqMConfig):
        super().__init__(config)
        
        self.config = config
        self.d_model = config.d_model
        self.num_actions = config.num_actions
        self.latent_dim = config.latent_dim
        
        # ========== Encoder ==========
        # Grid encoding
        self.pos_encoding = PositionalEncoding2D(config.d_model)
        
        # Input projection (3 RGB channels)
        self.input_proj = nn.Linear(3, config.d_model)
        
        # Optional CNN
        if config.use_cnn:
            self.cnn = CNNEncoder(in_channels=3, features=(32, 64))
            self.cnn_proj = nn.Linear(64, config.d_model)
        
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
        
        self.encoder_norm = nn.LayerNorm(config.d_model)
        
        # ========== Latent State Projection ==========
        # Project current grid to latent space (for initialization and targets)
        self.to_latent = nn.Linear(config.d_model, config.latent_dim)
        
        # Optional: reconstruct grid from latent (for auxiliary loss or visualization)
        self.from_latent = nn.Linear(config.latent_dim, config.d_model)
        
        # ========== EqM Gradient Field ==========
        self.eqm_field = EqMGradientField(
            condition_dim=config.d_model,
            latent_dim=config.latent_dim,
            action_dim=config.num_actions,
            hidden_dim=config.eqm_hidden_dim,
            num_layers=config.eqm_num_layers,
        )
        
        # ========== Optional: Explicit Energy ==========
        if config.use_explicit_energy:
            # The energy function uses the same gradient field
            # but we compute explicit energy values
            self.energy_type = config.energy_type
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def encode_state(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the current grid state into hidden representation.
        
        Args:
            obs: (batch, 3, height, width) game state
        
        Returns:
            hidden_pooled: (batch, d_model) - pooled representation for conditioning
            hidden_tokens: (batch, H*W, d_model) - all token representations
        """
        batch_size, channels, height, width = obs.shape
        device = obs.device
        
        # Encode grid
        if self.config.use_cnn:
            # CNN encoding
            cnn_features = self.cnn(obs)
            cnn_features = cnn_features.permute(0, 2, 3, 1)
            cnn_tokens = cnn_features.reshape(batch_size, height * width, -1)
            cnn_tokens = self.cnn_proj(cnn_tokens)
            
            if self.config.cnn_mode == "replace":
                grid_tokens = cnn_tokens
            else:  # append
                obs_flat = obs.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
                linear_tokens = self.input_proj(obs_flat)
                grid_tokens = torch.cat([linear_tokens, cnn_tokens], dim=1)
        else:
            # Linear projection
            obs_flat = obs.permute(0, 2, 3, 1)
            obs_flat = obs_flat.reshape(batch_size, height * width, channels)
            grid_tokens = self.input_proj(obs_flat)
        
        # Add positional encoding
        pos_enc = self.pos_encoding(height, width, device=device)
        
        if self.config.cnn_mode == "append" and self.config.use_cnn:
            pos_enc_full = torch.cat([pos_enc, pos_enc], dim=0)
            grid_tokens = grid_tokens + pos_enc_full.unsqueeze(0)
        else:
            grid_tokens = grid_tokens + pos_enc.unsqueeze(0)
        
        # Transformer encoding
        hidden_tokens = self.transformer(grid_tokens)
        
        # Pool for conditioning (mean pooling)
        hidden_pooled = hidden_tokens.mean(dim=1)
        hidden_pooled = self.encoder_norm(hidden_pooled)
        
        return hidden_pooled, hidden_tokens
    
    def gradient_magnitude_schedule(self, gamma: torch.Tensor) -> torch.Tensor:
        """
        Compute c(gamma) - the gradient magnitude schedule.
        
        Args:
            gamma: (batch,) interpolation factor [0, 1]
        
        Returns:
            c: (batch,) gradient magnitude
        """
        if self.config.gradient_schedule == "linear":
            # c(gamma) = lambda * (1 - gamma)
            c = 1.0 - gamma
        
        elif self.config.gradient_schedule == "truncated":
            # c(gamma) = 1 if gamma <= a, else (1 - gamma) / (1 - a)
            a = self.config.gradient_truncate_a
            c = torch.where(
                gamma <= a,
                torch.ones_like(gamma),
                (1.0 - gamma) / (1.0 - a)
            )
        
        elif self.config.gradient_schedule == "piecewise":
            # c(gamma) = b - (b-1)/a * gamma if gamma <= a, else (1 - gamma) / (1 - a)
            a = self.config.gradient_truncate_a
            b = self.config.gradient_piecewise_b
            c = torch.where(
                gamma <= a,
                b - (b - 1.0) / a * gamma,
                (1.0 - gamma) / (1.0 - a)
            )
        
        else:
            raise ValueError(f"Unknown gradient schedule: {self.config.gradient_schedule}")
        
        # Apply gradient multiplier
        c = c * self.config.gradient_multiplier
        
        return c
    
    def compute_eqm_loss(
        self,
        obs_t: torch.Tensor,
        obs_t_plus_1: torch.Tensor,
        action_t: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute EqM training loss.
        
        Args:
            obs_t: (batch, 3, H, W) current state
            obs_t_plus_1: (batch, 3, H, W) next state
            action_t: (batch,) actions taken (as class indices)
        
        Returns:
            Dictionary with loss components
        """
        batch_size = obs_t.shape[0]
        device = obs_t.device
        
        # Encode current state (conditioning)
        hidden_t, _ = self.encode_state(obs_t)  # (batch, d_model)
        
        # Encode target next state to latent
        hidden_t_plus_1, _ = self.encode_state(obs_t_plus_1)
        latent_target = self.to_latent(hidden_t_plus_1)  # (batch, latent_dim)
        
        # Convert action to one-hot
        action_target = torch.nn.functional.one_hot(action_t, self.num_actions).float()
        
        # Sample noise
        noise_latent = torch.randn_like(latent_target)
        noise_action = torch.randn(batch_size, self.num_actions, device=device)
        
        # Sample gamma (interpolation factor)
        gamma = torch.rand(batch_size, device=device)
        
        # Interpolate to get corrupted samples
        latent_gamma = gamma.view(-1, 1) * latent_target + (1 - gamma.view(-1, 1)) * noise_latent
        action_gamma = gamma.view(-1, 1) * action_target + (1 - gamma.view(-1, 1)) * noise_action
        
        # Predict gradients
        pred_grad_latent, pred_grad_action = self.eqm_field(
            condition=hidden_t,
            latent_state=latent_gamma,
            action_logits=action_gamma,
        )
        
        # Compute target gradients
        # Direction: (noise - target) descends from noise to target
        # Magnitude: c(gamma)
        c = self.gradient_magnitude_schedule(gamma).view(-1, 1)
        
        target_grad_latent = (noise_latent - latent_target) * c
        target_grad_action = (noise_action - action_target) * c
        
        # MSE loss
        loss_latent = torch.mean((pred_grad_latent - target_grad_latent) ** 2)
        loss_action = torch.mean((pred_grad_action - target_grad_action) ** 2)
        
        total_loss = loss_latent + loss_action
        
        return {
            "loss": total_loss,
            "loss_latent": loss_latent,
            "loss_action": loss_action,
        }
    
    @torch.no_grad()
    def sample(
        self,
        obs_t: torch.Tensor,
        return_trajectory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample (next_state, action) via gradient descent on the energy landscape.
        
        Args:
            obs_t: (batch, 3, H, W) current state
            return_trajectory: if True, return full optimization trajectory
        
        Returns:
            Dictionary containing:
                - latent_next: (batch, latent_dim) optimized next state in latent space
                - action_logits: (batch, num_actions) optimized action logits
                - action: (batch,) discrete action (argmax of logits)
                - num_steps: (batch,) number of optimization steps taken (if adaptive)
                - trajectory: list of (latent, action, grad_norm) if return_trajectory=True
        """
        batch_size = obs_t.shape[0]
        device = obs_t.device
        
        # Encode current state
        hidden_t, _ = self.encode_state(obs_t)
        
        # Initialize optimization variables
        # Option 1: Initialize from current state (warm start)
        latent_next = self.to_latent(hidden_t)
        
        # Option 2: Initialize from noise (cold start) - uncomment to use
        # latent_next = torch.randn(batch_size, self.latent_dim, device=device)
        
        action_logits = torch.randn(batch_size, self.num_actions, device=device)
        
        # For NAG momentum
        latent_prev = latent_next.clone()
        action_prev = action_logits.clone()
        
        # Tracking
        num_steps = torch.zeros(batch_size, device=device)
        trajectory = [] if return_trajectory else None
        
        # Gradient descent optimization
        max_steps = self.config.eqm_sampling_steps
        eta = self.config.eqm_step_size
        mu = self.config.eqm_nag_momentum
        threshold = self.config.eqm_adaptive_threshold
        
        for step in range(max_steps):
            # Nesterov look-ahead
            latent_look = latent_next + mu * (latent_next - latent_prev)
            action_look = action_logits + mu * (action_logits - action_prev)
            
            # Compute gradients at look-ahead point
            grad_latent, grad_action = self.eqm_field(
                condition=hidden_t,
                latent_state=latent_look,
                action_logits=action_look,
            )
            
            # Update previous
            latent_prev = latent_next.clone()
            action_prev = action_logits.clone()
            
            # Gradient descent step
            latent_next = latent_next - eta * grad_latent
            action_logits = action_logits - eta * grad_action
            
            # Track
            if return_trajectory:
                grad_norm = torch.sqrt(
                    (grad_latent ** 2).sum(dim=-1) + (grad_action ** 2).sum(dim=-1)
                )
                trajectory.append({
                    "latent": latent_next.clone(),
                    "action": action_logits.clone(),
                    "grad_norm": grad_norm,
                })
            
            # Adaptive stopping
            if threshold is not None:
                grad_norm = torch.sqrt(
                    (grad_latent ** 2).sum(dim=-1) + (grad_action ** 2).sum(dim=-1)
                )
                still_optimizing = grad_norm > threshold
                num_steps += still_optimizing.float()
                
                if not still_optimizing.any():
                    break
        
        # Extract discrete action
        action = action_logits.argmax(dim=-1)
        
        result = {
            "latent_next": latent_next,
            "action_logits": action_logits,
            "action": action,
        }
        
        if threshold is not None:
            result["num_steps"] = num_steps
        
        if return_trajectory:
            result["trajectory"] = trajectory
        
        return result
    
    def forward(
        self,
        obs: torch.Tensor,
        next_obs: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        mode: str = "sample",
    ):
        """
        Forward pass - supports both training and inference modes.
        
        Args:
            obs: (batch, 3, H, W) current state
            next_obs: (batch, 3, H, W) next state (required for mode="train")
            actions: (batch,) actions taken (required for mode="train")
            mode: "train" or "sample"
        
        Returns:
            If mode="train": loss dictionary
            If mode="sample": sample dictionary with (next_state, action)
        """
        if mode == "train":
            if next_obs is None or actions is None:
                raise ValueError("next_obs and actions required for training mode")
            return self.compute_eqm_loss(obs, next_obs, actions)
        
        elif mode == "sample":
            return self.sample(obs)
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'train' or 'sample'")
