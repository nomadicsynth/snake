"""
Integration utilities for using pretrained models with SB3 training.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
from pretrain_model import TransformerPolicyPretrainer


def load_pretrained_weights(
    checkpoint_path: str,
    device: str = 'cuda'
) -> dict:
    """
    Load pretrained model checkpoint.
    
    Returns:
        Dictionary with model state dict and config
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def transfer_pretrained_weights(
    sb3_extractor: nn.Module,
    pretrained_checkpoint: dict,
    strict: bool = False,
    freeze_encoder: bool = False
) -> nn.Module:
    """
    Transfer weights from pretrained model to SB3 TransformerExtractor.
    
    Args:
        sb3_extractor: The TransformerExtractor from SB3 policy
        pretrained_checkpoint: Checkpoint dict from load_pretrained_weights
        strict: If True, require exact match of all parameters
        freeze_encoder: If True, freeze transferred parameters
    
    Returns:
        Updated extractor with pretrained weights
    """
    pretrained_state = pretrained_checkpoint['model_state_dict']
    
    # Map pretrained keys to extractor keys
    # Pretrained model has: input_proj, pos_encoding, transformer, norm
    # SB3 extractor should have the same
    
    # Filter out action_head weights (we don't need those)
    encoder_state = {
        k: v for k, v in pretrained_state.items()
        if not k.startswith('action_head')
    }
    
    # Load weights
    result = sb3_extractor.load_state_dict(encoder_state, strict=strict)
    
    if result.missing_keys:
        print(f"Warning: Missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"Warning: Unexpected keys: {result.unexpected_keys}")
    
    # Freeze encoder if requested
    if freeze_encoder:
        for name, param in sb3_extractor.named_parameters():
            param.requires_grad = False
        print("Froze all pretrained encoder parameters")
    
    return sb3_extractor


def evaluate_pretrained_model(
    model: TransformerPolicyPretrainer,
    env,
    num_episodes: int = 100,
    device: str = 'cuda',
    render: bool = False
) -> dict:
    """
    Evaluate a pretrained policy model in the environment.
    
    Args:
        model: Pretrained TransformerPolicyPretrainer
        env: Snake environment
        num_episodes: Number of episodes to run
        device: Device to run on
        render: Whether to render episodes
    
    Returns:
        Dict with evaluation metrics
    """
    model.eval()
    model.to(device)
    
    episode_rewards = []
    episode_lengths = []
    episode_scores = []  # Number of apples eaten
    
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        initial_length = len(env.snake)
        
        while not done:
            # Convert state to tensor
            state_tensor = torch.from_numpy(state).float().to(device)
            
            # Predict action (greedy)
            with torch.no_grad():
                action = model.predict_action(state_tensor, deterministic=True)
            
            # Step environment
            state, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if render:
                env._display()
                import time
                time.sleep(0.1)
        
        # Calculate score (apples eaten)
        final_length = len(env.snake)
        score = final_length - initial_length
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        episode_scores.append(score)
    
    results = {
        'mean_reward': sum(episode_rewards) / len(episode_rewards),
        'std_reward': (sum((r - sum(episode_rewards)/len(episode_rewards))**2 
                          for r in episode_rewards) / len(episode_rewards)) ** 0.5,
        'mean_length': sum(episode_lengths) / len(episode_lengths),
        'mean_score': sum(episode_scores) / len(episode_scores),
        'max_score': max(episode_scores),
        'min_score': min(episode_scores)
    }
    
    return results


def create_sb3_model_with_pretrained_weights(
    env,
    pretrained_checkpoint_path: str,
    algorithm: str = 'DQN',
    freeze_encoder: bool = False,
    **sb3_kwargs
):
    """
    Create SB3 model initialized with pretrained weights.
    
    Args:
        env: Gym environment
        pretrained_checkpoint_path: Path to pretrained checkpoint
        algorithm: 'DQN', 'PPO', 'A2C', etc.
        freeze_encoder: Whether to freeze pretrained encoder
        **sb3_kwargs: Additional arguments for SB3 algorithm
    
    Returns:
        SB3 model with pretrained weights
    """
    from stable_baselines3 import DQN, PPO, A2C
    from sb3_snake import TransformerExtractor
    
    # Load pretrained checkpoint
    checkpoint = load_pretrained_weights(pretrained_checkpoint_path)
    config = checkpoint['config']
    
    # Extract model config
    model_config = config['model']
    
    # Create policy kwargs with same architecture
    policy_kwargs = dict(
        features_extractor_class=TransformerExtractor,
        features_extractor_kwargs=dict(
            d_model=model_config['d_model'],
            n_layers=model_config['num_layers'],
            n_heads=model_config['num_heads'],
            dropout=model_config['dropout']
        ),
    )
    
    # Update with user kwargs
    if 'policy_kwargs' in sb3_kwargs:
        policy_kwargs.update(sb3_kwargs.pop('policy_kwargs'))
    
    # Create model
    algo_map = {
        'DQN': DQN,
        'PPO': PPO,
        'A2C': A2C
    }
    
    if algorithm not in algo_map:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    model = algo_map[algorithm](
        "MlpPolicy",  # We'll override with TransformerExtractor
        env,
        policy_kwargs=policy_kwargs,
        **sb3_kwargs
    )
    
    # Transfer pretrained weights
    print("Transferring pretrained weights...")
    transfer_pretrained_weights(
        model.policy.features_extractor,
        checkpoint,
        freeze_encoder=freeze_encoder
    )
    
    print(f"Created {algorithm} model with pretrained encoder")
    print(f"  Encoder frozen: {freeze_encoder}")
    print(f"  Pretrained val accuracy: {checkpoint.get('val_acc', 'N/A')}")
    
    return model


if __name__ == "__main__":
    # Example usage
    import sys
    from snake import SnakeGame
    
    if len(sys.argv) < 2:
        print("Usage: python pretrain_integration.py <checkpoint_path>")
        print("\nThis will evaluate the pretrained model in the Snake environment.")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint '{checkpoint_path}' not found!")
        sys.exit(1)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = load_pretrained_weights(checkpoint_path)
    
    # Print info
    print(f"\nCheckpoint info:")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    
    config = checkpoint['config']
    print(f"\nModel config:")
    for k, v in config['model'].items():
        print(f"  {k}: {v}")
    
    # Create model
    print(f"\nCreating model...")
    model = TransformerPolicyPretrainer(**config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create environment
    print("\nCreating environment...")
    env = SnakeGame(width=20, height=20, wall_collision=True)
    
    # Evaluate
    print("\nEvaluating pretrained policy (100 episodes)...")
    results = evaluate_pretrained_model(
        model, env, num_episodes=100,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\nResults:")
    print(f"  Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean episode length: {results['mean_length']:.1f}")
    print(f"  Mean score: {results['mean_score']:.2f}")
    print(f"  Max score: {results['max_score']}")
    print(f"  Min score: {results['min_score']}")
    
    print("\nTo use this model with SB3 training:")
    print(f"  from pretrain_integration import create_sb3_model_with_pretrained_weights")
    print(f"  model = create_sb3_model_with_pretrained_weights(")
    print(f"      env, '{checkpoint_path}', algorithm='DQN')")
    print(f"  model.learn(total_timesteps=100000)")
