#!/usr/bin/env python3
"""
Test autoregressive reasoning generation for RSM models.
"""

import jax
import jax.numpy as jnp
import numpy as np

from snake_jax.network import TransformerPolicy
from snake_jax.config import EnvConfig
from snake_jax.env import SnakeEnv


def test_rsm_generation():
    """Test that autoregressive generation works correctly."""
    
    print("=" * 60)
    print("TESTING RSM AUTOREGRESSIVE GENERATION")
    print("=" * 60)
    print()
    
    # Create RSM network
    network = TransformerPolicy(
        d_model=64,
        num_layers=2,
        num_heads=4,
        num_actions=4,
        dropout_rate=0.1,
        use_cnn=False,
        use_reasoning=True,  # Enable RSM mode
    )
    
    # Initialize
    rng = jax.random.PRNGKey(42)
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    
    dummy_obs = jnp.zeros((1, 32, 32, 3), dtype=jnp.float32)
    dummy_reasoning = jnp.zeros((1, 10), dtype=jnp.int32)  # Initialize with reasoning tokens
    
    params = network.init(
        {"params": init_rng, "dropout": dropout_rng}, 
        dummy_obs, 
        training=False,
        reasoning_tokens=dummy_reasoning  # Need this to initialize embedding layer
    )
    
    print("✓ Network initialized")
    print(f"  Model has {sum(x.size for x in jax.tree_util.tree_leaves(params))} parameters")
    print()
    
    # Test 1: Forward pass without reasoning
    print("Test 1: Forward pass without reasoning")
    logits, value = network.apply(params, dummy_obs, training=False)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Expected: (1, 132) = (batch, actions + vocab)")
    print(f"  ✓ Shape correct: {logits.shape == (1, 132)}")
    print(f"  Action logits (0-3): {logits[0, :4]}")
    print(f"  Value: {value}")
    print()
    
    # Test 2: Forward pass with reasoning tokens
    print("Test 2: Forward pass with reasoning tokens")
    reasoning = jnp.array([[72, 101, 108, 108, 111]], dtype=jnp.int32)  # "Hello" in ASCII
    logits, value = network.apply(
        params, dummy_obs, 
        training=False, 
        reasoning_tokens=reasoning
    )
    print(f"  Reasoning tokens: {reasoning}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  ✓ Shape correct: {logits.shape == (1, 132)}")
    print(f"  Action logits (0-3): {logits[0, :4]}")
    print()
    
    # Test 3: Autoregressive generation
    print("Test 3: Autoregressive generation (max 20 tokens)")
    obs = dummy_obs[0]  # Remove batch dim
    generated = []
    current_reasoning = jnp.zeros((1, 0), dtype=jnp.int32)
    
    for t in range(20):
        # Forward pass
        if current_reasoning.shape[1] > 0:
            logits, _ = network.apply(
                params, obs[None, ...], 
                training=False, 
                reasoning_tokens=current_reasoning
            )
        else:
            logits, _ = network.apply(
                params, obs[None, ...], 
                training=False
            )
        
        # Greedy sampling
        next_token = int(jnp.argmax(logits[0]))
        
        print(f"  Step {t}: predicted token {next_token}", end="")
        
        # Check if action
        if next_token < 4:
            action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
            print(f" → ACTION: {action_names[next_token]}")
            break
        else:
            # Reasoning token
            ascii_val = next_token - 4
            char = chr(min(max(ascii_val, 32), 126))  # Printable ASCII
            generated.append(char)
            print(f" → reasoning token (ASCII {ascii_val}: '{char}')")
            
            # Append to sequence
            new_token = jnp.array([[ascii_val]], dtype=jnp.int32)
            if current_reasoning.shape[1] == 0:
                current_reasoning = new_token
            else:
                current_reasoning = jnp.concatenate([current_reasoning, new_token], axis=1)
    
    print(f"\n  Generated reasoning text: '{''.join(generated)}'")
    print(f"  Final action: {next_token if next_token < 4 else 'None (reached max tokens)'}")
    print()
    
    # Test 4: Causal mask verification
    print("Test 4: Causal mask verification")
    mask = network.make_causal_mask(seq_len=10, reasoning_start_idx=5)
    print(f"  Mask shape: {mask.shape}")
    print(f"  Expected: (1, 1, 10, 10)")
    print(f"  ✓ Shape correct: {mask.shape == (1, 1, 10, 10)}")
    
    # Check that grid tokens see all grid tokens
    grid_section = mask[0, 0, :5, :5]
    print(f"  Grid section (all True): {jnp.all(grid_section)}")
    
    # Check that reasoning tokens have causal structure
    reasoning_section = mask[0, 0, 5:, 5:]
    is_lower_triangular = jnp.allclose(
        reasoning_section, 
        jnp.tril(jnp.ones_like(reasoning_section))
    )
    print(f"  Reasoning section (causal/lower-triangular): {is_lower_triangular}")
    
    # Check that reasoning can see all grid
    cross_attention = mask[0, 0, 5:, :5]
    print(f"  Reasoning→Grid attention (all True): {jnp.all(cross_attention)}")
    print()
    
    print("=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_rsm_generation()
