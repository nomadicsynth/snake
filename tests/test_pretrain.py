"""
Test that all pretraining components work together.
Quick smoke test before running full pipeline.
"""

import sys
import numpy as np
import torch

print("Testing pretraining components...")
print("=" * 60)

# Test 1: Import all modules
print("\n1. Testing imports...")
try:
    from pretrain_utils import (
        astar_to_food, get_safe_actions, get_action_distribution,
        state_from_positions, augment_state_action, AUGMENTATIONS
    )
    from pretrain_dataset import (
        generate_random_state, generate_pretraining_dataset,
        SnakePretrainDataset
    )
    from pretrain_model import (
        TransformerPolicyPretrainer, count_parameters
    )
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Generate a few random states
print("\n2. Testing random state generation...")
try:
    states_generated = 0
    for i in range(10):
        state = generate_random_state(width=32, height=32, min_length=3, max_length=10)
        if state is not None:
            states_generated += 1
    
    if states_generated >= 5:
        print(f"   ✓ Generated {states_generated}/10 valid states")
    else:
        print(f"   ⚠ Only generated {states_generated}/10 states (might be unlucky)")
except Exception as e:
    print(f"   ✗ State generation failed: {e}")
    sys.exit(1)

# Test 3: A* pathfinding
print("\n3. Testing A* pathfinding...")
try:
    snake_pos = [(10, 10), (10, 9), (10, 8)]
    food_pos = (5, 15)
    path = astar_to_food(snake_pos, food_pos, width=32, height=32)
    
    if path is not None and len(path) > 0:
        print(f"   ✓ A* found path of length {len(path)}")
    else:
        print(f"   ⚠ A* returned no path (might be blocked)")
except Exception as e:
    print(f"   ✗ A* failed: {e}")
    sys.exit(1)

# Test 4: Action distribution
print("\n4. Testing action distribution...")
try:
    snake_pos = [(10, 10), (10, 9), (10, 8)]
    food_pos = (5, 15)
    action_probs = get_action_distribution(
        snake_pos, food_pos, width=32, height=32, use_astar=True
    )
    
    if action_probs.shape == (4,) and np.isclose(action_probs.sum(), 1.0):
        print(f"   ✓ Action probabilities: {action_probs}")
    else:
        print(f"   ✗ Invalid action distribution: {action_probs}")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Action distribution failed: {e}")
    sys.exit(1)

# Test 5: Augmentation
print("\n5. Testing augmentation...")
try:
    state = np.random.rand(32, 32, 3)
    action = 0  # Up
    
    aug_count = 0
    for aug_type in AUGMENTATIONS:
        aug_state, aug_action = augment_state_action(state, action, aug_type)
        if aug_state.shape == state.shape:
            aug_count += 1
    
    if aug_count == 8:
        print(f"   ✓ All {aug_count} augmentations work")
    else:
        print(f"   ✗ Only {aug_count}/8 augmentations work")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Augmentation failed: {e}")
    sys.exit(1)

# Test 6: Dataset generation (small)
print("\n6. Testing dataset generation (100 samples)...")
try:
    dataset = generate_pretraining_dataset(
        num_samples=100,
        width=32,
        height=32,
        use_astar=True,
        augment=False,  # Skip augmentation for speed
        seed=42
    )
    
    if len(dataset) >= 50:  # At least 50% success rate
        print(f"   ✓ Generated {len(dataset)} samples")
        
        # Check sample structure
        sample = dataset[0]
        assert 'state' in sample
        assert 'action' in sample
        assert 'action_probs' in sample
        assert 'metadata' in sample
        print(f"   ✓ Sample structure valid")
        print(f"      State shape: {sample['state'].shape}")
        print(f"      Action: {sample['action']}")
        print(f"      Snake length: {sample['metadata']['snake_length']}")
    else:
        print(f"   ✗ Only generated {len(dataset)} samples")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Dataset generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: PyTorch Dataset
print("\n7. Testing PyTorch Dataset...")
try:
    torch_dataset = SnakePretrainDataset(dataset[:10], use_soft_labels=True)
    sample = torch_dataset[0]
    
    assert isinstance(sample['state'], torch.Tensor)
    assert isinstance(sample['action'], torch.Tensor)
    assert isinstance(sample['action_probs'], torch.Tensor)
    assert sample['state'].shape == (32, 32, 3)
    assert sample['action_probs'].shape == (4,)
    
    print(f"   ✓ PyTorch Dataset works")
    print(f"      State dtype: {sample['state'].dtype}")
    print(f"      Action: {sample['action'].item()}")
except Exception as e:
    print(f"   ✗ PyTorch Dataset failed: {e}")
    sys.exit(1)

# Test 8: Model creation
print("\n8. Testing model creation...")
try:
    model = TransformerPolicyPretrainer(
        height=32, width=32,
        d_model=64, num_layers=2, num_heads=4, dropout=0.1
    )
    
    param_count = count_parameters(model)
    print(f"   ✓ Model created with {param_count:,} parameters")
    
    # Test forward pass
    batch = torch.randn(4, 32, 32, 3)
    logits = model(batch)
    
    assert logits.shape == (4, 4)
    print(f"   ✓ Forward pass works (input: {batch.shape}, output: {logits.shape})")
    
    # Test prediction
    action = model.predict_action(batch[0])
    assert 0 <= action <= 3
    print(f"   ✓ Prediction works (action: {action})")
    
except Exception as e:
    print(f"   ✗ Model test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: Training step (single batch)
print("\n9. Testing training step...")
try:
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    
    # Create small dataset
    small_dataset = SnakePretrainDataset(dataset[:20], use_soft_labels=False)
    loader = DataLoader(small_dataset, batch_size=8, shuffle=True)
    
    # Get a batch
    batch = next(iter(loader))
    
    # Forward pass
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    states = batch['state']
    actions = batch['action']
    
    logits = model(states)
    loss = F.cross_entropy(logits, actions)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"   ✓ Training step works")
    print(f"      Loss: {loss.item():.4f}")
    print(f"      Logits: {logits[0].detach().numpy()}")
    
except Exception as e:
    print(f"   ✗ Training step failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED!")
print("=" * 60)
print("\nYou're ready to run the full pipeline:")
print("  1. Generate dataset: python generate_dataset.py --num-samples 50000")
print("  2. Pretrain model:   python pretrain_snake.py --dataset <dataset.pkl>")
print("  3. Or use quickstart: ./quickstart_pretrain.sh")
print()
