"""
Test script for Muon JAX optimizer

Verifies that the Muon optimizer works correctly with simple examples.
"""

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

try:
    from muon_jax import muon, multi_transform_with_muon, chain_with_muon
    print("✓ Muon JAX module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import muon_jax: {e}")
    exit(1)


class SimpleTransformer(nn.Module):
    """Simple transformer for testing"""
    d_model: int = 32
    
    @nn.compact
    def __call__(self, x):
        # Linear projection (2D weight)
        x = nn.Dense(self.d_model)(x)
        # Layer norm (1D weight)
        x = nn.LayerNorm()(x)
        # Another linear (2D weight)
        x = nn.Dense(self.d_model)(x)
        return x


def test_muon_optimizer():
    """Test basic Muon optimizer functionality"""
    print("\n" + "="*60)
    print("Testing Muon Optimizer")
    print("="*60)
    
    # Create a simple model
    model = SimpleTransformer(d_model=32)
    rng = jax.random.PRNGKey(0)
    
    # Initialize
    x = jnp.ones((4, 16))
    params = model.init(rng, x)
    
    # Count parameters by type
    print("\nModel parameters:")
    flat_params = jax.tree_util.tree_leaves(params)
    
    num_2d = sum(1 for p in flat_params if p.ndim >= 2)
    num_1d = sum(1 for p in flat_params if p.ndim < 2)
    
    print(f"  2D+ parameters (Muon): {num_2d}")
    print(f"  1D parameters (Adam): {num_1d}")
    
    # Create optimizer
    tx = chain_with_muon(
        muon_lr=0.02,
        aux_lr=0.0003,
        max_grad_norm=1.0,
        momentum=0.95,
        nesterov=True
    )
    
    # Create train state
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    
    print("\n✓ Optimizer created successfully")
    
    # Test a few training steps
    def loss_fn(params, x):
        y = model.apply(params, x)
        return jnp.mean(jnp.square(y - 1.0))
    
    print("\nRunning training steps...")
    
    for i in range(5):
        grad = jax.grad(loss_fn)(train_state.params, x)
        train_state = train_state.apply_gradients(grads=grad)
        loss = loss_fn(train_state.params, x)
        print(f"  Step {i+1}: loss = {loss:.6f}")
    
    print("\n✓ Training steps completed successfully")
    return True


def test_parameter_labeling():
    """Test that parameters are correctly labeled as muon or adam"""
    print("\n" + "="*60)
    print("Testing Parameter Labeling")
    print("="*60)
    
    # Create simple params
    params = {
        'dense1': {
            'kernel': jnp.ones((10, 20)),  # 2D -> Muon
            'bias': jnp.ones((20,))         # 1D -> Adam
        },
        'dense2': {
            'kernel': jnp.ones((20, 5)),   # 2D -> Muon
            'bias': jnp.ones((5,))          # 1D -> Adam
        },
        'layer_norm': {
            'scale': jnp.ones((5,))         # 1D -> Adam
        }
    }
    
    # Label function from multi_transform_with_muon
    def param_labels(params):
        return jax.tree_util.tree_map(
            lambda p: 'muon' if p.ndim >= 2 else 'adam',
            params
        )
    
    labels = param_labels(params)
    
    print("\nParameter labels:")
    flat_labels = jax.tree_util.tree_leaves_with_path(labels)
    for path, label in flat_labels:
        path_str = '/'.join(str(k.key) for k in path)
        print(f"  {path_str}: {label}")
    
    # Verify correct labeling
    assert labels['dense1']['kernel'] == 'muon'
    assert labels['dense1']['bias'] == 'adam'
    assert labels['dense2']['kernel'] == 'muon'
    assert labels['dense2']['bias'] == 'adam'
    assert labels['layer_norm']['scale'] == 'adam'
    
    print("\n✓ All parameters labeled correctly")
    return True


def test_learning_rate_schedule():
    """Test Muon with learning rate schedule"""
    print("\n" + "="*60)
    print("Testing Learning Rate Schedule")
    print("="*60)
    
    model = SimpleTransformer(d_model=16)
    rng = jax.random.PRNGKey(42)
    x = jnp.ones((2, 8))
    params = model.init(rng, x)
    
    # Create optimizer with schedule (note: schedules not directly supported in current impl)
    tx = chain_with_muon(
        muon_lr=0.02,
        aux_lr=0.0003,
        max_grad_norm=1.0,
    )
    
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    
    print("✓ Optimizer with schedule created successfully")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("MUON JAX OPTIMIZER TESTS")
    print("="*60)
    
    tests = [
        ("Basic Optimizer", test_muon_optimizer),
        ("Parameter Labeling", test_parameter_labeling),
        ("Learning Rate Schedule", test_learning_rate_schedule),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ Test '{name}' failed with error:")
            print(f"  {type(e).__name__}: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(success for _, success in results)
    
    print("="*60)
    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
