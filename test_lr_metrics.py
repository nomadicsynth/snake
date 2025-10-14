"""
Test that learning rate is properly tracked in metrics
"""

import jax
import jax.numpy as jnp
from snake_jax.config import EnvConfig
from snake_jax.gymnax_wrapper import SnakeGymnaxWrapper
from purejaxrl.purejaxrl.wrappers import LogWrapper
from train_snake_purejaxrl_impl import make_train_step

def test_lr_in_metrics():
    """Test that learning rate appears in metrics"""
    
    # Simple config
    config = {
        "NUM_ENVS": 4,
        "NUM_STEPS": 8,
        "TOTAL_TIMESTEPS": 1000,
        "UPDATE_EPOCHS": 2,
        "NUM_MINIBATCHES": 2,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "LR": 0.001,
        "ANNEAL_LR": True,  # Enable LR annealing
        "D_MODEL": 32,
        "NUM_LAYERS": 1,
        "NUM_HEADS": 2,
        "DROPOUT": 0.0,
        "USE_MUON": False,
    }
    
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    
    # Create environment
    env_config = EnvConfig(width=5, height=5, max_steps=50)
    base_env = SnakeGymnaxWrapper(env_config)
    env = LogWrapper(base_env)
    env_params = env.default_params
    
    # Create training functions
    init_fn, make_update_fn = make_train_step(config, env, env_params)
    
    # Initialize
    rng = jax.random.PRNGKey(0)
    network, train_state, env_state, obsv, rng = init_fn(rng)
    runner_state = (train_state, env_state, obsv, rng)
    
    # Create update function
    update_fn = make_update_fn(network)
    
    print("Testing learning rate tracking...")
    
    # Run a few updates
    for update_idx in range(3):
        runner_state, metrics = update_fn(runner_state, update_idx)
        
        # Metrics is a tuple: (env_metrics, loss_metrics)
        env_metrics, loss_metrics = metrics
        
        # Transfer from GPU
        loss_metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), loss_metrics)
        
        # Check that learning rate is in metrics
        assert 'learning_rate' in loss_metrics, f"learning_rate not found in metrics! Keys: {loss_metrics.keys()}"
        
        lr = float(loss_metrics['learning_rate'])
        expected_lr = config['LR'] * (1.0 - update_idx / config['NUM_UPDATES'])
        
        print(f"  Update {update_idx}: LR = {lr:.6f} (expected: {expected_lr:.6f})")
        
        # Check LR is decreasing (annealing)
        assert abs(lr - expected_lr) < 1e-5, f"LR mismatch! Got {lr}, expected {expected_lr}"
    
    print("✅ Learning rate tracking test passed!")
    print()
    
    # Test with Muon optimizer
    print("Testing with Muon optimizer...")
    config_muon = config.copy()
    config_muon["USE_MUON"] = True
    config_muon["MUON_LR"] = 0.02
    config_muon["AUX_ADAM_LR"] = 0.0005
    config_muon["ANNEAL_LR"] = True  # Enable annealing for Muon
    
    try:
        from muon_jax import chain_with_muon
        
        # Reinitialize with Muon
        init_fn_muon, make_update_fn_muon = make_train_step(config_muon, env, env_params)
        rng = jax.random.PRNGKey(1)
        network, train_state, env_state, obsv, rng = init_fn_muon(rng)
        runner_state = (train_state, env_state, obsv, rng)
        update_fn_muon = make_update_fn_muon(network)
        
        # Run multiple updates to verify annealing
        for update_idx in range(3):
            runner_state, metrics = update_fn_muon(runner_state, update_idx)
            env_metrics, loss_metrics = metrics
            loss_metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), loss_metrics)
            
            # Check both learning rates are present
            assert 'learning_rate' in loss_metrics, "learning_rate not found with Muon!"
            assert 'aux_learning_rate' in loss_metrics, "aux_learning_rate not found with Muon!"
            
            lr = float(loss_metrics['learning_rate'])
            aux_lr = float(loss_metrics['aux_learning_rate'])
            
            # Calculate expected LRs with annealing
            frac = 1.0 - update_idx / config_muon['NUM_UPDATES']
            expected_muon_lr = config_muon['MUON_LR'] * frac
            expected_aux_lr = config_muon['AUX_ADAM_LR'] * frac
            
            print(f"  Update {update_idx}:")
            print(f"    Muon LR: {lr:.6f} (expected: {expected_muon_lr:.6f})")
            print(f"    Aux LR:  {aux_lr:.6f} (expected: {expected_aux_lr:.6f})")
            
            assert abs(lr - expected_muon_lr) < 1e-5, f"Muon LR mismatch! Got {lr}, expected {expected_muon_lr}"
            assert abs(aux_lr - expected_aux_lr) < 1e-5, f"Aux LR mismatch! Got {aux_lr}, expected {expected_aux_lr}"
        
        print("✅ Muon optimizer LR annealing test passed!")
    except ImportError:
        print("⚠️  Muon not available, skipping Muon test")
    
    print()
    print("=" * 60)
    print("ALL TESTS PASSED! ✅")
    print("=" * 60)


if __name__ == "__main__":
    test_lr_in_metrics()
