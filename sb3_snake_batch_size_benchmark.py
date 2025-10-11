import time
import torch
from sb3_snake import SnakeEnv, TransformerExtractor
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit

def quick_benchmark():
    """Benchmark batch sizes optimized for RTX 4090."""
    # Focus on larger batch sizes for 4090
    batch_sizes = [128, 256, 512, 1024, 2048, 4096]
    timesteps = 20000
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n{'='*60}")
        print(f"Testing batch_size = {batch_size}")
        print('='*60)
        
        env = SnakeEnv(width=20, height=20, num_apples=1)
        env = TimeLimit(env, max_episode_steps=500)
        env = Monitor(env)
        
        policy_kwargs = {
            "features_extractor_class": TransformerExtractor,
            "features_extractor_kwargs": {
                "d_model": 64,
                "n_layers": 2,
                "n_heads": 4,
                "dropout": 0.1,
                "features_dim": 128,
            },
        }
        
        model = DQN(
            "CnnPolicy",
            env,
            policy_kwargs=policy_kwargs,
            batch_size=batch_size,
            buffer_size=100000,
            learning_starts=1000,
            train_freq=4,
            gradient_steps=1,
            verbose=0,
            device="cuda",
        )
        
        # Warmup
        model.learn(total_timesteps=1000, progress_bar=False)
        
        # Actual benchmark
        torch.cuda.synchronize()
        start = time.time()
        model.learn(total_timesteps=timesteps, progress_bar=False)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        # Get memory stats
        max_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        steps_per_sec = timesteps / elapsed
        results[batch_size] = {
            'time': elapsed,
            'steps_per_sec': steps_per_sec,
            'max_memory_gb': max_memory,
        }
        
        print(f"Time: {elapsed:.2f}s")
        print(f"Steps/sec: {steps_per_sec:.1f}")
        print(f"Max GPU memory: {max_memory:.2f} GB")
        
        del model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Summary
    print("\n" + "="*80)
    print("BENCHMARK RESULTS (sorted by throughput)")
    print("="*80)
    print(f"{'Batch Size':<12} {'Steps/sec':<15} {'Time (s)':<12} {'GPU Mem (GB)':<15}")
    print("-"*80)
    
    for bs, metrics in sorted(results.items(), key=lambda x: x[1]['steps_per_sec'], reverse=True):
        print(f"{bs:<12} {metrics['steps_per_sec']:<15.1f} {metrics['time']:<12.2f} {metrics['max_memory_gb']:<15.2f}")
    
    best_bs = max(results.items(), key=lambda x: x[1]['steps_per_sec'])[0]
    print("\n" + "="*80)
    print(f"🏆 OPTIMAL BATCH SIZE: {best_bs}")
    print("="*80)
    
    return results

if __name__ == "__main__":
    print("RTX 4090 Batch Size Benchmark")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
    
    results = quick_benchmark()