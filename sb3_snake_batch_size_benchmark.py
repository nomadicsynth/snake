import time
import torch
from tqdm import tqdm
from sb3_snake import SnakeEnv, TransformerExtractor
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
from contextlib import nullcontext

def quick_benchmark(use_compile=False, sdpa_backend=None):
    """Benchmark batch sizes optimized for RTX 4090.
    
    Args:
        use_compile: Whether to use torch.compile() for potential speedup
        sdpa_backend: SDPA backend to use. Options: None (auto), 'flash_attention', 'efficient_attention', 'math'
    """
    # Focus on larger batch sizes for 4090
    batch_sizes = [128, 256, 512, 1024, 2048, 4096]
    timesteps = 20000
    results = {}
    
    compile_str = " (torch.compile)" if use_compile else ""
    backend_str = f" ({sdpa_backend})" if sdpa_backend else ""
    
    # Set SDPA backend if specified
    sdpa_context = nullcontext()
    if sdpa_backend and hasattr(torch.backends.cuda, 'sdp_kernel'):
        backends = {
            'flash_attention': torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=False
            ),
            'efficient_attention': torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=False, enable_mem_efficient=True
            ),
            'math': torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False
            ),
        }
        sdpa_context = backends.get(sdpa_backend, nullcontext())
    
    # Overall progress bar
    overall_pbar = tqdm(batch_sizes, desc=f"Overall Progress{compile_str}{backend_str}", position=0)
    
    for batch_size in overall_pbar:
        overall_pbar.set_description(f"Overall Progress (Testing batch_size={batch_size})")
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
        
        # Apply torch.compile if requested
        if use_compile:
            print("Compiling model with torch.compile()...")
            # Compile the policy network
            model.policy.features_extractor = torch.compile(
                model.policy.features_extractor,
                mode="reduce-overhead",  # Options: "default", "reduce-overhead", "max-autotune"
            )
            model.q_net = torch.compile(model.q_net, mode="reduce-overhead")
            if hasattr(model, 'q_net_target'):
                model.q_net_target = torch.compile(model.q_net_target, mode="reduce-overhead")
        
        # Warmup
        print("Running warmup...")
        with sdpa_context:
            model.learn(total_timesteps=1000, progress_bar=True)
        
        # Actual benchmark
        print("Running benchmark...")
        torch.cuda.synchronize()
        start = time.time()
        with sdpa_context:
            model.learn(total_timesteps=timesteps, progress_bar=True)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        # Get memory stats
        max_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        steps_per_sec = timesteps / elapsed
        results[batch_size] = {
            'time': elapsed,
            'steps_per_sec': steps_per_sec,
            'max_memory_gb': max_memory,
            'compiled': use_compile,
            'sdpa_backend': sdpa_backend or 'auto',
        }
        
        print(f"Time: {elapsed:.2f}s")
        print(f"Steps/sec: {steps_per_sec:.1f}")
        print(f"Max GPU memory: {max_memory:.2f} GB")
        
        del model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Close overall progress bar
    overall_pbar.close()
    
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
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"PyTorch version: {torch.__version__}\n")
    
    # Check available features
    has_compile = hasattr(torch, 'compile')
    has_sdpa = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    has_flash = has_sdpa and hasattr(torch.backends.cuda, 'sdp_kernel')
    
    print("Available optimizations:")
    print(f"  - torch.compile(): {'✓' if has_compile else '✗'}")
    print(f"  - SDPA (Scaled Dot-Product Attention): {'✓' if has_sdpa else '✗'}")
    print(f"  - FlashAttention backend: {'✓' if has_flash else '✗'}")
    print()
    
    if has_compile and has_flash:
        # Test all combinations
        print("=" * 80)
        print("Test 1/4: Baseline (no compile, auto SDPA)")
        print("=" * 80)
        results_baseline = quick_benchmark(use_compile=False, sdpa_backend=None)
        
        print("\n\n" + "=" * 80)
        print("Test 2/4: FlashAttention only")
        print("=" * 80)
        results_flash = quick_benchmark(use_compile=False, sdpa_backend='flash_attention')
        
        print("\n\n" + "=" * 80)
        print("Test 3/4: torch.compile() only (auto SDPA)")
        print("=" * 80)
        results_compile = quick_benchmark(use_compile=True, sdpa_backend=None)
        
        print("\n\n" + "=" * 80)
        print("Test 4/4: torch.compile() + FlashAttention")
        print("=" * 80)
        results_compile_flash = quick_benchmark(use_compile=True, sdpa_backend='flash_attention')
        
        # Comprehensive comparison
        print("\n\n" + "=" * 80)
        print("COMPREHENSIVE COMPARISON")
        print("=" * 80)
        print(f"{'Batch':<8} {'Baseline':<12} {'Flash':<12} {'Compile':<12} {'Both':<12} {'Best Speedup':<15}")
        print("-" * 80)
        
        for bs in results_baseline.keys():
            baseline = results_baseline[bs]['steps_per_sec']
            flash = results_flash[bs]['steps_per_sec']
            compile = results_compile[bs]['steps_per_sec']
            both = results_compile_flash[bs]['steps_per_sec']
            
            best = max(baseline, flash, compile, both)
            speedup = (best / baseline - 1) * 100
            
            print(f"{bs:<8} {baseline:<12.1f} {flash:<12.1f} {compile:<12.1f} {both:<12.1f} {speedup:+.1f}%")
        
        # Overall comparison
        avg_baseline = sum(r['steps_per_sec'] for r in results_baseline.values()) / len(results_baseline)
        avg_flash = sum(r['steps_per_sec'] for r in results_flash.values()) / len(results_flash)
        avg_compile = sum(r['steps_per_sec'] for r in results_compile.values()) / len(results_compile)
        avg_both = sum(r['steps_per_sec'] for r in results_compile_flash.values()) / len(results_compile_flash)
        
        print("\n" + "=" * 80)
        print("AVERAGE PERFORMANCE:")
        print(f"  Baseline:              {avg_baseline:.1f} steps/sec")
        print(f"  FlashAttention:        {avg_flash:.1f} steps/sec ({(avg_flash/avg_baseline-1)*100:+.1f}%)")
        print(f"  torch.compile():       {avg_compile:.1f} steps/sec ({(avg_compile/avg_baseline-1)*100:+.1f}%)")
        print(f"  Both combined:         {avg_both:.1f} steps/sec ({(avg_both/avg_baseline-1)*100:+.1f}%)")
        print("=" * 80)
        
    elif has_compile:
        # Only compile available
        print("=" * 80)
        print("Running benchmark WITHOUT torch.compile()")
        print("=" * 80)
        results_no_compile = quick_benchmark(use_compile=False)
        
        print("\n\n" + "=" * 80)
        print("Running benchmark WITH torch.compile()")
        print("=" * 80)
        results_compile = quick_benchmark(use_compile=True)
        
        # Comparison
        print("\n\n" + "=" * 80)
        print("COMPARISON: torch.compile() vs Normal")
        print("=" * 80)
        print(f"{'Batch Size':<12} {'Normal (s/s)':<15} {'Compiled (s/s)':<15} {'Speedup':<12}")
        print("-" * 80)
        
        for bs in results_no_compile.keys():
            normal_sps = results_no_compile[bs]['steps_per_sec']
            compiled_sps = results_compile[bs]['steps_per_sec']
            speedup = (compiled_sps / normal_sps - 1) * 100
            speedup_str = f"{speedup:+.1f}%"
            
            print(f"{bs:<12} {normal_sps:<15.1f} {compiled_sps:<15.1f} {speedup_str:<12}")
        
        # Overall speedup
        avg_normal = sum(r['steps_per_sec'] for r in results_no_compile.values()) / len(results_no_compile)
        avg_compiled = sum(r['steps_per_sec'] for r in results_compile.values()) / len(results_compile)
        overall_speedup = (avg_compiled / avg_normal - 1) * 100
        
        print("\n" + "=" * 80)
        print(f"Average speedup with torch.compile(): {overall_speedup:+.1f}%")
        print("=" * 80)
        
    else:
        print(f"⚠️  torch.compile() not available (requires PyTorch 2.0+)")
        print("Running benchmark without torch.compile()\n")
        results = quick_benchmark(use_compile=False)