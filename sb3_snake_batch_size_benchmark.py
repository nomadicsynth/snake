import time
import torch
import argparse
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sb3_snake import SnakeEnv, TransformerExtractor
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
from contextlib import nullcontext

def quick_benchmark(use_compile=False, sdpa_backend=None, use_amp=False, fast_mode=False):
    """Benchmark batch sizes optimized for RTX 4090.
    
    Args:
        use_compile: Whether to use torch.compile() for potential speedup
        sdpa_backend: SDPA backend to use. Options: None (auto), 'flash_attention', 'efficient_attention', 'math'
        use_amp: Whether to use automatic mixed precision (bf16) for FlashAttention support
        fast_mode: If True, use reduced steps for faster benchmarking (~3x faster)
    """
    # Focus on larger batch sizes for 4090
    batch_sizes = [128, 256, 512, 1024, 2048, 4096]
    
    # Configure benchmark parameters based on mode
    if fast_mode:
        timesteps = 1500
        warmup_steps = 500
        learning_starts = 500
        buffer_size = 10000
        print("🚀 FAST MODE: Using reduced steps for quicker benchmarking")
    else:
        timesteps = 5000
        warmup_steps = 1536
        learning_starts = 1000
        buffer_size = 100000
        print("📊 FULL MODE: Using full benchmark for accurate measurements")
    
    results = {}
    
    compile_str = " (torch.compile)" if use_compile else ""
    backend_str = f" ({sdpa_backend})" if sdpa_backend else ""
    amp_str = " (bf16)" if use_amp else ""
    
    # AMP context for mixed precision
    amp_context = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if use_amp else nullcontext()
    
    # Helper function to create SDPA context (must be fresh each time)
    def get_sdpa_context():
        if sdpa_backend and hasattr(torch.nn.attention, 'sdpa_kernel'):
            from torch.nn.attention import SDPBackend, sdpa_kernel
            if sdpa_backend == 'flash_attention':
                return sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])
            elif sdpa_backend == 'efficient_attention':
                return sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION])
            elif sdpa_backend == 'math':
                return sdpa_kernel(backends=[SDPBackend.MATH])
        return nullcontext()
    
    # Overall progress bar
    overall_pbar = tqdm(batch_sizes, desc=f"Overall Progress{compile_str}{backend_str}{amp_str}", position=0)
    
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
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            train_freq=4,
            gradient_steps=1,
            verbose=0,
            device="cuda",
        )
        
        # Note: We use autocast context manager instead of converting model dtype
        # This handles mixed precision automatically without dtype mismatch errors
        
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
        with get_sdpa_context():
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    model.learn(total_timesteps=warmup_steps, progress_bar=True)
            else:
                model.learn(total_timesteps=warmup_steps, progress_bar=True)
        
        # Track gradient updates
        initial_num_timesteps = model.num_timesteps
        initial_train_calls = model._n_updates
        
        # Actual benchmark
        print("Running benchmark...")
        torch.cuda.synchronize()
        start = time.time()
        with get_sdpa_context():
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    model.learn(total_timesteps=timesteps, progress_bar=True)
            else:
                model.learn(total_timesteps=timesteps, progress_bar=True)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        # Calculate actual metrics
        actual_timesteps = model.num_timesteps - initial_num_timesteps
        num_updates = model._n_updates - initial_train_calls
        samples_processed = num_updates * batch_size  # Total samples used for gradient updates
        
        # Get memory stats
        max_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        env_steps_per_sec = actual_timesteps / elapsed
        samples_per_sec = samples_processed / elapsed
        updates_per_sec = num_updates / elapsed
        
        results[batch_size] = {
            'time': elapsed,
            'env_steps_per_sec': env_steps_per_sec,
            'samples_per_sec': samples_per_sec,
            'updates_per_sec': updates_per_sec,
            'num_updates': num_updates,
            'samples_processed': samples_processed,
            'max_memory_gb': max_memory,
            'compiled': use_compile,
            'sdpa_backend': sdpa_backend or 'auto',
            'use_amp': use_amp,
        }
        
        print(f"Time: {elapsed:.2f}s")
        print(f"Env steps/sec: {env_steps_per_sec:.1f}")
        print(f"Gradient updates: {num_updates}")
        print(f"Samples/sec: {samples_per_sec:.1f}")
        print(f"Updates/sec: {updates_per_sec:.1f}")
        print(f"Max GPU memory: {max_memory:.2f} GB")
        
        del model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Close overall progress bar
    overall_pbar.close()
    
    # Summary
    print("\n" + "="*80)
    print("BENCHMARK RESULTS (sorted by training throughput)")
    print("="*80)
    print(f"{'Batch':<8} {'Samples/s':<12} {'Updates/s':<12} {'Env Steps/s':<12} {'GPU Mem (GB)':<15}")
    print("-"*80)
    
    for bs, metrics in sorted(results.items(), key=lambda x: x[1]['samples_per_sec'], reverse=True):
        print(f"{bs:<8} {metrics['samples_per_sec']:<12.1f} {metrics['updates_per_sec']:<12.1f} {metrics['env_steps_per_sec']:<12.1f} {metrics['max_memory_gb']:<15.2f}")
    
    best_bs = max(results.items(), key=lambda x: x[1]['samples_per_sec'])[0]
    print("\n" + "="*80)
    print(f"🏆 OPTIMAL BATCH SIZE (max samples/sec): {best_bs}")
    print("="*80)
    
    return results


def save_results_to_file(all_results, fast_mode=False):
    """Save benchmark results to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "fast" if fast_mode else "full"
    filename = f"benchmark_results_{mode}_{timestamp}.json"
    
    output = {
        "timestamp": timestamp,
        "mode": mode,
        "gpu": torch.cuda.get_device_name(0),
        "pytorch_version": torch.__version__,
        "results": all_results
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")
    return filename


def save_best_config_to_env(all_results):
    """Save the best configuration to .env file, preserving existing vars."""
    # Find the best overall configuration by peak performance
    best_config = None
    best_batch_size = None
    best_samples_per_sec = 0
    best_full_result = None
    
    for config_name, results in all_results.items():
        # Find the best batch size for this config
        for batch_size, metrics in results.items():
            if metrics['samples_per_sec'] > best_samples_per_sec:
                best_samples_per_sec = metrics['samples_per_sec']
                best_config = config_name
                best_batch_size = batch_size
                best_full_result = metrics
    
    if not best_config:
        return
    
    # Determine settings from config name
    use_compile = 'compile' in best_config.lower()
    use_flash = 'flash' in best_config.lower()
    use_amp = 'bf16' in best_config.lower() or 'flash' in best_config.lower()
    
    # Read existing .env and filter out our benchmark vars
    existing_vars = {}
    benchmark_vars = {
        'OPTIMAL_BATCH_SIZE', 'USE_TORCH_COMPILE', 'USE_FLASH_ATTENTION', 
        'USE_BF16', 'PEAK_THROUGHPUT_SAMPLES_PER_SEC', 'THROUGHPUT_UPDATES_PER_SEC', 
        'GPU_MEMORY_GB'
    }
    
    if Path('.env').exists():
        with open('.env', 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                # Parse key=value
                if '=' in line:
                    key = line.split('=')[0].strip()
                    # Only keep non-benchmark vars
                    if key not in benchmark_vars:
                        existing_vars[key] = line
    
    # Build new .env content
    env_lines = []
    
    # Add existing vars first
    if existing_vars:
        env_lines.append("# Existing configuration")
        for line in existing_vars.values():
            env_lines.append(line)
        env_lines.append("")
    
    # Add benchmark results
    env_lines.append(f"# Optimal training settings from benchmark run {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    env_lines.append(f"# Best configuration: {best_config}")
    env_lines.append(f"# Peak throughput: {best_samples_per_sec:.1f} samples/sec")
    env_lines.append("")
    env_lines.append("# Batch size")
    env_lines.append(f"OPTIMAL_BATCH_SIZE={best_batch_size}")
    env_lines.append("")
    env_lines.append("# Optimization flags")
    env_lines.append(f"USE_TORCH_COMPILE={'true' if use_compile else 'false'}")
    env_lines.append(f"USE_FLASH_ATTENTION={'true' if use_flash else 'false'}")
    env_lines.append(f"USE_BF16={'true' if use_amp else 'false'}")
    env_lines.append("")
    env_lines.append("# Performance metrics")
    env_lines.append(f"PEAK_THROUGHPUT_SAMPLES_PER_SEC={best_samples_per_sec:.1f}")
    env_lines.append(f"THROUGHPUT_UPDATES_PER_SEC={best_full_result['updates_per_sec']:.1f}")
    env_lines.append(f"GPU_MEMORY_GB={best_full_result['max_memory_gb']:.2f}")
    
    # Write to .env
    with open('.env', 'w') as f:
        f.write('\n'.join(env_lines) + '\n')
    
    print(f"\n📝 Best configuration saved to .env:")
    print(f"   Config: {best_config}")
    print(f"   Batch size: {best_batch_size}")
    print(f"   torch.compile: {use_compile}")
    print(f"   FlashAttention: {use_flash}")
    print(f"   BF16: {use_amp}")
    print(f"   Peak throughput: {best_samples_per_sec:.1f} samples/sec")
    

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='Benchmark DQN training with different batch sizes and optimizations')
        parser.add_argument('--fast', action='store_true', help='Use fast mode with reduced steps (~3x faster)')
        args = parser.parse_args()
        
        print("RTX 4090 Batch Size Benchmark")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"PyTorch version: {torch.__version__}\n")
        
        # Check available features
        has_compile = hasattr(torch, 'compile')
        has_sdpa = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        has_flash = hasattr(torch.nn.attention, 'sdpa_kernel')
        has_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
        
        print("Available optimizations:")
        print(f"  - torch.compile(): {'✓' if has_compile else '✗'}")
        print(f"  - SDPA (Scaled Dot-Product Attention): {'✓' if has_sdpa else '✗'}")
        print(f"  - FlashAttention backend control: {'✓' if has_flash else '✗'}")
        print(f"  - BFloat16 (for FlashAttention): {'✓' if has_bf16 else '✗'}")
        print()
        
        if has_compile and has_flash and has_bf16:
            # Test all combinations with bf16 support
            all_results = {}
            
            print("=" * 80)
            print("Test 1/4: Baseline (fp32, no compile, auto SDPA)")
            print("=" * 80)
            results_baseline = quick_benchmark(use_compile=False, sdpa_backend=None, use_amp=False, fast_mode=args.fast)
            all_results['baseline_fp32'] = results_baseline
            
            print("\n\n" + "=" * 80)
            print("Test 2/4: BF16 + FlashAttention only")
            print("=" * 80)
            results_flash = quick_benchmark(use_compile=False, sdpa_backend='flash_attention', use_amp=True, fast_mode=args.fast)
            all_results['bf16_flash'] = results_flash
            
            print("\n\n" + "=" * 80)
            print("Test 3/4: BF16 + torch.compile() (auto SDPA)")
            print("=" * 80)
            results_compile = quick_benchmark(use_compile=True, sdpa_backend=None, use_amp=True, fast_mode=args.fast)
            all_results['bf16_compile'] = results_compile
            
            print("\n\n" + "=" * 80)
            print("Test 4/4: BF16 + torch.compile() + FlashAttention")
            print("=" * 80)
            results_compile_flash = quick_benchmark(use_compile=True, sdpa_backend='flash_attention', use_amp=True, fast_mode=args.fast)
            all_results['bf16_compile_flash'] = results_compile_flash
            
            # Comprehensive comparison
            print("\n\n" + "=" * 80)
            print("COMPREHENSIVE COMPARISON")
            print("=" * 80)
            print(f"{'Batch':<8} {'Baseline':<12} {'Flash':<12} {'Compile':<12} {'Both':<12} {'Best Speedup':<15}")
            print("-" * 80)
            
            for bs in results_baseline.keys():
                baseline = results_baseline[bs]['samples_per_sec']
                flash = results_flash[bs]['samples_per_sec']
                compile = results_compile[bs]['samples_per_sec']
                both = results_compile_flash[bs]['samples_per_sec']
                
                best = max(baseline, flash, compile, both)
                speedup = (best / baseline - 1) * 100
                
                print(f"{bs:<8} {baseline:<12.1f} {flash:<12.1f} {compile:<12.1f} {both:<12.1f} {speedup:+.1f}%")
            
            # Overall comparison
            avg_baseline = sum(r['samples_per_sec'] for r in results_baseline.values()) / len(results_baseline)
            avg_flash = sum(r['samples_per_sec'] for r in results_flash.values()) / len(results_flash)
            avg_compile = sum(r['samples_per_sec'] for r in results_compile.values()) / len(results_compile)
            avg_both = sum(r['samples_per_sec'] for r in results_compile_flash.values()) / len(results_compile_flash)
            
            print("\n" + "=" * 80)
            print("AVERAGE PERFORMANCE (samples/sec = training throughput):")
            print(f"  Baseline (fp32):             {avg_baseline:.1f} samples/sec")
            print(f"  BF16 + FlashAttention:       {avg_flash:.1f} samples/sec ({(avg_flash/avg_baseline-1)*100:+.1f}%)")
            print(f"  BF16 + torch.compile():      {avg_compile:.1f} samples/sec ({(avg_compile/avg_baseline-1)*100:+.1f}%)")
            print(f"  BF16 + Both combined:        {avg_both:.1f} samples/sec ({(avg_both/avg_baseline-1)*100:+.1f}%)")
            print("=" * 80)
            
            # Save results
            save_results_to_file(all_results, fast_mode=args.fast)
            save_best_config_to_env(all_results)
            
        elif has_compile:
            # Only compile available
            all_results = {}
            
            print("=" * 80)
            print("Running benchmark WITHOUT torch.compile()")
            print("=" * 80)
            results_no_compile = quick_benchmark(use_compile=False, fast_mode=args.fast)
            all_results['no_compile'] = results_no_compile
            
            print("\n\n" + "=" * 80)
            print("Running benchmark WITH torch.compile()")
            print("=" * 80)
            results_compile = quick_benchmark(use_compile=True, fast_mode=args.fast)
            all_results['compile'] = results_compile
            
            # Comparison
            print("\n\n" + "=" * 80)
            print("COMPARISON: torch.compile() vs Normal")
            print("=" * 80)
            print(f"{'Batch Size':<12} {'Normal (samp/s)':<18} {'Compiled (samp/s)':<18} {'Speedup':<12}")
            print("-" * 80)
            
            for bs in results_no_compile.keys():
                normal_sps = results_no_compile[bs]['samples_per_sec']
                compiled_sps = results_compile[bs]['samples_per_sec']
                speedup = (compiled_sps / normal_sps - 1) * 100
                speedup_str = f"{speedup:+.1f}%"
                
                print(f"{bs:<12} {normal_sps:<18.1f} {compiled_sps:<18.1f} {speedup_str:<12}")
            
            # Overall speedup
            avg_normal = sum(r['samples_per_sec'] for r in results_no_compile.values()) / len(results_no_compile)
            avg_compiled = sum(r['samples_per_sec'] for r in results_compile.values()) / len(results_compile)
            overall_speedup = (avg_compiled / avg_normal - 1) * 100
            
            print("\n" + "=" * 80)
            print(f"Average speedup with torch.compile(): {overall_speedup:+.1f}%")
            print("=" * 80)
            
            # Save results
            save_results_to_file(all_results, fast_mode=args.fast)
            save_best_config_to_env(all_results)
            
        else:
            print(f"⚠️  torch.compile() not available (requires PyTorch 2.0+)")
            print("Running benchmark without torch.compile()\n")
            all_results = {}
            results = quick_benchmark(use_compile=False, fast_mode=args.fast)
            all_results['baseline'] = results
            
            # Save results
            save_results_to_file(all_results, fast_mode=args.fast)
            save_best_config_to_env(all_results)
    except KeyboardInterrupt:
        print("\nBenchmarking interrupted by user.")
        print("Exiting gracefully...")
 