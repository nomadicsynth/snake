#!/usr/bin/env python3
"""
Benchmark CNN modes against vanilla Transformer

This script trains the Snake agent with:
1. No CNN (vanilla Transformer)
2. CNN in replace mode
3. CNN in append mode

Then generates a comparison report.
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import argparse


def run_training(config_name, args_list, seeds=[42, 43, 44]):
    """
    Run training with given configuration across multiple seeds
    
    Args:
        config_name: Name for this configuration
        args_list: List of command-line arguments
        seeds: Random seeds to use for multiple runs
    
    Returns:
        List of result dictionaries
    """
    results = []
    
    for seed in seeds:
        print(f"\n{'='*70}")
        print(f"Training {config_name} (seed={seed})")
        print(f"{'='*70}\n")
        
        # Build command
        cmd = [
            "/mnt/embiggen/ai-stuff/snake/.venv/bin/python",
            "train_snake_purejaxrl.py",
            "--seed", str(seed),
            "--wandb",  # Enable wandb for tracking
            "--wandb-project", "snake-cnn-benchmark",
            "--run-name", f"{config_name}_seed{seed}",
        ] + args_list
        
        print(f"Command: {' '.join(cmd)}\n")
        
        start_time = time.time()
        
        try:
            # Run training
            result = subprocess.run(
                cmd,
                cwd="/mnt/embiggen/ai-stuff/snake",
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            elapsed_time = time.time() - start_time
            
            # Parse output for metrics
            output_lines = result.stdout.split('\n')
            
            metrics = {
                'config': config_name,
                'seed': seed,
                'elapsed_time': elapsed_time,
                'success': result.returncode == 0,
                'error': None if result.returncode == 0 else result.stderr,
            }
            
            # Extract key metrics from output
            for line in output_lines:
                if 'Training FPS:' in line:
                    try:
                        fps = line.split('Training FPS:')[1].strip().replace(',', '')
                        metrics['training_fps'] = float(fps)
                    except:
                        pass
                elif 'Best eval return:' in line:
                    try:
                        eval_return = line.split('Best eval return:')[1].strip()
                        metrics['best_eval_return'] = float(eval_return)
                    except:
                        pass
                elif 'Time per update:' in line:
                    try:
                        time_per_update = line.split('Time per update:')[1].strip().replace('s', '')
                        metrics['time_per_update'] = float(time_per_update)
                    except:
                        pass
            
            results.append(metrics)
            
            print(f"\n✅ Completed {config_name} (seed={seed})")
            print(f"   Time: {elapsed_time:.2f}s")
            if 'best_eval_return' in metrics:
                print(f"   Best eval return: {metrics['best_eval_return']:.2f}")
            
        except subprocess.TimeoutExpired:
            print(f"\n⚠️  Timeout for {config_name} (seed={seed})")
            results.append({
                'config': config_name,
                'seed': seed,
                'elapsed_time': 3600,
                'success': False,
                'error': 'Timeout',
            })
        except Exception as e:
            print(f"\n❌ Error for {config_name} (seed={seed}): {e}")
            results.append({
                'config': config_name,
                'seed': seed,
                'elapsed_time': 0,
                'success': False,
                'error': str(e),
            })
    
    return results


def generate_report(all_results, output_path):
    """Generate markdown report from results"""
    
    report = []
    report.append("# CNN Mode Benchmark Report")
    report.append("")
    report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Group results by configuration
    configs = {}
    for result in all_results:
        config = result['config']
        if config not in configs:
            configs[config] = []
        configs[config].append(result)
    
    # Summary table
    report.append("## Summary")
    report.append("")
    report.append("| Configuration | Runs | Success Rate | Avg Best Return | Avg FPS | Avg Time/Update |")
    report.append("|---------------|------|--------------|-----------------|---------|-----------------|")
    
    for config_name in ['vanilla', 'cnn_replace', 'cnn_append']:
        if config_name not in configs:
            continue
        
        results = configs[config_name]
        successful = [r for r in results if r['success']]
        success_rate = len(successful) / len(results) * 100 if results else 0
        
        avg_return = sum(r.get('best_eval_return', 0) for r in successful) / len(successful) if successful else 0
        avg_fps = sum(r.get('training_fps', 0) for r in successful) / len(successful) if successful else 0
        avg_time = sum(r.get('time_per_update', 0) for r in successful) / len(successful) if successful else 0
        
        report.append(f"| {config_name} | {len(results)} | {success_rate:.1f}% | {avg_return:.2f} | {avg_fps:,.0f} | {avg_time:.3f}s |")
    
    report.append("")
    
    # Detailed results
    report.append("## Detailed Results")
    report.append("")
    
    for config_name in ['vanilla', 'cnn_replace', 'cnn_append']:
        if config_name not in configs:
            continue
        
        report.append(f"### {config_name}")
        report.append("")
        
        results = configs[config_name]
        
        for result in results:
            report.append(f"**Seed {result['seed']}**:")
            report.append(f"- Success: {'✅' if result['success'] else '❌'}")
            report.append(f"- Elapsed Time: {result['elapsed_time']:.2f}s")
            
            if result['success']:
                if 'best_eval_return' in result:
                    report.append(f"- Best Eval Return: {result['best_eval_return']:.2f}")
                if 'training_fps' in result:
                    report.append(f"- Training FPS: {result['training_fps']:,.0f}")
                if 'time_per_update' in result:
                    report.append(f"- Time per Update: {result['time_per_update']:.3f}s")
            else:
                report.append(f"- Error: {result.get('error', 'Unknown')}")
            
            report.append("")
    
    # Performance comparison
    report.append("## Performance Comparison")
    report.append("")
    
    vanilla_results = [r for r in configs.get('vanilla', []) if r['success']]
    replace_results = [r for r in configs.get('cnn_replace', []) if r['success']]
    append_results = [r for r in configs.get('cnn_append', []) if r['success']]
    
    if vanilla_results:
        vanilla_avg_return = sum(r.get('best_eval_return', 0) for r in vanilla_results) / len(vanilla_results)
        
        report.append(f"**Baseline (Vanilla Transformer)**: {vanilla_avg_return:.2f} avg return")
        report.append("")
        
        if replace_results:
            replace_avg_return = sum(r.get('best_eval_return', 0) for r in replace_results) / len(replace_results)
            improvement = (replace_avg_return - vanilla_avg_return) / abs(vanilla_avg_return) * 100
            report.append(f"**CNN Replace Mode**: {replace_avg_return:.2f} avg return ({improvement:+.1f}% vs baseline)")
        
        if append_results:
            append_avg_return = sum(r.get('best_eval_return', 0) for r in append_results) / len(append_results)
            improvement = (append_avg_return - vanilla_avg_return) / abs(vanilla_avg_return) * 100
            report.append(f"**CNN Append Mode**: {append_avg_return:.2f} avg return ({improvement:+.1f}% vs baseline)")
        
        report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    report.append("")
    
    if replace_results and append_results and vanilla_results:
        all_returns = {
            'vanilla': sum(r.get('best_eval_return', 0) for r in vanilla_results) / len(vanilla_results),
            'replace': sum(r.get('best_eval_return', 0) for r in replace_results) / len(replace_results),
            'append': sum(r.get('best_eval_return', 0) for r in append_results) / len(append_results),
        }
        
        best_mode = max(all_returns, key=all_returns.get)
        
        report.append(f"Based on the benchmark results, **{best_mode}** mode performed best with an average return of {all_returns[best_mode]:.2f}.")
        report.append("")
        
        if best_mode == 'vanilla':
            report.append("The vanilla Transformer outperformed both CNN modes. Consider:")
            report.append("- The grid representation may already be sufficiently informative")
            report.append("- CNN overhead may not justify the added complexity for this task")
            report.append("- Try different CNN architectures or hyperparameters if you want to explore CNNs further")
        elif best_mode == 'replace':
            report.append("CNN replace mode performed best. This suggests:")
            report.append("- The CNN is effectively extracting spatial features")
            report.append("- The compressed representation is more informative than raw grid")
            report.append("- Consider using CNN replace mode for future training")
        else:  # append
            report.append("CNN append mode performed best. This suggests:")
            report.append("- Both raw and CNN features provide complementary information")
            report.append("- The Transformer benefits from having access to both representations")
            report.append("- Consider using CNN append mode for future training")
    else:
        report.append("Insufficient data to make recommendations. Some runs may have failed.")
    
    report.append("")
    
    # Raw data
    report.append("## Raw Data")
    report.append("")
    report.append("```json")
    report.append(json.dumps(all_results, indent=2))
    report.append("```")
    
    # Write report
    report_text = '\n'.join(report)
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    return report_text


def main():
    parser = argparse.ArgumentParser(description="Benchmark CNN modes vs vanilla Transformer")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="Timesteps per run")
    parser.add_argument("--num-envs", type=int, default=2048, help="Number of parallel environments")
    parser.add_argument("--eval-freq", type=int, default=25, help="Evaluation frequency")
    parser.add_argument("--eval-episodes", type=int, default=128, help="Episodes per evaluation")
    parser.add_argument("--seeds", type=int, nargs='+', default=[42, 43, 44], help="Random seeds to test")
    parser.add_argument("--d-model", type=int, default=64, help="Transformer dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of Transformer layers")
    parser.add_argument("--skip-vanilla", action="store_true", help="Skip vanilla baseline")
    parser.add_argument("--skip-replace", action="store_true", help="Skip CNN replace mode")
    parser.add_argument("--skip-append", action="store_true", help="Skip CNN append mode")
    
    args = parser.parse_args()
    
    print("="*70)
    print("CNN MODE BENCHMARK")
    print("="*70)
    print()
    print(f"Configuration:")
    print(f"  Total timesteps per run: {args.total_timesteps:,}")
    print(f"  Parallel environments: {args.num_envs}")
    print(f"  Evaluation frequency: every {args.eval_freq} updates")
    print(f"  Seeds: {args.seeds}")
    print(f"  Network: d_model={args.d_model}, layers={args.num_layers}")
    print()
    
    # Common arguments for all runs
    common_args = [
        "--total-timesteps", str(args.total_timesteps),
        "--num-envs", str(args.num_envs),
        "--eval-freq", str(args.eval_freq),
        "--eval-episodes", str(args.eval_episodes),
        "--d-model", str(args.d_model),
        "--num-layers", str(args.num_layers),
        "--save-freq", "1000",  # Save less frequently to speed up
    ]
    
    all_results = []
    
    # 1. Vanilla Transformer (no CNN)
    if not args.skip_vanilla:
        print("\n" + "="*70)
        print("1/3: VANILLA TRANSFORMER (NO CNN)")
        print("="*70)
        vanilla_results = run_training(
            "vanilla",
            common_args,
            seeds=args.seeds
        )
        all_results.extend(vanilla_results)
    
    # 2. CNN Replace Mode
    if not args.skip_replace:
        print("\n" + "="*70)
        print("2/3: CNN REPLACE MODE")
        print("="*70)
        replace_results = run_training(
            "cnn_replace",
            common_args + ["--use-cnn", "--cnn-mode", "replace"],
            seeds=args.seeds
        )
        all_results.extend(replace_results)
    
    # 3. CNN Append Mode
    if not args.skip_append:
        print("\n" + "="*70)
        print("3/3: CNN APPEND MODE")
        print("="*70)
        append_results = run_training(
            "cnn_append",
            common_args + ["--use-cnn", "--cnn-mode", "append"],
            seeds=args.seeds
        )
        all_results.extend(append_results)
    
    # Generate report
    print("\n" + "="*70)
    print("GENERATING REPORT")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(f"benchmark_cnn_report_{timestamp}.md")
    json_path = Path(f"benchmark_cnn_results_{timestamp}.json")
    
    report_text = generate_report(all_results, report_path)
    
    # Save JSON results
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Report saved to: {report_path}")
    print(f"✅ JSON results saved to: {json_path}")
    print()
    print("="*70)
    print("REPORT PREVIEW")
    print("="*70)
    print()
    print(report_text)


if __name__ == "__main__":
    main()
