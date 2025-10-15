#!/usr/bin/env python3
"""
Generate pretraining dataset for Snake.
Run this first before pretrain_snake.py
"""

import argparse
from pathlib import Path
from pretrain_dataset import (
    generate_pretraining_dataset,
    save_dataset,
    analyze_dataset
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Snake pretraining dataset"
    )
    
    parser.add_argument('--num-samples', type=int, default=50000,
                       help='Number of unique states to generate (before augmentation)')
    parser.add_argument('--width', type=int, default=20,
                       help='Grid width')
    parser.add_argument('--height', type=int, default=20,
                       help='Grid height')
    parser.add_argument('--min-length', type=int, default=3,
                       help='Minimum snake length')
    parser.add_argument('--max-length', type=int, default=30,
                       help='Maximum snake length')
    parser.add_argument('--use-astar', action='store_true', default=True,
                       help='Use A* for expert labels (default: True)')
    parser.add_argument('--no-astar', dest='use_astar', action='store_false',
                       help='Use heuristics instead of A*')
    parser.add_argument('--temperature', type=float, default=0.5,
                       help='Temperature for soft labels')
    parser.add_argument('--augment', action='store_true', default=True,
                       help='Apply 8x geometric augmentation (default: True)')
    parser.add_argument('--no-augment', dest='augment', action='store_false',
                       help='Skip augmentation')
    parser.add_argument('--failure-ratio', type=float, default=0.0,
                       help='Ratio of random/failure action samples to add')
    parser.add_argument('--epsilon-greedy-ratio', type=float, default=0.0,
                       help='Ratio of epsilon-greedy samples to add')
    parser.add_argument('--epsilon', type=float, default=0.3,
                       help='Epsilon value for epsilon-greedy sampling')
    parser.add_argument('--output', type=str, default='snake_pretrain_dataset.pkl',
                       help='Output file path')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SNAKE PRETRAINING DATASET GENERATION")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Samples: {args.num_samples:,}")
    print(f"  Grid: {args.height}x{args.width}")
    print(f"  Snake length: {args.min_length}-{args.max_length}")
    print(f"  Expert strategy: {'A*' if args.use_astar else 'Heuristic'}")
    print(f"  Augmentation: {'Yes (8x)' if args.augment else 'No'}")
    print(f"  Failure ratio: {args.failure_ratio:.2%}")
    print(f"  Epsilon-greedy ratio: {args.epsilon_greedy_ratio:.2%} (ε={args.epsilon})")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {args.output}")
    
    if args.augment:
        base_size = args.num_samples * 8
        failure_size = int(base_size * args.failure_ratio)
        epsilon_size = int(base_size * args.epsilon_greedy_ratio)
        final_size = base_size + failure_size + epsilon_size
        print(f"\n  Expected final size: {final_size:,} samples")
        print(f"    Expert: {base_size:,}")
        print(f"    Failure: {failure_size:,}")
        print(f"    Epsilon-greedy: {epsilon_size:,}")
    
    print("\n" + "=" * 60)
    
    # Generate dataset
    dataset = generate_pretraining_dataset(
        num_samples=args.num_samples,
        width=args.width,
        height=args.height,
        min_length=args.min_length,
        max_length=args.max_length,
        use_astar=args.use_astar,
        temperature=args.temperature,
        augment=args.augment,
        seed=args.seed,
        failure_ratio=args.failure_ratio,
        epsilon_greedy_ratio=args.epsilon_greedy_ratio,
        epsilon=args.epsilon
    )
    
    # Analyze
    analyze_dataset(dataset)
    
    # Save
    save_dataset(dataset, args.output)
    
    print("\n" + "=" * 60)
    print("DONE!")
    print(f"Dataset saved to: {args.output}")
    print(f"Total samples: {len(dataset):,}")
    
    # Estimate size
    import os
    if os.path.exists(args.output):
        size_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"File size: {size_mb:.1f} MB")
    
    print("\nNext step:")
    print(f"  python pretrain_snake.py --dataset {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
