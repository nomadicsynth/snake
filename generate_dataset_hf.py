#!/usr/bin/env python3
"""
Generate Snake pretraining dataset in HuggingFace format
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from pretrain_dataset import generate_pretraining_dataset
from pretrain_utils import state_from_positions


def convert_to_hf_dataset(samples, has_reasoning=False):
    """
    Convert list of samples to HuggingFace Dataset
    
    Args:
        samples: List of dicts with 'state', 'action', optionally 'reasoning_tokens'
        has_reasoning: Whether dataset has reasoning tokens (RSM mode)
    
    Returns:
        HuggingFace Dataset
    """
    # Extract arrays
    states = np.array([s['state'] for s in samples], dtype=np.float32)
    actions = np.array([s['action'] for s in samples], dtype=np.int64)
    
    data_dict = {
        'state': states,
        'action': actions,
    }
    
    if has_reasoning:
        reasoning_tokens = np.array([s['reasoning_tokens'] for s in samples], dtype=np.int64)
        data_dict['reasoning_tokens'] = reasoning_tokens
    
    return Dataset.from_dict(data_dict)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Snake dataset in HuggingFace format"
    )
    
    parser.add_argument('--num-samples', type=int, default=50000, 
                        help='Number of unique states to generate (before augmentation)')
    parser.add_argument('--width', type=int, default=20, help='Grid width')
    parser.add_argument('--height', type=int, default=20, help='Grid height')
    parser.add_argument('--min-length', type=int, default=3, help='Minimum snake length')
    parser.add_argument('--max-length', type=int, default=30, help='Maximum snake length')
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
    parser.add_argument('--output', type=str, default='snake_dataset_hf', 
                        help='Output directory path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--val-split', type=float, default=0.1, 
                        help='Validation split fraction')
    
    # Reasoning Snake Model (RSM) parameters
    parser.add_argument('--reasoning', action='store_true', 
                        help='Add CoT-style reasoning before actions (RSM mode)')
    parser.add_argument('--reasoning-depth', type=int, default=1, choices=[1, 2, 3], 
                        help='Lookahead depth for reasoning (1-3 steps)')
    parser.add_argument('--reasoning-format', type=str, default='compact', 
                        choices=['compact', 'verbose'], 
                        help='Reasoning text format')

    args = parser.parse_args()
    
    print("=" * 70)
    print("SNAKE DATASET GENERATION (HuggingFace format)")
    if args.reasoning:
        print("🧠 REASONING SNAKE MODEL (RSM) MODE ENABLED")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Samples: {args.num_samples:,}")
    print(f"  Grid: {args.width}x{args.height}")
    print(f"  Snake length: {args.min_length}-{args.max_length}")
    print(f"  Expert: {'A*' if args.use_astar else 'Heuristic'}")
    print(f"  Augmentation: {args.augment}")
    if args.failure_ratio > 0:
        print(f"  Failure samples: {args.failure_ratio:.1%}")
    if args.epsilon_greedy_ratio > 0:
        print(f"  Epsilon-greedy samples: {args.epsilon_greedy_ratio:.1%} (ε={args.epsilon})")
    if args.reasoning:
        print(f"  Reasoning depth: {args.reasoning_depth}")
        print(f"  Reasoning format: {args.reasoning_format}")
    print()
    
    # Generate dataset
    print("Generating dataset...")
    samples = generate_pretraining_dataset(
        num_samples=args.num_samples,
        width=args.width,
        height=args.height,
        min_length=args.min_length,
        max_length=args.max_length,
        use_astar=args.use_astar,
        temperature=args.temperature,
        augment=args.augment,
        failure_ratio=args.failure_ratio,
        epsilon_greedy_ratio=args.epsilon_greedy_ratio,
        epsilon=args.epsilon,
        seed=args.seed,
        add_reasoning=args.reasoning,
        reasoning_depth=args.reasoning_depth,
        reasoning_format=args.reasoning_format,
    )
    
    print(f"\n✅ Generated {len(samples)} samples")
    
    has_reasoning = args.reasoning
    
    # Split into train/val
    n_val = int(len(samples) * args.val_split)
    n_train = len(samples) - n_val
    
    train_samples = samples[:n_train]
    val_samples = samples[n_train:]
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_samples):,} samples")
    print(f"  Validation: {len(val_samples):,} samples")
    
    # Convert to HuggingFace datasets
    print("\nConverting to HuggingFace format...")
    train_dataset = convert_to_hf_dataset(train_samples, has_reasoning)
    val_dataset = convert_to_hf_dataset(val_samples, has_reasoning)
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
    })
    
    # Save
    output_path = Path(args.output)
    print(f"\nSaving to {output_path}...")
    dataset_dict.save_to_disk(str(output_path))
    
    print(f"\n✅ Dataset saved!")
    print(f"   Path: {output_path}")
    print(f"   Format: HuggingFace Dataset")
    print()
    
    # Print sample info
    sample = train_dataset[0]
    state_shape = np.array(sample['state']).shape
    print("Sample structure:")
    print(f"  state: {state_shape} (float32)")
    print(f"  action: scalar (int64)")
    if has_reasoning:
        reasoning_shape = np.array(sample['reasoning_tokens']).shape
        print(f"  reasoning_tokens: {reasoning_shape} (int64)")
    print()
    
    print("Dataset info:")
    print(dataset_dict)
    print()


if __name__ == "__main__":
    main()
