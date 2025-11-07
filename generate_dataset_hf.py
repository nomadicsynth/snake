#!/usr/bin/env python3
"""
Generate Snake pretraining dataset in HuggingFace format
"""

import argparse
from pathlib import Path
import random
import tempfile

from datasets import Dataset, DatasetDict
from datasets import Array3D, Features, Sequence, Value
import numpy as np
import torch

from pretrain_dataset import generate_pretraining_dataset


def _cast_sample(sample, has_reasoning: bool):
    """Cast fields to stable dtypes for storage and IO."""
    state = np.asarray(sample['state'], dtype='float32')
    action = int(np.asarray(sample['action'], dtype='int8'))
    out = {
        'state': state,
        'action': action,
    }
    if has_reasoning and 'reasoning_tokens' in sample:
        out['reasoning_tokens'] = np.asarray(sample['reasoning_tokens'], dtype='int32')
    return out


def yield_samples_in_chunks(total_samples: int, batch_size: int, gen_kwargs: dict, *,
                            has_reasoning: bool):
    """Yield up to total_samples items by repeatedly calling generate_pretraining_dataset.

    Only at most one batch is kept in RAM at a time.
    """
    num_yielded = 0
    while num_yielded < total_samples:
        current_batch = min(batch_size, total_samples - num_yielded)
        batch = generate_pretraining_dataset(num_samples=current_batch, **gen_kwargs)
        for sample in batch:
            yield _cast_sample(sample, has_reasoning)
            num_yielded += 1
            if num_yielded >= total_samples:
                break


def main():
    parser = argparse.ArgumentParser(
        description="Generate Snake dataset in HuggingFace format"
    )
    
    parser.add_argument('--num-samples', type=int, default=50000, help='Number of unique states to generate (before augmentation)')
    parser.add_argument('--width', type=int, default=20, help='Grid width')
    parser.add_argument('--height', type=int, default=20, help='Grid height')
    parser.add_argument('--min-length', type=int, default=3, help='Minimum snake length')
    parser.add_argument('--max-length', type=int, default=30, help='Maximum snake length')
    parser.add_argument('--use-astar', action='store_true', default=True, help='Use A* for expert labels (default: True)')
    parser.add_argument('--no-astar', dest='use_astar', action='store_false', help='Use heuristics instead of A*')
    parser.add_argument('--temperature', type=float, default=0.5, help='Temperature for soft labels')
    parser.add_argument('--augment', action='store_true', default=True, help='Apply 8x geometric augmentation (default: True)')
    parser.add_argument('--no-augment', dest='augment', action='store_false', help='Skip augmentation')
    parser.add_argument('--failure-ratio', type=float, default=0.0, help='Ratio of random/failure action samples to add')
    parser.add_argument('--epsilon-greedy-ratio', type=float, default=0.0, help='Ratio of epsilon-greedy samples to add')
    parser.add_argument('--epsilon', type=float, default=0.3, help='Epsilon value for epsilon-greedy sampling')
    parser.add_argument('--output', type=str, default='snake_dataset_hf', help='Output directory path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split fraction')
    parser.add_argument('--batch-size', type=int, default=10000, help='Batch size for streamed Arrow writing')
    
    # Reasoning Snake Model (RSM) parameters
    parser.add_argument('--reasoning', action='store_true', help='Add CoT-style reasoning before actions (RSM mode)')
    parser.add_argument('--reasoning-depth', type=int, default=1, choices=[1, 2, 3], help='Lookahead depth for reasoning (1-3 steps)')
    parser.add_argument('--reasoning-format', type=str, default='compact', choices=['compact', 'verbose'], help='Reasoning text format')

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
    
    # Seed global RNGs once; do not reseed per batch to avoid duplicates
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Prepare generation kwargs shared across batches
    gen_kwargs = dict(
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
        add_reasoning=args.reasoning,
        reasoning_depth=args.reasoning_depth,
        reasoning_format=args.reasoning_format,
    )

    has_reasoning = args.reasoning

    # Determine split sizes without ever holding all samples in RAM
    n_val = int(args.num_samples * args.val_split)
    n_train = args.num_samples - n_val

    print("Generating and writing datasets in a streamed manner...")
    print(f"  Training:   {n_train:,} samples")
    print(f"  Validation: {n_val:,} samples")

    # Build datasets from generators with small writer batches to keep RAM low
    def train_gen():
        yield from yield_samples_in_chunks(
            n_train, args.batch_size, gen_kwargs,
            has_reasoning=has_reasoning,
        )

    def val_gen():
        yield from yield_samples_in_chunks(
            n_val, args.batch_size, gen_kwargs,
            has_reasoning=has_reasoning,
        )

    # Build explicit HF Features so dtype/shape round-trip reliably
    print("\nConverting to HuggingFace format (streamed)...")
    state_feature = Array3D(dtype='float32', shape=(args.height, args.width, 3))
    features_dict = {
        'state': state_feature,
        'action': Value('int8'),
    }
    if has_reasoning:
        features_dict['reasoning_tokens'] = Sequence(Value('int32'))
    features = Features(features_dict)

    train_dataset = Dataset.from_generator(
        train_gen,
        writer_batch_size=args.batch_size,
        features=features,
        cache_dir=tempfile.mkdtemp()
    )
    val_dataset = Dataset.from_generator(
        val_gen,
        writer_batch_size=args.batch_size,
        features=features,
        cache_dir=tempfile.mkdtemp()
    )

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
    })

    # Save
    output_path = Path(args.output)
    print(f"\nSaving to {output_path}...")
    dataset_dict.save_to_disk(str(output_path))

    print("Cleaning up temporary cache directory...")
    train_dataset.cleanup_cache_files()
    val_dataset.cleanup_cache_files()

    print(f"\n✅ Dataset saved!")
    print(f"   Path: {output_path}")
    print(f"   Format: HuggingFace Dataset (streamed)")
    print()

    # Print sample info
    sample = train_dataset[0]
    state_shape = np.array(sample['state']).shape
    print("Sample structure:")
    print(f"  state: {state_shape} (float32)")
    print(f"  action: scalar (int8)")
    if has_reasoning:
        reasoning_shape = np.array(sample['reasoning_tokens']).shape
        print(f"  reasoning_tokens: {reasoning_shape} (int32)")
    print()

    print("Dataset info:")
    print(dataset_dict)
    print()


if __name__ == "__main__":
    main()
