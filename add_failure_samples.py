#!/usr/bin/env python3
"""
Add failure/suboptimal samples to existing dataset
"""

import argparse
import pickle
import numpy as np
from pretrain_utils import augment_state_action, AUGMENTATIONS


def corrupt_action(correct_action, corruption_rate=0.5):
    """Randomly corrupt action with probability corruption_rate"""
    if np.random.random() < corruption_rate:
        # Return a random different action
        actions = [0, 1, 2, 3]
        actions.remove(correct_action)
        return np.random.choice(actions)
    return correct_action


def main():
    parser = argparse.ArgumentParser(description="Add failure samples to dataset")
    parser.add_argument("--input", type=str, required=True, help="Input dataset file")
    parser.add_argument("--output", type=str, required=True, help="Output dataset file")
    parser.add_argument("--failure-ratio", type=float, default=0.2, help="Ratio of failure samples to add (0.2 = 20%)")
    parser.add_argument("--corruption-rate", type=float, default=0.5, help="Rate of action corruption in failure samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    print("=" * 70)
    print("🎯 ADDING FAILURE SAMPLES TO DATASET")
    print("=" * 70)
    print()
    
    # Load existing dataset
    print(f"Loading {args.input}...")
    with open(args.input, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"  Loaded {len(dataset):,} samples")
    
    # Calculate how many failure samples to add
    num_failures = int(len(dataset) * args.failure_ratio)
    print(f"\nAdding {num_failures:,} failure samples ({args.failure_ratio*100:.0f}% of original)")
    print(f"  Action corruption rate: {args.corruption_rate*100:.0f}%")
    
    # Create failure samples by corrupting random samples
    failure_samples = []
    for _ in range(num_failures):
        # Pick a random sample
        sample = dataset[np.random.randint(len(dataset))]
        
        # Corrupt the action
        corrupted_action = corrupt_action(sample['action'], args.corruption_rate)
        
        failure_samples.append({
            'state': sample['state'].copy(),
            'action': corrupted_action,
        })
    
    # Combine original and failure samples
    combined_dataset = dataset + failure_samples
    
    # Shuffle
    np.random.shuffle(combined_dataset)
    
    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(combined_dataset, f)
    
    # Calculate size
    import os
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    
    print()
    print("=" * 70)
    print("✅ DATASET AUGMENTED SUCCESSFULLY")
    print("=" * 70)
    print(f"Original samples: {len(dataset):,}")
    print(f"Failure samples: {num_failures:,}")
    print(f"Total samples: {len(combined_dataset):,}")
    print(f"File size: {size_mb:.1f} MB")
    print(f"Output: {args.output}")
    print()
    print("Breakdown:")
    print(f"  Expert: {len(dataset)/len(combined_dataset)*100:.1f}%")
    print(f"  Failures: {num_failures/len(combined_dataset)*100:.1f}%")
    print()


if __name__ == "__main__":
    main()
