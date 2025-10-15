"""
Generate a mixed dataset for pretraining that includes:
1. Expert A* trajectories (optimal behavior)
2. Near-optimal trajectories (occasionally suboptimal moves)
3. Failure trajectories (leading to death - what NOT to do)

This helps the model learn both good and bad behaviors, making it more robust
to the exploration that happens during RL training.
"""

import argparse
import pickle
import numpy as np
from snake_jax.config import EnvConfig
from snake_jax.gymnax_wrapper import SnakeGymnaxWrapper
from pretrain_utils import astar_to_food, action_from_move, augment_state_action, AUGMENTATIONS
import jax
import jax.numpy as jnp


def augment_samples(samples):
    """Apply all 8 geometric augmentations to samples"""
    augmented = []
    for sample in samples:
        for aug in AUGMENTATIONS:
            aug_state, aug_action = augment_state_action(sample['state'], sample['action'], aug)
            augmented.append({
                'state': aug_state,
                'action': aug_action,
            })
    return augmented


def generate_expert_trajectory(env, max_steps=100):
    """Generate optimal A* trajectory until death or max steps"""
    rng = jax.random.PRNGKey(np.random.randint(0, 1000000))
    rng, reset_rng = jax.random.split(rng)
    obs, state = env.reset(reset_rng)
    
    samples = []
    for _ in range(max_steps):
        # Get current positions
        head_pos = state.head_pos
        snake_positions = []
        curr_pos = head_pos
        for i in range(state.length):
            snake_positions.append((int(curr_pos[0]), int(curr_pos[1])))
            if i < state.length - 1:
                direction = state.body_dirs[i]
                if direction == 0:
                    curr_pos = (curr_pos[0] - 1, curr_pos[1])
                elif direction == 1:
                    curr_pos = (curr_pos[0], curr_pos[1] + 1)
                elif direction == 2:
                    curr_pos = (curr_pos[0] + 1, curr_pos[1])
                else:
                    curr_pos = (curr_pos[0], curr_pos[1] - 1)
        
        food_pos = (int(state.food_pos[0]), int(state.food_pos[1]))
        
        # Get A* action
        try:
            path = astar_to_food(snake_positions, food_pos, env.config.width, env.config.height)
            if path and len(path) > 1:
                action = action_from_move(snake_positions[0], path[1])
            else:
                action = np.random.randint(0, 4)
        except:
            action = np.random.randint(0, 4)
        
        # Store sample
        samples.append({
            'state': np.array(obs),
            'action': int(action),
        })
        
        # Step environment
        rng, step_rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(step_rng, state, action)
        
        if done:
            break
    
    return samples


def generate_epsilon_greedy_trajectory(env, epsilon=0.2, max_steps=100):
    """
    Generate trajectory using epsilon-greedy A* 
    (follows A* most of the time, but sometimes takes random actions)
    """
    rng = jax.random.PRNGKey(np.random.randint(0, 1000000))
    rng, reset_rng = jax.random.split(rng)
    obs, state = env.reset(reset_rng)
    
    samples = []
    for _ in range(max_steps):
        # Get current positions
        head_pos = state.head_pos
        snake_positions = []
        curr_pos = head_pos
        for i in range(state.length):
            snake_positions.append((int(curr_pos[0]), int(curr_pos[1])))
            if i < state.length - 1:
                direction = state.body_dirs[i]
                if direction == 0:
                    curr_pos = (curr_pos[0] - 1, curr_pos[1])
                elif direction == 1:
                    curr_pos = (curr_pos[0], curr_pos[1] + 1)
                elif direction == 2:
                    curr_pos = (curr_pos[0] + 1, curr_pos[1])
                else:
                    curr_pos = (curr_pos[0], curr_pos[1] - 1)
        
        food_pos = (int(state.food_pos[0]), int(state.food_pos[1]))
        
        # Epsilon-greedy: sometimes take random action
        if np.random.random() < epsilon:
            action = np.random.randint(0, 4)
        else:
            # Try A* to get optimal action
            try:
                path = astar_to_food(snake_positions, food_pos, env.config.width, env.config.height)
                if path and len(path) > 1:
                    action = action_from_move(snake_positions[0], path[1])
                else:
                    action = np.random.randint(0, 4)
            except:
                action = np.random.randint(0, 4)
        
        # Store sample
        samples.append({
            'state': np.array(obs),
            'action': int(action),
        })
        
        # Step environment
        rng, step_rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(step_rng, state, action)
        
        if done:
            break
    
    return samples


def generate_random_walk_trajectory(env, max_steps=100):
    """Generate a random trajectory until death or max steps"""
    rng = jax.random.PRNGKey(np.random.randint(0, 1000000))
    rng, reset_rng = jax.random.split(rng)
    obs, state = env.reset(reset_rng)
    
    samples = []
    for _ in range(max_steps):
        # Random action
        rng, action_rng = jax.random.split(rng)
        action = jax.random.randint(action_rng, (), 0, 4)
        
        # Store sample
        samples.append({
            'state': np.array(obs),
            'action': int(action),
        })
        
        # Step environment
        rng, step_rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(step_rng, state, action)
        
        if done:
            break
    
    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate mixed pretraining dataset")
    parser.add_argument("--expert-samples", type=int, default=5000, help="Number of expert A* trajectories")
    parser.add_argument("--suboptimal-samples", type=int, default=3000, help="Number of epsilon-greedy trajectories")
    parser.add_argument("--failure-samples", type=int, default=2000, help="Number of random walk (failure) trajectories")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Epsilon for epsilon-greedy trajectories")
    parser.add_argument("--augment", action="store_true", help="Apply geometric augmentation")
    parser.add_argument("--output", type=str, default="snake_pretrain_dataset_mixed.pkl", help="Output file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    from snake_jax.config import EnvConfig
    env_config = EnvConfig(width=10, height=10)
    
    print("=" * 70)
    print("🎯 MIXED DATASET GENERATION")
    print("=" * 70)
    print()
    print(f"Expert samples (A*): {args.expert_samples}")
    print(f"Suboptimal samples (ε-greedy, ε={args.epsilon}): {args.suboptimal_samples}")
    print(f"Failure samples (random): {args.failure_samples}")
    print(f"Augmentation: {'enabled' if args.augment else 'disabled'}")
    print()
    
    all_samples = []
    
    # Create environment
    env = SnakeEnv(env_config)
    
    # Generate expert trajectories
    print("Generating expert A* trajectories...")
    expert_count = 0
    while expert_count < args.expert_samples:
        trajectory = generate_expert_trajectory(env)
        all_samples.extend(trajectory)
        expert_count += len(trajectory)
        if (expert_count // 100) != ((expert_count - len(trajectory)) // 100):
            print(f"  Progress: {expert_count}/{args.expert_samples}")
    print(f"  Generated {expert_count} expert samples")
    
    # Generate epsilon-greedy trajectories
    print(f"Generating epsilon-greedy trajectories (ε={args.epsilon})...")
    suboptimal_count = 0
    while suboptimal_count < args.suboptimal_samples:
        trajectory = generate_epsilon_greedy_trajectory(env, epsilon=args.epsilon)
        all_samples.extend(trajectory)
        suboptimal_count += len(trajectory)
        if (suboptimal_count // 100) != ((suboptimal_count - len(trajectory)) // 100):
            print(f"  Progress: {suboptimal_count}/{args.suboptimal_samples}")
    print(f"  Generated {suboptimal_count} suboptimal samples")
    
    # Generate failure trajectories
    print("Generating random walk (failure) trajectories...")
    failure_count = 0
    while failure_count < args.failure_samples:
        trajectory = generate_random_walk_trajectory(env)
        all_samples.extend(trajectory)
        failure_count += len(trajectory)
        if (failure_count // 100) != ((failure_count - len(trajectory)) // 100):
            print(f"  Progress: {failure_count}/{args.failure_samples}")
    print(f"  Generated {failure_count} failure samples")
    
    # Apply augmentation if requested
    if args.augment:
        print("\nApplying geometric augmentation...")
        original_count = len(all_samples)
        all_samples = augment_samples(all_samples)
        print(f"  {original_count} → {len(all_samples)} samples (8x augmentation)")
    
    # Save dataset
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(all_samples, f)
    
    # Calculate size
    import os
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    
    print()
    print("=" * 70)
    print("✅ DATASET GENERATED SUCCESSFULLY")
    print("=" * 70)
    print(f"Total samples: {len(all_samples):,}")
    print(f"File size: {size_mb:.1f} MB")
    print(f"Output: {args.output}")
    print()
    print("Breakdown:")
    expert_pct = (expert_count / len(all_samples)) * 100 if not args.augment else ((expert_count * 8) / len(all_samples)) * 100
    print(f"  Expert: ~{expert_pct:.1f}%")
    print(f"  Suboptimal: ~{((suboptimal_count * (8 if args.augment else 1)) / len(all_samples)) * 100:.1f}%")
    print(f"  Failures: ~{((failure_count * (8 if args.augment else 1)) / len(all_samples)) * 100:.1f}%")
    print()


if __name__ == "__main__":
    main()
