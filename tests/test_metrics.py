#!/usr/bin/env python3
"""
Test script to verify that training logs include the required eval metrics.
Runs a short training session and checks TensorBoard logs for presence of:
- Looping metrics: eval/loop_is_loop, eval/loop_repeats, eval/loop_period, eval/loop_penalty, eval/loop_end_bonus
- Termination metrics: eval/terminated, eval/truncated, eval/ate
- Score metrics: eval/score, eval/episode_reward, eval/episode_length, eval/episode_time
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Import the training function
sys.path.insert(0, os.path.dirname(__file__))
from sb3_snake import train_sb3

def run_short_training(log_dir):
    """Run a very short training to generate logs."""
    print(f"Running short training with log dir: {log_dir}")

    try:
        train_sb3(
            width=8,
            height=8,
            num_episodes=1,  # Very short
            batch_size=32,
            gamma=0.99,
            eps_start=1.0,
            eps_end=0.1,
            target_update=100,
            d_model=64,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
            lr=1e-3,
            lr_schedule="constant",
            lr_end=1e-4,
            replay_size=1000,
            log_interval=1,
            max_steps=50,  # Short episodes
            model_path=os.path.join(log_dir, "test_model.zip"),
            seed=42,
            eval_episodes=1,
            eval_max_steps=50,
            exploration_fraction=0.1,
            learning_starts=10,
            record_dir=None,
            record_episodes=0,
            eval_record_dir=None,
            eval_record_episodes=0,
            step_penalty=0.0,
            shaping_coef=0.0,
            apple_reward=10.0,
            death_penalty=-10.0,
            max_score=None,
            curriculum_path=None,
            loop_penalty_coef=0.02,  # Enable loop shaping
            loop_end_bonus=0.1,
            loop_min_period=4,
            loop_max_period=20,
            tensorboard_log=log_dir,  # Use custom log dir
        )
    except Exception as e:
        print(f"Training failed: {e}")
        return None

    return log_dir

def find_event_file(log_dir):
    """Find the TensorBoard event file in the log directory."""
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                return os.path.join(root, file)
    return None

def check_metrics(log_dir):
    """Check if required metrics are present in the logs."""
    event_file = find_event_file(log_dir)
    if not event_file:
        print("No TensorBoard event file found!")
        return False

    print(f"Checking metrics in: {event_file}")

    # Load events
    ea = EventAccumulator(event_file)
    ea.Reload()

    # Required eval metrics
    required_scalars = [
        "eval/mean_ep_length",
        "eval/mean_reward",
        "eval/loop_is_loop",
        "eval/loop_repeats",
        "eval/loop_period",
        "eval/loop_penalty",
        "eval/loop_end_bonus",
        "eval/terminated",
        "eval/truncated",
        "eval/ate",
        "eval/score",
        "eval/episode_reward",
        "eval/episode_length",
        "eval/episode_time",
    ]

    missing = []
    for scalar in required_scalars:
        if scalar not in ea.Tags()['scalars']:
            missing.append(scalar)
        else:
            print(f"✓ Found metric: {scalar}")

    if missing:
        print(f"✗ Missing metrics: {missing}")
        return False
    else:
        print("All required eval metrics are present!")
        return True

def main():
    # Use a temporary directory for logs
    temp_log_dir = tempfile.mkdtemp(prefix="test_tb_")
    try:
        log_dir = run_short_training(temp_log_dir)
        if not log_dir:
            return 1

        success = check_metrics(log_dir)
        return 0 if success else 1
    finally:
        shutil.rmtree(temp_log_dir, ignore_errors=True)

if __name__ == "__main__":
    sys.exit(main())