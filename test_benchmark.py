import time
import torch
from sb3_snake import SnakeEnv, TransformerExtractor
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit

# Quick test with minimal timesteps
batch_size = 128
timesteps = 200  # Very short test

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

print("Creating model...")
model = DQN(
    "CnnPolicy",
    env,
    policy_kwargs=policy_kwargs,
    batch_size=batch_size,
    buffer_size=10000,
    learning_starts=50,
    train_freq=4,
    gradient_steps=1,
    verbose=0,
    device="cuda",
)

print("\nTest 1: FP32 baseline")
initial_updates = model._n_updates
initial_timesteps = model.num_timesteps
model.learn(total_timesteps=timesteps, progress_bar=True)
updates = model._n_updates - initial_updates
actual_timesteps = model.num_timesteps - initial_timesteps
samples = updates * batch_size
print(f"✓ FP32: {actual_timesteps} env steps, {updates} updates, {samples} samples processed")

print("\nTest 2: BF16 with autocast")
# Reset to allow more training
model.num_timesteps = 0  
initial_updates = model._n_updates
initial_timesteps = model.num_timesteps
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    model.learn(total_timesteps=timesteps, progress_bar=True)
updates = model._n_updates - initial_updates
actual_timesteps = model.num_timesteps - initial_timesteps
samples = updates * batch_size
print(f"✓ BF16: {actual_timesteps} env steps, {updates} updates, {samples} samples processed")

print("\nTest 3: FlashAttention context")
from torch.nn.attention import SDPBackend, sdpa_kernel
model.num_timesteps = 0  # Reset again
initial_updates = model._n_updates
initial_timesteps = model.num_timesteps
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        model.learn(total_timesteps=timesteps, progress_bar=True)
updates = model._n_updates - initial_updates
actual_timesteps = model.num_timesteps - initial_timesteps
samples = updates * batch_size
print(f"✓ Flash: {actual_timesteps} env steps, {updates} updates, {samples} samples processed")

print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
