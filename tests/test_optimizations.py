#!/usr/bin/env python3
"""
Test script to verify that optimization settings from .env are loaded and applied correctly.
"""
import os
from dotenv import load_dotenv

# Load .env
load_dotenv()

print("="*60)
print("Environment Variable Configuration Test")
print("="*60)

# Test reading environment variables
batch_size = os.getenv('OPTIMAL_BATCH_SIZE')
use_compile = os.getenv('USE_TORCH_COMPILE')
use_flash = os.getenv('USE_FLASH_ATTENTION')
use_bf16 = os.getenv('USE_BF16')

print(f"\nRaw values from .env:")
print(f"  OPTIMAL_BATCH_SIZE: {batch_size}")
print(f"  USE_TORCH_COMPILE: {use_compile}")
print(f"  USE_FLASH_ATTENTION: {use_flash}")
print(f"  USE_BF16: {use_bf16}")

# Test helper functions
from sb3_snake import get_env_int, get_env_bool, ENV_BATCH_SIZE, ENV_USE_TORCH_COMPILE, ENV_USE_FLASH_ATTENTION, ENV_USE_BF16

print(f"\nParsed values (used by sb3_snake.py):")
print(f"  ENV_BATCH_SIZE: {ENV_BATCH_SIZE}")
print(f"  ENV_USE_TORCH_COMPILE: {ENV_USE_TORCH_COMPILE}")
print(f"  ENV_USE_FLASH_ATTENTION: {ENV_USE_FLASH_ATTENTION}")
print(f"  ENV_USE_BF16: {ENV_USE_BF16}")

# Test feature availability
import torch
print(f"\n{'='*60}")
print("PyTorch Feature Availability:")
print("="*60)
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"  BF16 supported: {torch.cuda.is_bf16_supported()}")
print(f"  torch.compile available: {hasattr(torch, 'compile')}")
print(f"  SDPA available: {hasattr(torch.nn.functional, 'scaled_dot_product_attention')}")
print(f"  SDPA kernel control available: {hasattr(torch.nn.attention, 'sdpa_kernel')}")

# Recommendation
print(f"\n{'='*60}")
print("Configuration Summary:")
print("="*60)
print(f"Based on your .env file, training will use:")
print(f"  ✓ Batch size: {ENV_BATCH_SIZE}")
if ENV_USE_TORCH_COMPILE:
    if hasattr(torch, 'compile'):
        print(f"  ✓ torch.compile: ENABLED and AVAILABLE")
    else:
        print(f"  ⚠ torch.compile: ENABLED but NOT AVAILABLE (requires PyTorch 2.0+)")
else:
    print(f"  ✗ torch.compile: DISABLED")

if ENV_USE_FLASH_ATTENTION:
    if hasattr(torch.nn.attention, 'sdpa_kernel'):
        print(f"  ✓ FlashAttention: ENABLED and AVAILABLE")
    else:
        print(f"  ⚠ FlashAttention: ENABLED but NOT AVAILABLE")
else:
    print(f"  ✗ FlashAttention: DISABLED")

if ENV_USE_BF16:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        print(f"  ✓ BF16 mixed precision: ENABLED and AVAILABLE")
    else:
        print(f"  ⚠ BF16 mixed precision: ENABLED but NOT AVAILABLE")
else:
    print(f"  ✗ BF16 mixed precision: DISABLED")

print(f"\nTo override these settings, use command-line flags:")
print(f"  --no-compile         : Disable torch.compile")
print(f"  --no-flash-attention : Disable FlashAttention")
print(f"  --no-bf16           : Disable BF16 mixed precision")
print("="*60)
