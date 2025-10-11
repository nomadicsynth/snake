#!/usr/bin/env python3
"""Quick test of the save functionality."""

from sb3_snake_batch_size_benchmark import save_results_to_file, save_best_config_to_env

# Mock results
all_results = {
    'baseline_fp32': {
        128: {'samples_per_sec': 100.0, 'updates_per_sec': 10.0, 'max_memory_gb': 2.5},
        256: {'samples_per_sec': 150.0, 'updates_per_sec': 8.0, 'max_memory_gb': 3.0},
    },
    'bf16_compile_flash': {
        128: {'samples_per_sec': 300.0, 'updates_per_sec': 25.0, 'max_memory_gb': 2.8},
        256: {'samples_per_sec': 450.0, 'updates_per_sec': 20.0, 'max_memory_gb': 3.2},
    }
}

print("Testing save_results_to_file...")
filename = save_results_to_file(all_results, fast_mode=True)
print(f"✓ Saved to {filename}")

print("\nTesting save_best_config_to_env...")
save_best_config_to_env(all_results)
print("✓ Saved to .env")

print("\n.env contents:")
with open('.env', 'r') as f:
    print(f.read())
