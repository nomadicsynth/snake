#!/usr/bin/env python3
"""Test with real benchmark results."""

from sb3_snake_batch_size_benchmark import save_best_config_to_env

# Your actual results
all_results = {
    'baseline_fp32': {
        128: {'samples_per_sec': 5082.8, 'updates_per_sec': 39.7, 'max_memory_gb': 2.5},
        256: {'samples_per_sec': 6080.6, 'updates_per_sec': 23.8, 'max_memory_gb': 3.0},
        512: {'samples_per_sec': 6239.2, 'updates_per_sec': 12.2, 'max_memory_gb': 3.5},
        1024: {'samples_per_sec': 6514.9, 'updates_per_sec': 6.4, 'max_memory_gb': 4.0},
        2048: {'samples_per_sec': 6695.0, 'updates_per_sec': 3.3, 'max_memory_gb': 5.0},
        4096: {'samples_per_sec': 6806.1, 'updates_per_sec': 1.7, 'max_memory_gb': 7.0},
    },
    'bf16_flash': {
        128: {'samples_per_sec': 9080.2, 'updates_per_sec': 70.9, 'max_memory_gb': 2.3},
        256: {'samples_per_sec': 16590.4, 'updates_per_sec': 64.8, 'max_memory_gb': 2.8},
        512: {'samples_per_sec': 20353.4, 'updates_per_sec': 39.8, 'max_memory_gb': 3.3},
        1024: {'samples_per_sec': 22746.1, 'updates_per_sec': 22.2, 'max_memory_gb': 4.2},
        2048: {'samples_per_sec': 24805.6, 'updates_per_sec': 12.1, 'max_memory_gb': 5.8},
        4096: {'samples_per_sec': 25519.1, 'updates_per_sec': 6.2, 'max_memory_gb': 9.0},
    },
    'bf16_compile': {
        128: {'samples_per_sec': 6271.9, 'updates_per_sec': 49.0, 'max_memory_gb': 2.6},
        256: {'samples_per_sec': 10918.4, 'updates_per_sec': 42.7, 'max_memory_gb': 3.1},
        512: {'samples_per_sec': 18690.2, 'updates_per_sec': 36.5, 'max_memory_gb': 3.7},
        1024: {'samples_per_sec': 22265.8, 'updates_per_sec': 21.7, 'max_memory_gb': 4.5},
        2048: {'samples_per_sec': 24621.4, 'updates_per_sec': 12.0, 'max_memory_gb': 6.2},
        4096: {'samples_per_sec': 25759.2, 'updates_per_sec': 6.3, 'max_memory_gb': 9.5},
    },
    'bf16_compile_flash': {
        128: {'samples_per_sec': 9454.8, 'updates_per_sec': 73.9, 'max_memory_gb': 2.4},
        256: {'samples_per_sec': 14487.6, 'updates_per_sec': 56.6, 'max_memory_gb': 2.9},
        512: {'samples_per_sec': 18853.5, 'updates_per_sec': 36.8, 'max_memory_gb': 3.5},
        1024: {'samples_per_sec': 22236.9, 'updates_per_sec': 21.7, 'max_memory_gb': 4.4},
        2048: {'samples_per_sec': 24625.2, 'updates_per_sec': 12.0, 'max_memory_gb': 6.1},
        4096: {'samples_per_sec': 25803.0, 'updates_per_sec': 6.3, 'max_memory_gb': 9.3},
    }
}

print("Testing with real benchmark results...")
save_best_config_to_env(all_results)

print("\n✅ Should select: bf16_compile_flash @ batch_size=4096 with 25803.0 samples/sec")

print("\n.env contents:")
with open('.env', 'r') as f:
    print(f.read())
