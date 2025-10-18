# 04 - Ablations

Run ablation studies to understand the impact of gradient schedules, sampling steps, latent dimensions, and architecture choices on EqM performance.

## Required Scripts

- `ablation_gradient_schedules.sh` - Compare linear, truncated, piecewise schedules
- `ablation_sampling_steps.sh` - Test 5, 10, 20, 50 steps and adaptive compute
- `ablation_latent_dims.sh` - Compare 32, 64, 128, 256 dimensional latents
- `ablation_architectures.sh` - Test CNN modes and encoder/decoder depths
- `run_all.sh` - Execute all ablation experiments
- `compare_ablations.py` - Generate comparison tables and plots
