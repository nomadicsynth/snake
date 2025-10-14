#!/bin/bash
# Complete feature showcase for the enhanced JAX PPO training

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║         🎯 ENHANCED JAX PPO TRAINING - FEATURE SHOWCASE           ║"
echo "╔════════════════════════════════════════════════════════════════════╗"
echo ""
echo "This demonstrates all the new features added from SB3:"
echo ""
echo "  ✅ Beautiful tqdm progress bar with modern styling"
echo "  ✅ Real-time metrics updates (episode returns, speed)"
echo "  ✅ Full WandB integration with comprehensive logging"
echo "  ✅ Comprehensive CLI arguments (like SB3)"
echo "  ✅ Automatic model saving and organization"
echo "  ✅ Rich console output with emojis and formatting"
echo "  ✅ Performance tracking (FPS, timing, throughput)"
echo "  ✅ WandB artifact uploading"
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo ""

PYTHON=".venv/bin/python"

# Show help
echo "📖 Available command-line arguments:"
echo ""
$PYTHON train_snake_purejaxrl_progressive.py --help | head -40
echo "    ... (see --help for full list)"
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Run training demo
echo "🚀 Running training demo with all features enabled..."
echo ""
echo "Configuration:"
echo "  • 100K total timesteps (quick demo)"
echo "  • 512 parallel environments"
echo "  • WandB logging (offline mode)"
echo "  • Real-time progress bar"
echo "  • Custom run name"
echo ""
echo "Press Ctrl+C to cancel, or Enter to start..."
read -r

WANDB_MODE=offline $PYTHON train_snake_purejaxrl_progressive.py \
    --total-timesteps 100000 \
    --num-envs 512 \
    --num-steps 64 \
    --wandb \
    --run-name feature_showcase_demo \
    --d-model 64 \
    --num-layers 2

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "✅ Demo complete!"
echo ""
echo "📊 Check the results:"
echo "  • Model saved in: models/feature_showcase_demo/"
echo "  • WandB logs in: wandb/ (offline mode)"
echo ""
echo "🚀 For a full training run with online WandB:"
echo "  $PYTHON train_snake_purejaxrl_progressive.py \\"
echo "    --wandb \\"
echo "    --total-timesteps 5000000 \\"
echo "    --wandb-project snake-jax-ppo \\"
echo "    --run-name my_experiment"
echo ""
echo "📖 See ENHANCED_TRAINING.md for complete documentation"
echo ""
