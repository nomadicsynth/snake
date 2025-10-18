#!/bin/bash
# Install flash-attn with required build flags
# This is kept separate from requirements.txt because it needs --no-build-isolation

set -e

echo "Installing flash-attn..."
echo "Note: This may take 5-10 minutes to compile on first install"
echo ""

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "Warning: nvcc (CUDA compiler) not found in PATH"
    echo "FlashAttention requires CUDA toolkit to be installed"
    echo "Attempting installation anyway..."
    echo ""
fi

# Install with --no-build-isolation flag
pip install flash-attn --no-build-isolation --use-pep517

echo ""
echo "✓ flash-attn installed successfully!"
echo ""
echo "PyTorch 2.0+ will automatically use FlashAttention when available."
