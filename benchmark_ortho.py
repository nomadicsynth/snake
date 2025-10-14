"""
Detailed benchmark of Muon orthogonalization step

This isolates the Newton-Schulz orthogonalization to measure its performance.
"""

import jax
import jax.numpy as jnp
import time
from muon_jax import muon


def benchmark_orthogonalization():
    """Benchmark just the orthogonalization step"""

    # Create a Muon optimizer to access the internal function
    opt = muon(learning_rate=0.02, momentum=0.95, nesterov=True)

    # Get the orthogonalization function
    def newton_schulz_orthogonalize(G, steps=5, eps=1e-7):
        """
        Orthogonalize matrix G using Newton-Schulz iteration.
        """
        orig_shape = G.shape
        if len(orig_shape) == 2:
            # 2D case - add dummy dimension
            G = G[..., None]
        
        # Normalize to make it closer to orthogonal
        a, b, c = G.shape
        # Reshape to 2D for orthogonalization
        G_2d = G.reshape(a, b * c)

        # Initial scaling
        norm_sq = jnp.sum(G_2d * G_2d)
        G_2d = G_2d / jnp.sqrt(norm_sq + eps)

        # Newton-Schulz iterations
        def ns_step(G_mat, _):
            G_T_G = jnp.matmul(G_mat.T, G_mat)
            G_mat = 1.5 * G_mat - 0.5 * jnp.matmul(G_mat, G_T_G)
            return G_mat, None

        G_2d, _ = jax.lax.scan(ns_step, G_2d, None, length=steps)

        # Reshape back
        result = G_2d.reshape(a, b, c)
        if len(orig_shape) == 2:
            result = result[..., 0]
        return result

    # Create test matrices of different sizes (simulating different weight matrices)
    test_shapes = [
        (64, 64),      # Typical transformer weight matrix
        (64, 192),     # Attention projection
        (256, 64),     # Feed-forward layer
        (64, 64, 3),   # 3D tensor (if any)
    ]

    print("Benchmarking Newton-Schulz orthogonalization...")
    print("=" * 60)

    for shape in test_shapes:
        print(f"\nShape: {shape}")

        # Create random matrix
        rng = jax.random.PRNGKey(42)
        if len(shape) == 2:
            G = jax.random.normal(rng, shape)
        else:
            G = jax.random.normal(rng, shape)

        # JIT compile the function
        ortho_fn = jax.jit(lambda x: newton_schulz_orthogonalize(x, steps=5))

        # Warmup
        _ = ortho_fn(G)
        _ = jax.block_until_ready(_)

        # Benchmark
        num_runs = 100
        start_time = time.time()
        for _ in range(num_runs):
            result = ortho_fn(G)
            jax.block_until_ready(result)
        end_time = time.time()

        avg_time = (end_time - start_time) / num_runs
        print(f"  Time per orthogonalization: {avg_time:.6f}s")
        print(f"  Orthogonalizations per second: {1.0/avg_time:.2f}")


if __name__ == "__main__":
    benchmark_orthogonalization()