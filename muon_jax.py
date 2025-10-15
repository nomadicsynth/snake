"""
JAX implementation of the Muon optimizer for Flax/Optax

Muon (Momentum Orthogonalized by Newton-schulz) is a momentum-based optimizer
designed for transformer architectures. This is a JAX/Optax implementation
based on the original PyTorch version.

Reference: https://github.com/KellerJordan/Muon
"""

import jax
import jax.numpy as jnp
from typing import Any, Callable, NamedTuple, Optional, Union
import optax
from optax._src import base


class MuonState(NamedTuple):
    """State for the Muon optimizer."""
    momentum: Any
    count: jnp.ndarray


def muon(
    learning_rate: Union[float, Callable[[int], float]],
    momentum: float = 0.95,
    nesterov: bool = True,
) -> base.GradientTransformation:
    """
    Muon optimizer for 2D+ weight matrices (transformer layers).
    
    Muon applies orthogonalization to momentum updates using Newton-Schulz iteration,
    which helps maintain orthogonality of weight matrices during training.
    
    Args:
        learning_rate: Learning rate (can be a schedule function)
        momentum: Momentum coefficient (default: 0.95)
        nesterov: Whether to use Nesterov momentum (default: True)
    
    Returns:
        An optax GradientTransformation
    """
    
    def newton_schulz_orthogonalize(G, steps=5, eps=1e-7):
        """
        Orthogonalize matrix G using Newton-Schulz iteration.
        
        This iteratively computes G * (3I - G^T G) / 2 to produce an orthogonal matrix.
        Handles 2D (dense), 3D (multi-head), and 4D (conv) tensors.
        """
        orig_shape = G.shape
        
        # Reshape to 2D for orthogonalization
        # For 2D: (a, b) -> (a, b)
        # For 3D: (a, b, c) -> (a, b*c)
        # For 4D: (a, b, c, d) -> (a*b, c*d) or (a, b*c*d) depending on convention
        if G.ndim == 2:
            G_2d = G
        elif G.ndim == 3:
            a, b, c = G.shape
            G_2d = G.reshape(a, b * c)
        elif G.ndim >= 4:
            # For conv kernels (k_h, k_w, in_channels, out_channels)
            # Reshape to (out_channels, k_h * k_w * in_channels)
            G_2d = G.reshape(-1, G.shape[-1]).T
        else:
            return G  # Shouldn't happen, but safe fallback
        
        # Initial scaling - use linalg.norm for better performance
        norm_val = jnp.linalg.norm(G_2d)
        G_2d = G_2d / (norm_val + eps)
        
        # Newton-Schulz iterations
        def ns_step(G_mat, _):
            G_T_G = jnp.matmul(G_mat.T, G_mat)
            G_mat = 1.5 * G_mat - 0.5 * jnp.matmul(G_mat, G_T_G)
            return G_mat, None
        
        G_2d, _ = jax.lax.scan(ns_step, G_2d, None, length=steps)
        
        # Reshape back to original shape
        if G.ndim == 2:
            result = G_2d
        elif G.ndim == 3:
            result = G_2d.reshape(orig_shape)
        elif G.ndim >= 4:
            # Reshape back from (out_channels, k_h * k_w * in_channels)
            result = G_2d.T.reshape(orig_shape)
        
        return result
    
    def init_fn(params):
        momentum = jax.tree_util.tree_map(jnp.zeros_like, params)
        return MuonState(momentum=momentum, count=jnp.zeros([], jnp.int32))
    
    def update_fn(updates, state, params=None):
        del params  # Unused
        
        # Get learning rate
        if callable(learning_rate):
            step_size = learning_rate(state.count)
        else:
            step_size = learning_rate
        
        # Update momentum and apply Muon transformation
        new_updates = {}
        new_momentum = {}
        
        def process_leaf(path, grad, mom):
            # For 2D+ tensors, apply Newton-Schulz orthogonalization
            if grad.ndim >= 2:
                # Update momentum
                mom_new = momentum * mom + grad
                
                # Apply Newton-Schulz orthogonalization for weight matrices
                mom_new = newton_schulz_orthogonalize(mom_new, steps=5)
                
                # Apply Nesterov momentum if requested
                if nesterov:
                    upd = momentum * mom_new + grad
                else:
                    upd = mom_new
                
                # Scale by learning rate
                upd = -step_size * upd
            else:
                # For 1D tensors (biases, norms), just do standard momentum
                mom_new = momentum * mom + grad
                if nesterov:
                    upd = momentum * mom_new + grad
                else:
                    upd = mom_new
                upd = -step_size * upd
            
            return upd, mom_new
        
        # Process all leaves
        flat_updates = jax.tree_util.tree_leaves_with_path(updates)
        flat_momentum = jax.tree_util.tree_leaves(state.momentum)
        
        processed = []
        for (path, grad), mom in zip(flat_updates, flat_momentum):
            upd, mom_new = process_leaf(path, grad, mom)
            processed.append((upd, mom_new))
        
        # Reconstruct trees
        updates_flat = [upd for upd, _ in processed]
        momentum_flat = [mom for _, mom in processed]
        
        new_updates = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(updates), updates_flat
        )
        new_momentum = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(state.momentum), momentum_flat
        )
        
        new_state = MuonState(
            momentum=new_momentum,
            count=state.count + 1
        )
        
        return new_updates, new_state
    
    return base.GradientTransformation(init_fn, update_fn)


def multi_transform_with_muon(
    muon_lr,  # Can be float or schedule (callable)
    aux_lr,   # Can be float or schedule (callable)
    momentum: float = 0.95,
    nesterov: bool = True,
) -> base.GradientTransformation:
    """
    Create a multi-transform optimizer that applies Muon to weight matrices
    and Adam to auxiliary parameters (biases, embeddings, etc.).
    
    Args:
        muon_lr: Learning rate for Muon (weight matrices). Can be a float or a schedule (callable).
        aux_lr: Learning rate for Adam (auxiliary parameters). Can be a float or a schedule (callable).
        momentum: Momentum for Muon
        nesterov: Whether to use Nesterov momentum for Muon
    
    Returns:
        An optax GradientTransformation
    """
    
    def param_labels(params):
        """Label params as 'muon' or 'adam' based on dimensionality."""
        return jax.tree_util.tree_map(
            lambda p: 'muon' if p.ndim >= 2 else 'adam',
            params
        )
    
    muon_opt = muon(muon_lr, momentum=momentum, nesterov=nesterov)
    adam_opt = optax.adam(aux_lr, eps=1e-5)
    
    return optax.multi_transform(
        {'muon': muon_opt, 'adam': adam_opt},
        param_labels
    )


def chain_with_muon(
    muon_lr,  # Can be float or schedule (callable)
    aux_lr,   # Can be float or schedule (callable)
    max_grad_norm: float,
    momentum: float = 0.95,
    nesterov: bool = True,
) -> base.GradientTransformation:
    """
    Create a chained optimizer with gradient clipping and Muon/Adam.
    
    This is the recommended way to use Muon with gradient clipping.
    
    Args:
        muon_lr: Learning rate for Muon (weight matrices). Can be a float or a schedule (callable).
        aux_lr: Learning rate for Adam (auxiliary parameters). Can be a float or a schedule (callable).
        max_grad_norm: Maximum gradient norm for clipping
        momentum: Momentum for Muon
        nesterov: Whether to use Nesterov momentum for Muon
    
    Returns:
        An optax GradientTransformation that clips gradients then applies Muon/Adam
    """
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        multi_transform_with_muon(muon_lr, aux_lr, momentum, nesterov)
    )
