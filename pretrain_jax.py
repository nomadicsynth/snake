"""
JAX Supervised Pretraining for Snake Transformer

Train the JAX TransformerPolicy with supervised learning on expert trajectories.
This proves the architecture can learn before trying RL.
"""

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import pickle
import argparse
from pathlib import Path
from datetime import datetime
import time
from tqdm.auto import tqdm
import wandb

from snake_jax.network import TransformerPolicy


class TrainState(train_state.TrainState):
    """Extended train state with batch stats if needed"""
    pass


def load_dataset(dataset_path):
    """Load the pretraining dataset"""
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    # Dataset is a list of dicts with 'state' and 'action' keys
    states = jnp.array([sample['state'] for sample in data], dtype=jnp.float32)
    actions = jnp.array([sample['action'] for sample in data], dtype=jnp.int32)
    
    print(f"Dataset loaded: {len(states)} samples")
    print(f"  State shape: {states.shape}")
    print(f"  Action distribution: {jnp.bincount(actions, length=4)}")
    
    return states, actions


def create_batches(states, actions, batch_size, rng):
    """Create shuffled batches"""
    n_samples = len(states)
    n_batches = n_samples // batch_size
    
    # Shuffle
    perm = jax.random.permutation(rng, n_samples)
    states_shuffled = states[perm]
    actions_shuffled = actions[perm]
    
    # Trim to fit batch size
    states_trimmed = states_shuffled[:n_batches * batch_size]
    actions_trimmed = actions_shuffled[:n_batches * batch_size]
    
    # Reshape into batches
    states_batched = states_trimmed.reshape(n_batches, batch_size, *states.shape[1:])
    actions_batched = actions_trimmed.reshape(n_batches, batch_size)
    
    return states_batched, actions_batched


def compute_metrics(logits, labels):
    """Compute accuracy and loss"""
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    accuracy = (logits.argmax(axis=-1) == labels).mean()
    return loss, accuracy


@jax.jit
def train_step(state, batch_states, batch_actions, dropout_rng):
    """Single training step"""
    
    def loss_fn(params):
        logits, _ = state.apply_fn(params, batch_states, training=True, rngs={'dropout': dropout_rng})
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch_actions).mean()
        return loss, logits
    
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    
    accuracy = (logits.argmax(axis=-1) == batch_actions).mean()
    
    return state, loss, accuracy


def eval_step(params, apply_fn, batch_states, batch_actions):
    """Single evaluation step"""
    # No dropout during evaluation
    logits, _ = apply_fn(params, batch_states, training=False)
    loss, accuracy = compute_metrics(logits, batch_actions)
    return loss, accuracy


def train_epoch(state, states_batched, actions_batched, rng, desc="Training"):
    """Train for one epoch"""
    total_loss = 0.0
    total_acc = 0.0
    n_batches = len(states_batched)
    
    for batch_idx in tqdm(range(n_batches), desc=desc, leave=False):
        # Split RNG for this batch
        rng, dropout_rng = jax.random.split(rng)
        
        state, loss, acc = train_step(
            state,
            states_batched[batch_idx],
            actions_batched[batch_idx],
            dropout_rng
        )
        
        # Block until ready for accurate metrics
        loss = loss.block_until_ready()
        acc = acc.block_until_ready()
        
        total_loss += float(loss)
        total_acc += float(acc)
    
    return state, total_loss / n_batches, total_acc / n_batches, rng


def evaluate(state, states_batched, actions_batched, desc="Validation"):
    """Evaluate on validation set"""
    total_loss = 0.0
    total_acc = 0.0
    n_batches = len(states_batched)
    
    for batch_idx in tqdm(range(n_batches), desc=desc, leave=False):
        loss, acc = eval_step(
            state.params,
            state.apply_fn,
            states_batched[batch_idx],
            actions_batched[batch_idx]
        )
        
        loss = loss.block_until_ready()
        acc = acc.block_until_ready()
        
        total_loss += float(loss)
        total_acc += float(acc)
    
    return total_loss / n_batches, total_acc / n_batches


def main():
    parser = argparse.ArgumentParser(description="Pretrain JAX Transformer on Snake dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset pickle file")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--d-model", type=int, default=64, help="Transformer dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--use-cnn", action="store_true", help="Use CNN encoder")
    parser.add_argument("--cnn-mode", type=str, default="replace", choices=["replace", "append"])
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--wandb", action="store_true", help="Use wandb logging")
    parser.add_argument("--wandb-project", type=str, default="snake-pretrain-jax")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="pretrain_models")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("JAX TRANSFORMER PRETRAINING")
    print("=" * 70)
    print()
    
    # Load dataset
    states, actions = load_dataset(args.dataset)
    
    # Split into train/val
    n_samples = len(states)
    n_val = int(n_samples * args.val_split)
    n_train = n_samples - n_val
    
    train_states = states[:n_train]
    train_actions = actions[:n_train]
    val_states = states[n_train:]
    val_actions = actions[n_train:]
    
    print(f"\nDataset split:")
    print(f"  Training samples: {n_train}")
    print(f"  Validation samples: {n_val}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Batches per epoch: {n_train // args.batch_size}")
    print()
    
    # Initialize model
    print("Initializing model...")
    network = TransformerPolicy(
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_actions=4,
        dropout_rate=args.dropout,
        use_cnn=args.use_cnn,
        cnn_mode=args.cnn_mode,
    )
    
    # Initialize parameters
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)
    
    dummy_input = states[0:1]  # Single sample for initialization
    params = network.init({'params': init_rng, 'dropout': dropout_rng}, dummy_input, training=False)
    
    print(f"Model initialized:")
    print(f"  d_model: {args.d_model}")
    print(f"  layers: {args.num_layers}")
    print(f"  heads: {args.num_heads}")
    print(f"  CNN: {args.use_cnn}")
    if args.use_cnn:
        print(f"  CNN mode: {args.cnn_mode}")
    print()
    
    # Create optimizer
    optimizer = optax.adam(args.lr)
    state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=optimizer,
    )
    
    # Setup wandb
    if args.wandb:
        run_name = args.run_name or f"pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "d_model": args.d_model,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "dropout": args.dropout,
                "use_cnn": args.use_cnn,
                "cnn_mode": args.cnn_mode if args.use_cnn else None,
                "train_samples": n_train,
                "val_samples": n_val,
            }
        )
        print(f"📊 WandB initialized: {wandb.run.url}")
        print()
    
    # Training loop
    print("Starting training...")
    print("=" * 70)
    print()
    
    best_val_acc = 0.0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Create shuffled batches
        rng, shuffle_rng = jax.random.split(rng)
        train_states_batched, train_actions_batched = create_batches(
            train_states, train_actions, args.batch_size, shuffle_rng
        )
        
        val_states_batched, val_actions_batched = create_batches(
            val_states, val_actions, args.batch_size, shuffle_rng
        )
        
        # Train epoch
        state, train_loss, train_acc, rng = train_epoch(
            state, train_states_batched, train_actions_batched, rng,
            desc=f"Epoch {epoch+1}/{args.epochs} [Train]"
        )
        
        # Validate
        val_loss, val_acc = evaluate(
            state, val_states_batched, val_actions_batched,
            desc=f"Epoch {epoch+1}/{args.epochs} [Val]"
        )
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Log to wandb
        if args.wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "epoch_time": epoch_time,
            })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = save_dir / "best_model.pkl"
            with open(best_model_path, 'wb') as f:
                pickle.dump({
                    'params': state.params,
                    'config': {
                        'd_model': args.d_model,
                        'num_layers': args.num_layers,
                        'num_heads': args.num_heads,
                        'dropout': args.dropout,
                        'use_cnn': args.use_cnn,
                        'cnn_mode': args.cnn_mode if args.use_cnn else None,
                    },
                    'val_accuracy': float(val_acc),
                    'epoch': epoch + 1,
                }, f)
            print(f"  🏆 New best model saved! (val_acc: {val_acc:.4f})")
        
        print()
    
    print("=" * 70)
    print("✅ PRETRAINING COMPLETE!")
    print("=" * 70)
    print()
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {save_dir / 'best_model.pkl'}")
    print()
    
    # Save final model
    final_model_path = save_dir / "final_model.pkl"
    with open(final_model_path, 'wb') as f:
        pickle.dump({
            'params': state.params,
            'config': {
                'd_model': args.d_model,
                'num_layers': args.num_layers,
                'num_heads': args.num_heads,
                'dropout': args.dropout,
                'use_cnn': args.use_cnn,
                'cnn_mode': args.cnn_mode if args.use_cnn else None,
            },
            'final_val_accuracy': float(val_acc),
            'epochs': args.epochs,
        }, f)
    print(f"Final model saved to: {final_model_path}")
    
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
