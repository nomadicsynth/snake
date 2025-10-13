"""
Pretrain Snake transformer with teacher-forcing and cross-entropy loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime

from pretrain_dataset import (
    generate_pretraining_dataset,
    save_dataset,
    load_dataset,
    SnakePretrainDataset,
    analyze_dataset
)
from pretrain_model import (
    TransformerPolicyPretrainer,
    MultiTaskPretrainer,
    count_parameters
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    use_soft_labels: bool = False,
    label_smoothing: float = 0.0
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch in pbar:
        states = batch['state'].to(device)
        actions = batch['action'].to(device)
        
        # Forward pass
        logits = model(states)
        
        # Compute loss
        if use_soft_labels:
            action_probs = batch['action_probs'].to(device)
            # KL divergence loss
            log_probs = F.log_softmax(logits, dim=1)
            loss = F.kl_div(log_probs, action_probs, reduction='batchmean')
        else:
            # Cross-entropy loss
            if label_smoothing > 0:
                criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            else:
                criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, actions)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        pred_actions = logits.argmax(dim=1)
        correct += (pred_actions == actions).sum().item()
        total += actions.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.2f}%'
        })
    
    return {
        'loss': total_loss / len(train_loader),
        'accuracy': 100.0 * correct / total
    }


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    use_soft_labels: bool = False
) -> dict:
    """Evaluate model on validation set."""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Per-action metrics
    action_correct = [0, 0, 0, 0]
    action_total = [0, 0, 0, 0]
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            states = batch['state'].to(device)
            actions = batch['action'].to(device)
            
            logits = model(states)
            
            # Compute loss
            if use_soft_labels:
                action_probs = batch['action_probs'].to(device)
                log_probs = F.log_softmax(logits, dim=1)
                loss = F.kl_div(log_probs, action_probs, reduction='batchmean')
            else:
                criterion = nn.CrossEntropyLoss()
                loss = criterion(logits, actions)
            
            total_loss += loss.item()
            
            # Accuracy
            pred_actions = logits.argmax(dim=1)
            correct += (pred_actions == actions).sum().item()
            total += actions.size(0)
            
            # Per-action accuracy
            for i in range(4):
                mask = actions == i
                action_total[i] += mask.sum().item()
                action_correct[i] += ((pred_actions == actions) & mask).sum().item()
    
    # Compute per-action accuracies
    action_names = ['Up', 'Right', 'Down', 'Left']
    action_accs = {}
    for i, name in enumerate(action_names):
        if action_total[i] > 0:
            action_accs[name] = 100.0 * action_correct[i] / action_total[i]
        else:
            action_accs[name] = 0.0
    
    return {
        'loss': total_loss / len(val_loader),
        'accuracy': 100.0 * correct / total,
        'action_accuracies': action_accs
    }


def plot_training_curves(history: dict, save_path: str):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy curve
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {save_path}")


def pretrain(
    dataset_path: str,
    output_dir: str,
    # Model args
    d_model: int = 64,
    num_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.1,
    # Training args
    batch_size: int = 256,
    num_epochs: int = 50,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    warmup_epochs: int = 5,
    use_soft_labels: bool = False,
    label_smoothing: float = 0.1,
    # Data args
    val_split: float = 0.1,
    # System args
    device: str = 'cuda',
    seed: int = 42,
    num_workers: int = 4
):
    """
    Main pretraining function.
    
    Args:
        dataset_path: Path to pickled dataset
        output_dir: Directory to save model and logs
        d_model: Transformer hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        batch_size: Training batch size
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization
        warmup_epochs: Number of warmup epochs
        use_soft_labels: Use soft labels with KL loss
        label_smoothing: Label smoothing factor (if not using soft labels)
        val_split: Validation split ratio
        device: Device to train on
        seed: Random seed
        num_workers: DataLoader workers
    """
    set_seed(seed)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = {
        'model': {
            'd_model': d_model,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'dropout': dropout
        },
        'training': {
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'lr': lr,
            'weight_decay': weight_decay,
            'warmup_epochs': warmup_epochs,
            'use_soft_labels': use_soft_labels,
            'label_smoothing': label_smoothing
        },
        'data': {
            'dataset_path': dataset_path,
            'val_split': val_split
        },
        'seed': seed,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(dataset_path)
    
    # Analyze dataset
    analyze_dataset(dataset)
    
    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_data, val_data = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    # Create datasets and loaders
    train_dataset = SnakePretrainDataset(
        [dataset[i] for i in train_data.indices],
        use_soft_labels=use_soft_labels
    )
    val_dataset = SnakePretrainDataset(
        [dataset[i] for i in val_data.indices],
        use_soft_labels=use_soft_labels
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Get grid dimensions from first sample
    sample_state = dataset[0]['state']
    height, width, _ = sample_state.shape
    
    # Create model
    print(f"\nCreating model (grid={height}x{width}, d_model={d_model}, "
          f"layers={num_layers}, heads={num_heads})...")
    
    model = TransformerPolicyPretrainer(
        height=height,
        width=width,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...\n")
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            use_soft_labels=use_soft_labels,
            label_smoothing=label_smoothing if not use_soft_labels else 0.0
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, device, use_soft_labels=use_soft_labels)
        
        # Update scheduler
        scheduler.step()
        
        # Log
        print(f"Train Loss: {train_metrics['loss']:.4f}, "
              f"Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.2f}%")
        print(f"Val Acc by action: {val_metrics['action_accuracies']}")
        print()
        
        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['lr'].append(scheduler.get_last_lr()[0])
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['accuracy'],
                'config': config
            }, output_path / 'best_model.pth')
            print(f"Saved new best model (val_acc={best_val_acc:.2f}%)")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['accuracy'],
                'config': config
            }, output_path / f'checkpoint_epoch_{epoch + 1}.pth')
    
    # Save final model
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_metrics['accuracy'],
        'config': config
    }, output_path / 'final_model.pth')
    
    # Save history
    with open(output_path / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot curves
    plot_training_curves(history, str(output_path / 'training_curves.png'))
    
    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Models and logs saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain Snake Transformer with teacher-forcing"
    )
    
    # I/O
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to dataset pickle file')
    parser.add_argument('--output-dir', type=str, default='pretrain_output',
                       help='Output directory for models and logs')
    
    # Model architecture
    parser.add_argument('--d-model', type=int, default=64,
                       help='Transformer hidden dimension')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of transformer layers')
    parser.add_argument('--num-heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                       help='Warmup epochs')
    parser.add_argument('--use-soft-labels', action='store_true',
                       help='Use soft labels with KL divergence loss')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                       help='Label smoothing (if not using soft labels)')
    
    # Data
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split ratio')
    
    # System
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='DataLoader workers')
    
    args = parser.parse_args()
    
    # Check dataset exists
    if not Path(args.dataset).exists():
        print(f"Error: Dataset file '{args.dataset}' not found!")
        print("Generate dataset first using pretrain_dataset.py")
        return
    
    # Run pretraining
    pretrain(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        use_soft_labels=args.use_soft_labels,
        label_smoothing=args.label_smoothing,
        val_split=args.val_split,
        device=args.device,
        seed=args.seed,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()
