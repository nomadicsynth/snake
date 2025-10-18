#!/usr/bin/env python3
"""
Train Snake Transformer with HuggingFace Trainer

PyTorch + Transformers + Datasets implementation
"""

import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from datasets import load_from_disk
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
import wandb

from model_pytorch import TransformerPolicy, SnakeTransformerConfig

try:
    from muon import Muon
    try:
        from muon import MuonWithAuxAdam
        MUON_AUX_AVAILABLE = True
    except ImportError:
        MUON_AUX_AVAILABLE = False
    MUON_AVAILABLE = True
except ImportError:
    MUON_AVAILABLE = False
    MUON_AUX_AVAILABLE = False


class SnakeDataCollator:
    """Custom collator for Snake dataset"""
    
    def __init__(self, use_reasoning=False):
        self.use_reasoning = use_reasoning
    
    def __call__(self, features):
        """
        Collate batch of samples
        
        Args:
            features: List of dicts with 'state', 'action', optionally 'reasoning_tokens'
            
        Returns:
            Dict with batched tensors
        """
        # Stack states and transpose to (batch, channels, height, width)
        states = np.stack([f['state'] for f in features])  # (batch, H, W, 3)
        states = np.transpose(states, (0, 3, 1, 2))  # (batch, 3, H, W)
        states = torch.tensor(states, dtype=torch.float32)
        
        actions = torch.tensor([f['action'] for f in features], dtype=torch.long)
        
        batch = {
            'obs': states,
            'labels': actions,
        }
        
        if self.use_reasoning:
            reasoning = torch.tensor(np.stack([f['reasoning_tokens'] for f in features]), dtype=torch.long)
            batch['reasoning_tokens'] = reasoning
        
        return batch


class SnakeTrainer(Trainer):
    """Custom Trainer for Snake policy"""
    
    def __init__(self, *args, use_reasoning=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_reasoning = use_reasoning
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss computation
        
        Args:
            model: The model
            inputs: Dict with 'obs', 'labels', optionally 'reasoning_tokens'
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch (new in transformers API)
            
        Returns:
            loss or (loss, outputs)
        """
        obs = inputs['obs']
        labels = inputs['labels']
        reasoning_tokens = inputs.get('reasoning_tokens', None)
        
        # Forward pass
        logits, hidden = model(obs, reasoning_tokens=reasoning_tokens)
        
        # For RSM models, extract action logits
        if self.use_reasoning:
            action_logits = logits[:, :model.num_actions]
        else:
            action_logits = logits
        
        # Cross-entropy loss
        loss = nn.functional.cross_entropy(action_logits, labels)
        
        if return_outputs:
            return loss, {'logits': action_logits, 'hidden': hidden}
        return loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Custom prediction step for evaluation
        """
        device = next(model.parameters()).device
        
        with torch.no_grad():
            obs = inputs['obs'].to(device)
            labels = inputs['labels'].to(device)
            reasoning_tokens = inputs.get('reasoning_tokens', None)
            if reasoning_tokens is not None:
                reasoning_tokens = reasoning_tokens.to(device)
            
            # Forward pass
            logits, _ = model(obs, reasoning_tokens=reasoning_tokens)
            
            # Extract action logits for RSM models
            if self.use_reasoning:
                action_logits = logits[:, :model.num_actions]
            else:
                action_logits = logits
            
            # Compute loss
            loss = nn.functional.cross_entropy(action_logits, labels)
            
            if prediction_loss_only:
                return (loss, None, None)
            
            return (loss, action_logits, labels)


class MetricsCallback(TrainerCallback):
    """Callback to compute and log accuracy"""
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """Compute accuracy from eval predictions"""
        # Accuracy is computed by compute_metrics function
        pass


def compute_metrics(eval_pred):
    """
    Compute accuracy metric
    
    Args:
        eval_pred: EvalPrediction with predictions and labels
        
    Returns:
        Dict with metrics
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    
    return {'accuracy': accuracy}


def main():
    parser = argparse.ArgumentParser(description="Train Snake Transformer with HuggingFace")
    
    # Dataset
    parser.add_argument("--dataset", type=str, required=True, 
                        help="Path to HuggingFace dataset directory")
    
    # Model architecture
    parser.add_argument("--d-model", type=int, default=128, help="Transformer dimension")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--use-cnn", action="store_true", default=True, help="Use CNN encoder")
    parser.add_argument("--cnn-mode", type=str, default="append", choices=["replace", "append"],
                        help="CNN mode: replace grid tokens or append")
    
    # Training
    parser.add_argument("--output-dir", type=str, default="outputs/models/snake_hf_output",
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--eval-batch-size", type=int, default=512, help="Evaluation batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.0,
                        help="Warmup ratio (fraction of total steps)")
    parser.add_argument("--lr-scheduler", type=str, default="cosine",
                        choices=["linear", "cosine", "constant"],
                        help="Learning rate scheduler")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm for clipping")
    
    # Optimizer
    parser.add_argument("--optimizer", type=str, default="muon", choices=["adamw", "muon"], help="Optimizer to use")
    parser.add_argument("--muon-lr", type=float, default=0.01, help="Learning rate for Muon optimizer")
    parser.add_argument("--muon-momentum", type=float, default=0.95, help="Momentum for Muon")

    # Logging and saving
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="snake-hf", help="W&B project name")
    parser.add_argument("--run-name", type=str, default=None, help="Run name for logging")
    parser.add_argument("--logging-steps", type=int, default=50, help="Log every N steps")
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--save-steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--save-total-limit", type=int, default=3, help="Maximum number of checkpoints to keep")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bfloat16 precision")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")

    args = parser.parse_args()
    
    print("=" * 70)
    print("SNAKE TRANSFORMER TRAINING (HuggingFace)")
    print("=" * 70)
    print()
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    dataset = load_from_disk(args.dataset)
    print(f"✅ Dataset loaded:")
    print(f"   Train: {len(dataset['train']):,} samples")
    print(f"   Validation: {len(dataset['validation']):,} samples")
    
    # Check if RSM mode
    sample = dataset['train'][0]
    use_reasoning = 'reasoning_tokens' in sample
    
    if use_reasoning:
        print("\n🧠 REASONING SNAKE MODEL (RSM) MODE DETECTED")
        print(f"   Reasoning sequence length: {len(sample['reasoning_tokens'])}")
    
    print()
    
    # Create model
    print("Initializing model...")
    config = SnakeTransformerConfig(
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_actions=4,
        dropout_rate=args.dropout,
        use_cnn=args.use_cnn,
        cnn_mode=args.cnn_mode,
        use_reasoning=use_reasoning,
    )
    
    model = TransformerPolicy(config)
    
    print(f"✅ Model initialized:")
    print(f"   d_model: {args.d_model}")
    print(f"   layers: {args.num_layers}")
    print(f"   heads: {args.num_heads}")
    print(f"   CNN: {args.use_cnn}")
    if args.use_cnn:
        print(f"   CNN mode: {args.cnn_mode}")
    if use_reasoning:
        print(f"   RSM: Enabled")
    print()
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,} total, {n_trainable:,} trainable")
    print()
    
    # Setup training arguments
    run_name = args.run_name or f"snake_hf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) / run_name
    
    # Optimizer configuration
    use_muon = args.optimizer == "muon"
    if use_muon:
        if not MUON_AVAILABLE:
            print("⚠️  Muon optimizer requested but not installed!")
            print("    Install with: pip install git+https://github.com/KellerJordan/Muon")
            print("    Falling back to AdamW")
            print()
            use_muon = False
        else:
            # Initialize distributed for Muon (single process)
            if not dist.is_initialized():
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '12355'
                os.environ['RANK'] = '0'
                os.environ['WORLD_SIZE'] = '1'
                dist.init_process_group(backend='gloo', rank=0, world_size=1)
            
            print(f"✅ Using Muon optimizer")
            print(f"   LR: {args.muon_lr}")
            print(f"   Momentum: {args.muon_momentum}")
            print()
            # Override learning rate for Muon
            effective_lr = args.muon_lr
    
    if not use_muon:
        effective_lr = args.lr
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=effective_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        
        # Logging
        logging_dir=str(output_dir / "logs"),
        logging_steps=args.logging_steps,
        report_to="wandb" if args.wandb else "none",
        run_name=run_name,
        
        # Saving
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        
        # Other
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=4,
        remove_unused_columns=False,  # Important: keep all columns
    )
    
    # Create custom optimizer using Muon pattern from reference implementation
    optimizers = (None, None)
    if use_muon:
        if MUON_AUX_AVAILABLE:
            # Use MuonWithAuxAdam for proper handling of 2D+ params with Muon and 1D params with AdamW
            muon_params = []
            aux_params = []
            
            for name, p in model.named_parameters():
                if p.requires_grad:
                    if p.ndim >= 2:
                        muon_params.append(p)
                    else:
                        aux_params.append(p)
            
            param_groups = []
            if len(muon_params) > 0:
                param_groups.append({
                    'params': muon_params,
                    'use_muon': True,
                    'lr': args.muon_lr,
                    'weight_decay': args.weight_decay,
                })
            
            if len(aux_params) > 0:
                param_groups.append({
                    'params': aux_params,
                    'use_muon': False,
                    'lr': args.lr,  # Use regular LR for aux params
                    'betas': (0.9, 0.95),
                    'weight_decay': args.weight_decay,
                })
            
            if len(param_groups) > 0:
                optimizer = MuonWithAuxAdam(param_groups)
                optimizers = (optimizer, None)
                print(f"   MuonWithAuxAdam: {len(muon_params)} weight params, {len(aux_params)} aux params")
        else:
            print("⚠️  MuonWithAuxAdam not available, using basic Muon (2D params only)")
            # Fallback to basic Muon for 2D params only
            muon_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
            if len(muon_params) > 0:
                optimizer = Muon(muon_params, lr=args.muon_lr, momentum=args.muon_momentum)
                optimizers = (optimizer, None)
                print(f"   Muon: {len(muon_params)} params (2D only, 1D params not optimized!)")
    
    # Initialize W&B if requested
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                **vars(args),
                **vars(config),
            },
        )
        print(f"📊 Weights & Biases initialized: {wandb.run.url}")
        print()
    
    # Data collator
    data_collator = SnakeDataCollator(use_reasoning=use_reasoning)
    
    # Create trainer
    trainer = SnakeTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[MetricsCallback()],
        use_reasoning=use_reasoning,
        optimizers=optimizers,
    )
    
    # Resume from checkpoint if specified
    checkpoint = None
    if args.resume:
        checkpoint = args.resume
        print(f"Resuming from checkpoint: {checkpoint}")
    else:
        # Check for last checkpoint in output directory
        last_checkpoint = get_last_checkpoint(str(output_dir))
        if last_checkpoint:
            print(f"Found checkpoint: {last_checkpoint}")
            checkpoint = last_checkpoint
    
    # Train
    print("Starting training...")
    print("=" * 70)
    print()
    
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    print()
    print("=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print()
    
    # Save final model
    print("Saving final model...")
    trainer.save_model(str(output_dir / "final_model"))
    
    # Save metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Final evaluation
    print("\nRunning final evaluation...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    print(f"\nFinal Results:")
    print(f"  Train Loss: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"  Val Loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")
    print(f"  Val Accuracy: {eval_metrics.get('eval_accuracy', 'N/A'):.4f}")
    print()
    
    print(f"Model saved to: {output_dir / 'final_model'}")
    print()
    
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
