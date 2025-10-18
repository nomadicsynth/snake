#!/usr/bin/env python3
"""
Train Snake World Model with Equilibrium Matching (EqM)

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
)
from transformers.trainer_utils import get_last_checkpoint
import wandb

from model.model_eqm import SnakeWorldEqM, SnakeWorldEqMConfig


class SnakeWorldDataCollator:
    """
    Custom collator for Snake world model dataset
    
    NOTE: This expects a dataset with 'state', 'next_state', and 'action' fields.
    The standard Snake dataset only has 'state' and 'action'.
    
    To create a dataset with next_state:
    1. Collect trajectories with (state_t, action_t, state_t+1) tuples
    2. Or use a simulator to compute next_state from (state_t, action_t)
    
    For initial testing, you can modify generate_dataset_hf.py to include next_state.
    """
    
    def __call__(self, features):
        """
        Collate batch of samples
        
        Args:
            features: List of dicts with 'state', 'next_state', 'action'
            
        Returns:
            Dict with batched tensors
        """
        # Stack states and transpose to (batch, channels, height, width)
        states = np.stack([f['state'] for f in features])  # (batch, H, W, 3)
        states = np.transpose(states, (0, 3, 1, 2))  # (batch, 3, H, W)
        states = torch.tensor(states, dtype=torch.float32)
        
        # Stack next states
        next_states = np.stack([f['next_state'] for f in features])  # (batch, H, W, 3)
        next_states = np.transpose(next_states, (0, 3, 1, 2))  # (batch, 3, H, W)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        
        actions = torch.tensor([f['action'] for f in features], dtype=torch.long)
        
        batch = {
            'obs': states,
            'next_obs': next_states,
            'actions': actions,
        }
        
        return batch


class SnakeWorldTrainer(Trainer):
    """Custom Trainer for Snake World Model with EqM"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss computation using EqM objective
        
        Args:
            model: The SnakeWorldEqM model
            inputs: Dict with 'obs', 'next_obs', 'actions'
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch (new in transformers API)
            
        Returns:
            loss or (loss, outputs)
        """
        obs = inputs['obs']
        next_obs = inputs['next_obs']
        actions = inputs['actions']
        
        # Forward pass with EqM training mode
        loss_dict = model(
            obs=obs,
            next_obs=next_obs,
            actions=actions,
            mode='train',
        )
        
        total_loss = loss_dict['loss']
        
        # Log component losses
        if self.state.global_step % self.args.logging_steps == 0:
            for key, value in loss_dict.items():
                if key != 'loss' and isinstance(value, torch.Tensor):
                    self.log({f'train/{key}': value.item()})
        
        if return_outputs:
            return total_loss, loss_dict
        return total_loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Custom prediction step for evaluation
        
        During evaluation, we:
        1. Compute the training loss (EqM objective)
        2. Sample actions and compute accuracy
        """
        device = next(model.parameters()).device
        
        with torch.no_grad():
            obs = inputs['obs'].to(device)
            next_obs = inputs['next_obs'].to(device)
            actions = inputs['actions'].to(device)
            
            # Compute EqM loss
            loss_dict = model(
                obs=obs,
                next_obs=next_obs,
                actions=actions,
                mode='train',
            )
            loss = loss_dict['loss']
            
            # Sample actions for accuracy computation
            sample_dict = model.sample(obs)
            predicted_actions = sample_dict['action']
            
            if prediction_loss_only:
                return (loss, None, None)
            
            return (loss, predicted_actions, actions)


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
    parser = argparse.ArgumentParser(description="Train Snake World Model with EqM")
    
    # Dataset
    parser.add_argument("--dataset", type=str, required=True, 
                        help="Path to HuggingFace dataset directory")
    
    # Model architecture - Encoder
    parser.add_argument("--d-model", type=int, default=128, help="Transformer dimension")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--use-cnn", action="store_true", default=True, help="Use CNN encoder")
    parser.add_argument("--cnn-mode", type=str, default="append", choices=["replace", "append"],
                        help="CNN mode: replace grid tokens or append")
    
    # EqM-specific architecture
    parser.add_argument("--latent-dim", type=int, default=64, 
                        help="Dimension of latent next-state representation")
    parser.add_argument("--eqm-hidden-dim", type=int, default=128,
                        help="Hidden dimension for EqM gradient network")
    parser.add_argument("--eqm-num-layers", type=int, default=3,
                        help="Number of layers in EqM gradient network")
    
    # EqM gradient schedule
    parser.add_argument("--gradient-schedule", type=str, default="linear",
                        choices=["linear", "truncated", "piecewise"],
                        help="Gradient magnitude schedule c(gamma)")
    parser.add_argument("--gradient-truncate-a", type=float, default=0.3,
                        help="Truncation point for truncated/piecewise schedules")
    parser.add_argument("--gradient-piecewise-b", type=float, default=2.0,
                        help="Starting value for piecewise schedule")
    parser.add_argument("--gradient-multiplier", type=float, default=1.0,
                        help="Overall gradient scale (lambda)")
    
    # EqM sampling config
    parser.add_argument("--eqm-sampling-steps", type=int, default=10,
                        help="Number of gradient descent steps during sampling")
    parser.add_argument("--eqm-step-size", type=float, default=0.1,
                        help="Step size for gradient descent")
    parser.add_argument("--eqm-nag-momentum", type=float, default=0.9,
                        help="Nesterov momentum for sampling")
    parser.add_argument("--eqm-adaptive-threshold", type=float, default=None,
                        help="Gradient norm threshold for adaptive compute (None = fixed steps)")
    
    # Optional explicit energy
    parser.add_argument("--use-explicit-energy", action="store_true",
                        help="Use explicit energy learning")
    parser.add_argument("--energy-type", type=str, default="dot_product",
                        choices=["dot_product", "squared_l2"],
                        help="Type of explicit energy function")
    
    # Training
    parser.add_argument("--output-dir", type=str, default="outputs/models/snake_world_eqm",
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
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon"], 
                        help="Optimizer to use")
    parser.add_argument("--muon-lr", type=float, default=0.01, help="Learning rate for Muon optimizer")
    parser.add_argument("--muon-momentum", type=float, default=0.95, help="Momentum for Muon")

    # Logging and saving
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="snake-world-eqm", help="W&B project name")
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
    
    use_muon = args.optimizer == "muon"
    if use_muon:
        try:
            from muon import MuonWithAuxAdam
        except ImportError:
            print("⚠️  Muon optimizer requested but MuonWithAuxAdam not available!")
            print("    Install with: pip install git+https://github.com/KellerJordan/Muon")
            raise ImportError("MuonWithAuxAdam is required for muon optimizer")
    
    print("=" * 70)
    print("SNAKE WORLD MODEL TRAINING (EqM)")
    print("=" * 70)
    print()
    
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    dataset = load_from_disk(args.dataset)
    print(f"✅ Dataset loaded:")
    print(f"   Train: {len(dataset['train']):,} samples")
    print(f"   Validation: {len(dataset['validation']):,} samples")
    
    # Check dataset format
    sample = dataset['train'][0]
    if 'next_state' not in sample:
        raise ValueError(
            "Dataset must contain 'next_state' field for world model training!\n"
            "Expected fields: 'state', 'next_state', 'action'"
        )
    
    print()
    
    # Create model
    print("Initializing EqM World Model...")
    config = SnakeWorldEqMConfig(
        # Encoder config
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_actions=4,
        dropout_rate=args.dropout,
        use_cnn=args.use_cnn,
        cnn_mode=args.cnn_mode,
        
        # EqM config
        latent_dim=args.latent_dim,
        eqm_hidden_dim=args.eqm_hidden_dim,
        eqm_num_layers=args.eqm_num_layers,
        
        # Gradient schedule
        gradient_schedule=args.gradient_schedule,
        gradient_truncate_a=args.gradient_truncate_a,
        gradient_piecewise_b=args.gradient_piecewise_b,
        gradient_multiplier=args.gradient_multiplier,
        
        # Sampling
        eqm_sampling_steps=args.eqm_sampling_steps,
        eqm_step_size=args.eqm_step_size,
        eqm_nag_momentum=args.eqm_nag_momentum,
        eqm_adaptive_threshold=args.eqm_adaptive_threshold,
        
        # Explicit energy
        use_explicit_energy=args.use_explicit_energy,
        energy_type=args.energy_type,
    )
    
    model = SnakeWorldEqM(config)
    
    print(f"✅ Model initialized:")
    print(f"   Encoder d_model: {args.d_model}")
    print(f"   Encoder layers: {args.num_layers}")
    print(f"   Encoder heads: {args.num_heads}")
    print(f"   CNN: {args.use_cnn}")
    if args.use_cnn:
        print(f"   CNN mode: {args.cnn_mode}")
    print(f"\n   EqM latent dim: {args.latent_dim}")
    print(f"   EqM hidden dim: {args.eqm_hidden_dim}")
    print(f"   EqM layers: {args.eqm_num_layers}")
    print(f"   Gradient schedule: {args.gradient_schedule}")
    print(f"   Gradient multiplier: {args.gradient_multiplier}")
    print(f"   Sampling steps: {args.eqm_sampling_steps}")
    print(f"   Step size: {args.eqm_step_size}")
    print(f"   NAG momentum: {args.eqm_nag_momentum}")
    if args.use_explicit_energy:
        print(f"   Explicit energy: {args.energy_type}")
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
        # Build parameter groups similar to the reference:
        # - hidden_weights: 2D weight parameters from the model "body" (transformer + CNN) -> use_muon=True
        # - hidden_gains_biases + nonhidden_params: 1D params from body + head/embed parameters -> use_muon=False
        # We map our model modules to those concepts: transformer and CNN (and cnn_proj/input_proj) are the "body",
        # action_head/reasoning_head and reasoning embeddings are treated as head/embed respectively.

        body_modules = []
        if hasattr(model, 'transformer'):
            body_modules.append(model.transformer)
        if hasattr(model, 'cnn'):
            body_modules.append(model.cnn)
        if hasattr(model, 'cnn_proj'):
            body_modules.append(model.cnn_proj)
        if hasattr(model, 'input_proj'):
            body_modules.append(model.input_proj)

        hidden_weights = []
        hidden_gains_biases = []

        for mod in body_modules:
            for p in mod.parameters():
                if not p.requires_grad:
                    continue
                if p.ndim >= 2:
                    hidden_weights.append(p)
                else:
                    hidden_gains_biases.append(p)

        # Non-hidden params: heads and embedding layers
        nonhidden_params = []
        for name in ('action_head', 'reasoning_head', 'reasoning_embed', 'reasoning_pos'):
            if hasattr(model, name):
                nonhidden_params.extend([p for p in getattr(model, name).parameters() if p.requires_grad])

        # Deduplicate parameters (a parameter can be referenced in multiple lists through shared modules)
        def unique_params(param_list):
            seen = set()
            uniq = []
            for p in param_list:
                if id(p) in seen:
                    continue
                seen.add(id(p))
                uniq.append(p)
            return uniq

        hidden_weights = unique_params(hidden_weights)
        hidden_gains_biases = unique_params(hidden_gains_biases)
        nonhidden_params = unique_params(nonhidden_params)

        # Fallback: if some trainable params were not captured (rare), put them into the aux group
        captured = set([id(p) for p in hidden_weights + hidden_gains_biases + nonhidden_params])
        other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in captured]

        # Compose final param groups
        param_groups = []
        if len(hidden_weights) > 0:
            param_groups.append({
                'params': hidden_weights,
                'use_muon': True,
                'lr': args.muon_lr,
                'weight_decay': args.weight_decay,
            })

        aux_group_params = hidden_gains_biases + nonhidden_params + other_params
        if len(aux_group_params) > 0:
            param_groups.append({
                'params': aux_group_params,
                'use_muon': False,
                'lr': args.lr,
                'betas': (0.9, 0.95),
                'weight_decay': args.weight_decay,
            })

        if len(param_groups) > 0:
            optimizer = MuonWithAuxAdam(param_groups)
            optimizers = (optimizer, None)
            n_muon = sum(p.numel() for g in param_groups if g.get('use_muon') for p in g['params'])
            n_aux = sum(p.numel() for g in param_groups if not g.get('use_muon') for p in g['params'])
            print(f"   MuonWithAuxAdam: {n_muon:,} muon params, {n_aux:,} aux params")
    
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
    data_collator = SnakeWorldDataCollator()
    
    # Create trainer
    trainer = SnakeWorldTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[MetricsCallback()],
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
