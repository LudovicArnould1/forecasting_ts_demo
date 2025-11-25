"""
Simplified training pipeline with wandb integration for time series forecasting.

This module provides a concise pipeline for training and evaluation:
- Training: Train model with checkpointing and wandb logging
- Evaluation: Evaluate on held-out datasets

All stages log to wandb (offline mode supported).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import wandb
import yaml
from torch.optim import AdamW

from data_processing.dataloader import create_train_dataloader
from model.moirai import MinimalMOIRAI, mse_loss
from monitor_and_setup.checkpoint import CheckpointManager
from monitor_and_setup.reproducibility import log_environment_to_wandb, set_seed

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    
    loss: float
    mse: float
    mae: float
    grad_norm: float
    pred_std: float
    target_std: float


class TrainingPipeline:
    """
    Simplified training pipeline with wandb integration.
    
    Features:
    - Automatic checkpoint management
    - Wandb logging (offline supported)
    - Resumption support
    - Multi-dataset evaluation
    """
    
    def __init__(
        self,
        config_dir: Path | str = "config",
        project: str = "ts-forecasting",
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        offline: bool = True,
    ):
        """
        Initialize training pipeline.
        
        Args:
            config_dir: Directory containing config YAML files
            project: W&B project name
            entity: W&B entity (username/team)
            run_name: Optional run name
            offline: Run wandb in offline mode
        """
        self.config_dir = Path(config_dir)
        self.project = project
        self.entity = entity
        self.run_name = run_name
        self.offline = offline
        
        # Load configs
        self.data_config = self._load_yaml("data.yaml")
        self.model_config = self._load_yaml("model.yaml")
        self.training_config = self._load_yaml("training.yaml")
        
        self.wandb_run = None
        self.checkpoint_manager = None
        
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load YAML config file."""
        config_path = self.config_dir / filename
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    
    def _train_step(
        self,
        model: nn.Module,
        batch: dict,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        max_grad_norm: float = 1.0,
    ) -> Optional[TrainingMetrics]:
        """
        Execute one training step.
        
        Args:
            model: The forecasting model
            batch: Batch from dataloader
            optimizer: Optimizer
            device: Device to train on
            max_grad_norm: Maximum gradient norm for clipping
            
        Returns:
            TrainingMetrics if successful, None if batch invalid
        """
        model.train()
        
        # Move data to device
        context = batch["context_patches"].to(device)
        target = batch["target_patches"].to(device)
        context_mask = batch["context_mask"].to(device)
        target_mask = batch["target_mask"].to(device)
        
        # Check for NaN
        if torch.isnan(context).any() or torch.isnan(target).any():
            logger.warning("Batch contains NaN, skipping...")
            return None
        
        # Get prediction length from target shape
        batch_size, num_variates, pred_len, patch_size = target.shape
        
        # Forward pass
        predictions, target_norm = model(
            context=context,
            context_mask=context_mask,
            prediction_length=pred_len,
            target=target,
        )
        
        # Compute loss
        loss = mse_loss(predictions, target_norm, target_mask)
        
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("Loss is NaN/Inf, skipping batch...")
            return None
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=max_grad_norm
        )
        
        optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            mse = ((predictions - target_norm) ** 2 * target_mask.unsqueeze(1).unsqueeze(-1)).sum() / target_mask.sum()
            mae = ((predictions - target_norm).abs() * target_mask.unsqueeze(1).unsqueeze(-1)).sum() / target_mask.sum()
            pred_std = predictions.std().item()
            target_std = target.std().item()
        
        return TrainingMetrics(
            loss=loss.item(),
            mse=mse.item(),
            mae=mae.item(),
            grad_norm=grad_norm.item(),
            pred_std=pred_std,
            target_std=target_std,
        )
    
    @torch.no_grad()
    def _eval_step(
        self,
        model: nn.Module,
        batch: dict,
        device: torch.device,
    ) -> Optional[Dict[str, float]]:
        """
        Execute one evaluation step.
        
        Args:
            model: The forecasting model
            batch: Batch from dataloader
            device: Device to evaluate on
            
        Returns:
            Dictionary with metrics, or None if batch invalid
        """
        model.eval()
        
        # Move data to device
        context = batch["context_patches"].to(device)
        target = batch["target_patches"].to(device)
        context_mask = batch["context_mask"].to(device)
        target_mask = batch["target_mask"].to(device)
        
        # Check for NaN
        if torch.isnan(context).any() or torch.isnan(target).any():
            return None
        
        # Get prediction length
        batch_size, num_variates, pred_len, patch_size = target.shape
        
        # Forward pass
        predictions, target_norm = model(
            context=context,
            context_mask=context_mask,
            prediction_length=pred_len,
            target=target,
        )
        
        # Compute loss
        loss = mse_loss(predictions, target_norm, target_mask)
        
        if torch.isnan(loss) or torch.isinf(loss):
            return None
        
        # Compute metrics
        mse = ((predictions - target_norm) ** 2 * target_mask.unsqueeze(1).unsqueeze(-1)).sum() / target_mask.sum()
        mae = ((predictions - target_norm).abs() * target_mask.unsqueeze(1).unsqueeze(-1)).sum() / target_mask.sum()
        
        return {
            "loss": loss.item(),
            "mse": mse.item(),
            "mae": mae.item(),
        }
    
    def _evaluate_on_dataset(
        self,
        model: nn.Module,
        dataset_name: str,
        device: torch.device,
        batch_size: int = 64,
        num_eval_batches: int = 10,
    ) -> Optional[Dict[str, float]]:
        """
        Evaluate model on a specific dataset.
        
        Args:
            model: Trained model
            dataset_name: Name of dataset to evaluate on
            device: Device to evaluate on
            batch_size: Batch size for evaluation
            num_eval_batches: Number of batches to evaluate
            
        Returns:
            Dictionary with average metrics, or None if evaluation failed
        """
        logger.info(f"\nEvaluating on {dataset_name}...")
        
        data_dir = Path("data/training") / dataset_name
        
        if not data_dir.exists():
            logger.warning(f"Dataset not found: {data_dir}")
            return None
        
        # Create eval dataloader
        loader_config = self.data_config["loader_config"]
        
        try:
            eval_loader = create_train_dataloader(
                data_dir=data_dir,
                batch_size=batch_size,
                num_workers=self.data_config.get("num_workers", 2),
                **loader_config,
            )
        except Exception as e:
            logger.error(f"Failed to create dataloader for {dataset_name}: {e}")
            logger.info(f"Skipping evaluation on {dataset_name}")
            return None
        
        # Evaluate on multiple batches
        eval_metrics = []
        eval_attempts = 0
        
        for batch in eval_loader:
            if len(eval_metrics) >= num_eval_batches:
                break
            eval_attempts += 1
            if eval_attempts > num_eval_batches * 3:
                break
            
            metrics = self._eval_step(model, batch, device)
            if metrics is not None:
                eval_metrics.append(metrics)
        
        if not eval_metrics:
            logger.warning(f"No valid evaluation batches for {dataset_name}")
            return None
        
        # Compute average metrics
        avg_metrics = {
            "loss": sum(m["loss"] for m in eval_metrics) / len(eval_metrics),
            "mse": sum(m["mse"] for m in eval_metrics) / len(eval_metrics),
            "mae": sum(m["mae"] for m in eval_metrics) / len(eval_metrics),
            "num_batches": len(eval_metrics),
        }
        
        logger.info(f"  Loss: {avg_metrics['loss']:.4f} | MSE: {avg_metrics['mse']:.4f} | MAE: {avg_metrics['mae']:.4f}")
        
        return avg_metrics
    
    def train(
        self,
        resume_from_checkpoint: bool = False,
        checkpoint_dir: Optional[Path | str] = None,
    ) -> Dict[str, Any]:
        """
        Run full training pipeline.
        
        Args:
            resume_from_checkpoint: Whether to resume from existing checkpoint
            checkpoint_dir: Directory for checkpoints (defaults to checkpoints/)
            
        Returns:
            Dictionary with training results
        """
        # Set checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = Path("checkpoints")
        else:
            checkpoint_dir = Path(checkpoint_dir)
        
        # Check for existing checkpoint
        wandb_run_id = None
        if resume_from_checkpoint:
            temp_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
            if temp_manager.has_checkpoint():
                wandb_run_id = temp_manager.get_resume_run_id()
                if wandb_run_id:
                    logger.info(f"Resuming W&B run: {wandb_run_id}")
        
        # Initialize wandb
        if self.offline:
            import os
            os.environ["WANDB_MODE"] = "offline"
            logger.info("Running in OFFLINE mode - logs saved locally")
        
        self.wandb_run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.run_name,
            config={
                **self.data_config,
                **self.model_config,
                **self.training_config,
            },
            resume="allow" if resume_from_checkpoint else None,
            id=wandb_run_id if resume_from_checkpoint else None,
        )
        
        # Log environment
        log_environment_to_wandb(self.wandb_run)
        
        # Set seed for reproducibility
        seed = self.training_config.get("seed", 42)
        set_seed(seed, deterministic=self.training_config.get("deterministic", False))
        
        # Setup device
        device_str = self.training_config.get("device")
        device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"Using device: {device}")
        
        # Setup checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            keep_last_n=2,
            wandb_run=self.wandb_run,
        )
        
        # Build model
        model = MinimalMOIRAI(
            d_model=self.model_config["d_model"],
            num_heads=self.model_config["num_heads"],
            num_layers=self.model_config["num_layers"],
            d_ff=self.model_config["d_ff"],
            dropout=self.model_config["dropout"],
            patch_sizes=self.data_config["supported_patch_sizes"],
        ).to(device)
        
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model initialized with {n_params:,} parameters")
        
        # Setup optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=self.training_config["learning_rate"],
            weight_decay=self.training_config.get("weight_decay", 0.01),
        )
        
        # Resume from checkpoint if needed
        start_step = 0
        best_val_loss = float("inf")
        
        if resume_from_checkpoint and self.checkpoint_manager.has_checkpoint():
            logger.info("Loading checkpoint...")
            checkpoint_state = self.checkpoint_manager.load_checkpoint(device=device)
            
            if checkpoint_state:
                model.load_state_dict(checkpoint_state.model_state_dict)
                optimizer.load_state_dict(checkpoint_state.optimizer_state_dict)
                start_step = checkpoint_state.global_step
                best_val_loss = checkpoint_state.best_val_loss
                
                logger.info(
                    f"Resumed from step {start_step}, "
                    f"best_val_loss={best_val_loss:.4f}"
                )
        
        # Watch model with wandb
        if self.wandb_run:
            self.wandb_run.watch(model, log="gradients", log_freq=100)
        
        # Create training dataloader
        train_data_dir = Path(self.data_config["train_data_dir"])
        loader_config = self.data_config["loader_config"]
        
        train_loader = create_train_dataloader(
            data_dir=train_data_dir,
            batch_size=self.data_config["batch_size"],
            num_workers=self.data_config.get("num_workers", 2),
            **loader_config,
        )
        
        # Training loop
        logger.info("\n" + "=" * 80)
        logger.info("STARTING TRAINING")
        logger.info("=" * 80)
        
        num_training_steps = self.training_config["num_training_steps"]
        log_interval = self.training_config["log_interval"]
        max_grad_norm = self.training_config.get("max_grad_norm", 1.0)
        window_size = self.training_config.get("window_size", 100)
        
        # Metrics tracking
        recent_metrics = defaultdict(list)
        global_step = start_step
        
        train_iter = iter(train_loader)
        
        while global_step < num_training_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                # Restart iterator
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            # Training step
            metrics = self._train_step(
                model, batch, optimizer, device, max_grad_norm
            )
            
            if metrics is None:
                continue
            
            global_step += 1
            
            # Track metrics
            recent_metrics["loss"].append(metrics.loss)
            recent_metrics["mse"].append(metrics.mse)
            recent_metrics["mae"].append(metrics.mae)
            recent_metrics["grad_norm"].append(metrics.grad_norm)
            
            # Keep window size
            if len(recent_metrics["loss"]) > window_size:
                for key in recent_metrics:
                    recent_metrics[key] = recent_metrics[key][-window_size:]
            
            # Log to wandb
            if self.wandb_run and global_step % 10 == 0:
                self.wandb_run.log({
                    "train/loss": metrics.loss,
                    "train/mse": metrics.mse,
                    "train/mae": metrics.mae,
                    "train/grad_norm": metrics.grad_norm,
                    "train/pred_std": metrics.pred_std,
                    "train/target_std": metrics.target_std,
                    "global_step": global_step,
                })
            
            # Log progress
            if global_step % log_interval == 0:
                avg_loss = sum(recent_metrics["loss"]) / len(recent_metrics["loss"])
                avg_grad_norm = sum(recent_metrics["grad_norm"]) / len(recent_metrics["grad_norm"])
                
                logger.info(
                    f"Step {global_step:5d}/{num_training_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Grad: {avg_grad_norm:.3f}"
                )
            
            # Save checkpoint periodically
            checkpoint_interval = self.training_config.get("checkpoint_interval", 500)
            if global_step % checkpoint_interval == 0:
                avg_loss = sum(recent_metrics["loss"]) / len(recent_metrics["loss"])
                is_best = avg_loss < best_val_loss
                
                self.checkpoint_manager.save_checkpoint(
                    epoch=global_step // 100,  # Approximate epoch
                    global_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    val_loss=avg_loss,
                    best_val_loss=best_val_loss,
                    model_config={**self.model_config, "n_params": n_params},
                    is_best=is_best,
                )
                
                if is_best:
                    best_val_loss = avg_loss
                    if self.wandb_run:
                        self.wandb_run.summary["best_train_loss"] = best_val_loss
                        self.wandb_run.summary["best_step"] = global_step
        
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        
        # Evaluation phase
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION PHASE")
        logger.info("=" * 80)
        
        eval_datasets = self.data_config.get("eval_datasets", [])
        num_eval_batches = self.data_config.get("num_eval_batches", 10)
        
        eval_results = {}
        for dataset_name in eval_datasets:
            try:
                metrics = self._evaluate_on_dataset(
                    model, dataset_name, device, 
                    batch_size=self.data_config["batch_size"],
                    num_eval_batches=num_eval_batches,
                )
                
                if metrics:
                    eval_results[dataset_name] = metrics
                    
                    # Log to wandb
                    if self.wandb_run:
                        self.wandb_run.log({
                            f"eval/{dataset_name}/loss": metrics["loss"],
                            f"eval/{dataset_name}/mse": metrics["mse"],
                            f"eval/{dataset_name}/mae": metrics["mae"],
                        })
            except Exception as e:
                logger.error(f"Error evaluating on {dataset_name}: {e}")
                logger.info(f"Continuing with next dataset...")
                continue
        
        # Finish wandb run
        if self.wandb_run:
            wandb.finish()
        
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 80)
        
        return {
            "best_train_loss": best_val_loss,
            "final_step": global_step,
            "eval_results": eval_results,
        }

