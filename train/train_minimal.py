"""Minimal training script to test if the model can learn from time series data.

This script trains on 2 datasets for a few steps to verify the model is working.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW

from data_processing.dataloader import create_train_dataloader
from model.moirai import MinimalMOIRAI, mse_loss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def train_step(
    model: nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_details: bool = False,
) -> dict[str, float] | None:
    """Execute one training step.
    
    Args:
        model: The forecasting model
        batch: Batch from dataloader
        optimizer: Optimizer
        device: Device to train on
        log_details: If True, log detailed diagnostics
        
    Returns:
        Dictionary with loss and metrics, or None if batch contains NaN
    """
    model.train()
    
    # Move data to device
    context = batch["context_patches"].to(device)
    target = batch["target_patches"].to(device)
    context_mask = batch["context_mask"].to(device)
    target_mask = batch["target_mask"].to(device)
    
    # Check for NaN in data
    if torch.isnan(context).any() or torch.isnan(target).any():
        logger.warning("Batch contains NaN values, skipping...")
        return None
    
    # Get actual prediction length from target shape
    batch_size, num_variates, pred_len, patch_size = target.shape
    
    # Forward pass - model normalizes both context and target using same RevIN stats
    # Returns (predictions, normalized_target), both in normalized space
    predictions, target_norm = model(
        context=context,
        context_mask=context_mask,
        prediction_length=pred_len,
        target=target,
    )
    
    # Log details if requested
    if log_details:
        logger.info("\n" + "-"*80)
        logger.info("TRAIN STEP DETAILS (After Forward Pass)")
        logger.info("-"*80)
        logger.info(f"Target (original) - min: {target.min():.4f}, max: {target.max():.4f}, mean: {target.mean():.4f}, std: {target.std():.4f}")
        logger.info(f"Predictions (normalized) - min: {predictions.min():.4f}, max: {predictions.max():.4f}, mean: {predictions.mean():.4f}, std: {predictions.std():.4f}")
        logger.info(f"Target (normalized) - min: {target_norm.min():.4f}, max: {target_norm.max():.4f}, mean: {target_norm.mean():.4f}, std: {target_norm.std():.4f}")
        logger.info("-"*80 + "\n")
    
    # Compute loss in NORMALIZED space (scale-invariant)
    loss = mse_loss(predictions, target_norm, target_mask)
    
    # Check if loss is valid
    if torch.isnan(loss) or torch.isinf(loss):
        logger.warning("Loss is NaN or Inf, skipping batch...")
        return None
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping and diagnostics - load max_norm from config if needed elsewhere
    grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    # Compute additional metrics and diagnostics (in normalized space)
    with torch.no_grad():
        mse = ((predictions - target_norm) ** 2 * target_mask.unsqueeze(1).unsqueeze(-1)).sum() / target_mask.sum()
        mae = ((predictions - target_norm).abs() * target_mask.unsqueeze(1).unsqueeze(-1)).sum() / target_mask.sum()
        
        # Diagnostic statistics
        pred_std = predictions.std().item()
        target_std = target.std().item()
    
    return {
        "loss": loss.item(),
        "mse": mse.item(),
        "mae": mae.item(),
        "grad_norm": grad_norm_before_clip.item(),
        "pred_std": pred_std,
        "target_std": target_std,
    }


@torch.no_grad()
def eval_step(
    model: nn.Module,
    batch: dict,
    device: torch.device,
) -> dict[str, float] | None:
    """Execute one evaluation step.
    
    Args:
        model: The forecasting model
        batch: Batch from dataloader
        device: Device to evaluate on
        
    Returns:
        Dictionary with loss and metrics, or None if batch contains NaN
    """
    model.eval()
    
    # Move data to device
    context = batch["context_patches"].to(device)
    target = batch["target_patches"].to(device)
    context_mask = batch["context_mask"].to(device)
    target_mask = batch["target_mask"].to(device)
    
    # Check for NaN in data
    if torch.isnan(context).any() or torch.isnan(target).any():
        return None
    
    # Get actual prediction length from target shape
    batch_size, num_variates, pred_len, patch_size = target.shape
    
    # Forward pass - model normalizes both context and target using same RevIN stats
    predictions, target_norm = model(
        context=context,
        context_mask=context_mask,
        prediction_length=pred_len,
        target=target,
    )
    
    # Compute loss in NORMALIZED space (scale-invariant)
    loss = mse_loss(predictions, target_norm, target_mask)
    
    # Check if loss is valid
    if torch.isnan(loss) or torch.isinf(loss):
        return None
    
    # Compute metrics (in normalized space)
    mse = ((predictions - target_norm) ** 2 * target_mask.unsqueeze(1).unsqueeze(-1)).sum() / target_mask.sum()
    mae = ((predictions - target_norm).abs() * target_mask.unsqueeze(1).unsqueeze(-1)).sum() / target_mask.sum()
    
    return {
        "loss": loss.item(),
        "mse": mse.item(),
        "mae": mae.item(),
    }


def evaluate_on_dataset(
    model: nn.Module,
    dataset_name: str,
    device: torch.device,
    batch_size: int = 64,
    num_eval_batches: int = 10,
) -> dict[str, float] | None:
    """Evaluate model on a specific dataset.
    
    Args:
        model: Trained model
        dataset_name: Name of dataset to evaluate on
        device: Device to evaluate on
        batch_size: Batch size for evaluation
        num_eval_batches: Number of batches to evaluate
        
    Returns:
        Dictionary with average metrics, or None if evaluation failed
    """
    logger.info(f"\nEvaluating on dataset: {dataset_name}")
    logger.info("-" * 60)
    
    data_dir = Path("data/training") / dataset_name
    
    if not data_dir.exists():
        logger.warning(f"Dataset directory not found: {data_dir}")
        return None
    
    # Create dataloader with appropriate parameters (use slightly longer min_length for eval)
    loader_config = {
        "max_window_length": 64,
        "min_context_length": 16,
        "max_context_length": 32,
        "min_prediction_length": 4,
        "max_prediction_length": 16,
        "min_length": 512,  # Longer for evaluation
    }
    
    try:
        eval_loader = create_train_dataloader(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=2,
            **loader_config,
        )
    except Exception as e:
        logger.error(f"Failed to create dataloader: {e}")
        return None
    
    # Evaluate on multiple batches
    eval_metrics = []
    eval_attempts = 0
    
    for batch in eval_loader:
        if len(eval_metrics) >= num_eval_batches:
            break
        eval_attempts += 1
        if eval_attempts > num_eval_batches * 3:  # Don't try too many times
            break
        
        metrics = eval_step(model, batch, device)
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
    }
    
    logger.info(f"Results on {dataset_name}:")
    logger.info(f"  Avg Loss: {avg_metrics['loss']:.4f}")
    logger.info(f"  Avg MSE:  {avg_metrics['mse']:.4f}")
    logger.info(f"  Avg MAE:  {avg_metrics['mae']:.4f}")
    logger.info(f"  Evaluated on {len(eval_metrics)} batches")
    
    return avg_metrics


def main():
    """Train on train_sample and evaluate on specific datasets."""
    
    # Load configuration files
    config_dir = Path(__file__).parent.parent / "config"
    
    with open(config_dir / "data.yaml", "r") as f:
        data_config = yaml.safe_load(f)
    
    with open(config_dir / "model.yaml", "r") as f:
        model_config = yaml.safe_load(f)
    
    with open(config_dir / "training.yaml", "r") as f:
        training_config = yaml.safe_load(f)
    
    logger.info("Loaded configuration files successfully")
    
    # Device configuration
    device = torch.device(training_config["device"] or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")
    
    # Extract configurations
    train_data_dir = Path(data_config["train_data_dir"])
    eval_datasets = data_config["eval_datasets"]
    batch_size = data_config["batch_size"]
    num_training_steps = training_config["num_training_steps"]
    learning_rate = training_config["learning_rate"]
    log_interval = training_config["log_interval"]
    num_eval_batches = data_config["num_eval_batches"]
    
    # ============================================================================
    # TRAINING PHASE
    # ============================================================================
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING PHASE")
    logger.info("="*80)
    logger.info(f"Training on: {train_data_dir}")
    
    if not train_data_dir.exists():
        logger.error(f"Training directory not found: {train_data_dir}")
        return
    
    # Create training dataloader
    loader_config = data_config["loader_config"]
    
    try:
        train_loader = create_train_dataloader(
            data_dir=train_data_dir,
            batch_size=batch_size,
            num_workers=data_config["num_workers"],
            **loader_config,
        )
    except Exception as e:
        logger.error(f"Failed to create training dataloader: {e}")
        return
    
    # Get supported patch sizes from data config
    supported_patch_sizes = data_config["supported_patch_sizes"]
    logger.info(f"Model will support patch sizes: {supported_patch_sizes}")
    
    # Verify we can get a batch
    try:
        first_batch = next(iter(train_loader))
        patch_size = first_batch["context_patches"].shape[-1]
        logger.info(f"First batch has patch_size: {patch_size}")
        assert patch_size in supported_patch_sizes, f"Unexpected patch_size: {patch_size}"
    except Exception as e:
        logger.error(f"Failed to get first batch: {e}")
        return
    
    # Initialize model with all supported patch sizes
    model_config["patch_sizes"] = supported_patch_sizes
    model = MinimalMOIRAI(**model_config).to(device)
    
    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Training for {num_training_steps} steps...")
    
    # Training loop with windowed averaging
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    valid_steps = 0
    skipped_batches = 0
    
    # For moving average
    window_size = training_config["window_size"]
    recent_losses = []
    recent_mses = []
    recent_maes = []
    
    # Store initial parameters for first step diagnostics
    initial_params = None
    
    for step, batch in enumerate(train_loader):
        if valid_steps >= num_training_steps:
            break
        
        # Add detailed diagnostics for first batch
        if valid_steps == 0:
            logger.info("\n" + "="*80)
            logger.info("FIRST BATCH DIAGNOSTICS")
            logger.info("="*80)
            context = batch["context_patches"]
            target = batch["target_patches"]
            logger.info(f"Context shape: {context.shape}")
            logger.info(f"Target shape: {target.shape}")
            logger.info(f"Context - min: {context.min():.4f}, max: {context.max():.4f}, mean: {context.mean():.4f}, std: {context.std():.4f}")
            logger.info(f"Target - min: {target.min():.4f}, max: {target.max():.4f}, mean: {target.mean():.4f}, std: {target.std():.4f}")
            logger.info(f"Context has NaN: {torch.isnan(context).any()}")
            logger.info(f"Target has NaN: {torch.isnan(target).any()}")
            logger.info(f"Patch size: {context.shape[-1]}")
            logger.info("="*80 + "\n")
            
            # Store initial parameters
            initial_params = {name: param.clone().detach() for name, param in model.named_parameters()}
        
        # Log details for first step
        metrics = train_step(model, batch, optimizer, device, log_details=(valid_steps == 0))
        
        if metrics is None:
            skipped_batches += 1
            continue
        
        valid_steps += 1
        total_loss += metrics["loss"]
        total_mse += metrics["mse"]
        total_mae += metrics["mae"]
        
        # Check parameter updates after first step
        if valid_steps == 1 and initial_params is not None:
            logger.info("\n" + "="*80)
            logger.info("PARAMETER UPDATE CHECK (After first step)")
            logger.info("="*80)
            max_param_change = 0.0
            max_param_name = None
            for name, param in model.named_parameters():
                if name in initial_params:
                    change = (param - initial_params[name]).abs().max().item()
                    if change > max_param_change:
                        max_param_change = change
                        max_param_name = name
            logger.info(f"Max parameter change: {max_param_change:.6e} (in {max_param_name})")
            if max_param_change < 1e-6:
                logger.warning("⚠️  Parameters barely changed! Possible learning issue.")
            else:
                logger.info("✓ Parameters are being updated")
            logger.info("="*80 + "\n")
        
        # Track recent metrics for moving average
        recent_losses.append(metrics["loss"])
        recent_mses.append(metrics["mse"])
        recent_maes.append(metrics["mae"])
        if len(recent_losses) > window_size:
            recent_losses.pop(0)
            recent_mses.pop(0)
            recent_maes.pop(0)
        
        if valid_steps % log_interval == 0:
            avg_loss = total_loss / valid_steps
            avg_mse = total_mse / valid_steps
            avg_mae = total_mae / valid_steps
            
            # Compute moving averages
            moving_avg_loss = sum(recent_losses) / len(recent_losses)
            moving_avg_mse = sum(recent_mses) / len(recent_mses)
            moving_avg_mae = sum(recent_maes) / len(recent_maes)
            
            logger.info(
                f"Step {valid_steps}/{num_training_steps}: "
                f"Loss={metrics['loss']:.4f}, MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}"
            )
            logger.info(
                f"  Moving avg (last {len(recent_losses)}): "
                f"Loss={moving_avg_loss:.4f}, MSE={moving_avg_mse:.4f}, MAE={moving_avg_mae:.4f}"
            )
            logger.info(
                f"  Cumulative avg: Loss={avg_loss:.4f}, MSE={avg_mse:.4f}, MAE={avg_mae:.4f}"
            )
            logger.info(
                f"  Grad norm: {metrics['grad_norm']:.4f}, "
                f"Pred std: {metrics['pred_std']:.4f}, "
                f"Target std: {metrics['target_std']:.4f}"
            )
    
    if skipped_batches > 0:
        logger.warning(f"Skipped {skipped_batches} batches due to NaN values")
    
    # Final training statistics
    if valid_steps == 0:
        logger.error("No valid batches during training, all contained NaN!")
        return
        
    avg_loss = total_loss / valid_steps
    avg_mse = total_mse / valid_steps
    avg_mae = total_mae / valid_steps
    
    logger.info("\n" + "-"*80)
    logger.info("Final training statistics:")
    logger.info(f"  Average Loss: {avg_loss:.4f}")
    logger.info(f"  Average MSE:  {avg_mse:.4f}")
    logger.info(f"  Average MAE:  {avg_mae:.4f}")
    logger.info(f"  Completed {valid_steps} training steps")
    logger.info("-"*80)
    
    # ============================================================================
    # EVALUATION PHASE
    # ============================================================================
    
    logger.info("\n" + "="*80)
    logger.info("EVALUATION PHASE")
    logger.info("="*80)
    
    model.eval()
    eval_results = {}
    
    for dataset_name in eval_datasets:
        result = evaluate_on_dataset(
            model=model,
            dataset_name=dataset_name,
            device=device,
            batch_size=batch_size,
            num_eval_batches=num_eval_batches,
        )
        if result is not None:
            eval_results[dataset_name] = result
    
    # Summary of evaluation results
    logger.info("\n" + "="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*80)
    
    if eval_results:
        for dataset_name, metrics in eval_results.items():
            logger.info(f"\n{dataset_name}:")
            logger.info(f"  Loss: {metrics['loss']:.4f}")
            logger.info(f"  MSE:  {metrics['mse']:.4f}")
            logger.info(f"  MAE:  {metrics['mae']:.4f}")
    else:
        logger.warning("No successful evaluations!")
    
    logger.info("\n" + "="*80)
    logger.info("Training and evaluation completed successfully!")
    logger.info("="*80)


if __name__ == "__main__":
    main()

