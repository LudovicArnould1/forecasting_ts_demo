#!/usr/bin/env python
"""
Simple training script with wandb integration.

Usage:
    python train/run_train.py                    # Run training with defaults
    python train/run_train.py --steps 5000       # Override training steps
    python train/run_train.py --online           # Run in online mode (sync to wandb cloud)
    python train/run_train.py --resume           # Resume from checkpoint
    python train/run_train.py --help             # Show all options
"""
import argparse
import logging
from pathlib import Path

from train.pipeline import TrainingPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train time series forecasting model with wandb tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Training arguments
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help="Directory containing config YAML files (default: config)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of training steps (overrides config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    
    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint",
    )
    
    # W&B arguments
    parser.add_argument(
        "--project",
        type=str,
        default="ts-forecasting",
        help="W&B project name (default: ts-forecasting)",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="W&B entity (username/team)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="W&B run name (auto-generated if not specified)",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Run W&B in online mode (default: offline)",
    )
    
    args = parser.parse_args()
    
    # Print configuration
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Config dir: {args.config_dir}")
    logger.info(f"Checkpoint dir: {args.checkpoint_dir}")
    logger.info(f"Resume: {args.resume}")
    logger.info(f"\nW&B Settings:")
    logger.info(f"  Project: {args.project}")
    logger.info(f"  Entity: {args.entity or '(default)'}")
    logger.info(f"  Run name: {args.run_name or '(auto-generated)'}")
    logger.info(f"  Mode: {'ONLINE' if args.online else 'OFFLINE'}")
    
    if args.steps:
        logger.info(f"\nOverrides:")
        logger.info(f"  Training steps: {args.steps}")
    if args.lr:
        logger.info(f"  Learning rate: {args.lr}")
    if args.batch_size:
        logger.info(f"  Batch size: {args.batch_size}")
    
    logger.info("=" * 80)
    
    # Create pipeline
    pipeline = TrainingPipeline(
        config_dir=args.config_dir,
        project=args.project,
        entity=args.entity,
        run_name=args.run_name,
        offline=not args.online,
    )
    
    # Apply overrides
    if args.steps:
        pipeline.training_config["num_training_steps"] = args.steps
    if args.lr:
        pipeline.training_config["learning_rate"] = args.lr
    if args.batch_size:
        pipeline.data_config["batch_size"] = args.batch_size
    
    # Run training
    results = pipeline.train(
        resume_from_checkpoint=args.resume,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    # Print results summary
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Best train loss: {results['best_train_loss']:.4f}")
    logger.info(f"Final step: {results['final_step']}")
    
    if results["eval_results"]:
        logger.info("\nEvaluation Results:")
        for dataset_name, metrics in results["eval_results"].items():
            logger.info(f"  {dataset_name}:")
            logger.info(f"    Loss: {metrics['loss']:.4f}")
            logger.info(f"    MSE:  {metrics['mse']:.4f}")
            logger.info(f"    MAE:  {metrics['mae']:.4f}")
    
    logger.info("=" * 80)
    
    if not args.online:
        logger.info("\nTo sync offline run to W&B cloud:")
        logger.info("  wandb sync wandb/offline-run-<run_id>")


if __name__ == "__main__":
    main()

