"""
Checkpoint management for training with resumption support.

This module provides checkpoint saving, loading, cleanup, and W&B artifact integration
for robust training with failure recovery and disk space management.
"""
from __future__ import annotations
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import wandb

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    """
    Complete training state for exact resumption.
    
    Attributes:
        epoch: Current epoch number
        global_step: Global training step
        model_state_dict: Model weights
        optimizer_state_dict: Optimizer state
        optimizer_other_state_dict: Secondary optimizer state (if used)
        best_val_loss: Best validation loss so far
        wandb_run_id: W&B run ID for resumption
        rng_state: Random number generator states
        model_config: Model configuration dict
    """
    epoch: int
    global_step: int
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]
    optimizer_other_state_dict: Optional[Dict[str, Any]]
    best_val_loss: float
    wandb_run_id: Optional[str]
    rng_state: Dict[str, Any]
    model_config: Dict[str, Any]


class CheckpointManager:
    """
    Manages checkpoints with automatic cleanup and W&B integration.
    
    Features:
    - Save complete training state (model, optimizer, RNG, W&B run ID)
    - Load and restore for exact resumption
    - Keep only best + last N checkpoints
    - Upload checkpoints as W&B artifacts with versioning
    """
    
    def __init__(
        self,
        checkpoint_dir: Path | str,
        keep_last_n: int = 2,
        wandb_run: Optional[wandb.sdk.wandb_run.Run] = None,
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            keep_last_n: Number of recent checkpoints to keep (in addition to best)
            wandb_run: Active W&B run for artifact upload
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.wandb_run = wandb_run
        
        # Paths
        self.best_checkpoint_path = self.checkpoint_dir / "best_model.pt"
        self.resume_info_path = self.checkpoint_dir / "resume_info.json"
        
    def save_checkpoint(
        self,
        epoch: int,
        global_step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        val_loss: float,
        best_val_loss: float,
        model_config: Dict[str, Any],
        is_best: bool = False,
        optimizer_other: Optional[torch.optim.Optimizer] = None,
    ) -> Path:
        """
        Save complete training checkpoint.
        
        Args:
            epoch: Current epoch
            global_step: Global training step
            model: Model to save
            optimizer: Primary optimizer
            val_loss: Current validation loss
            best_val_loss: Best validation loss so far
            model_config: Model configuration dict
            is_best: Whether this is the best checkpoint
            optimizer_other: Secondary optimizer (optional)
            
        Returns:
            Path to saved checkpoint
        """
        # Capture RNG states
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        
        # Prepare checkpoint
        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "optimizer_other_state_dict": optimizer_other.state_dict() if optimizer_other else None,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "wandb_run_id": self.wandb_run.id if self.wandb_run else None,
            "rng_state": rng_state,
            "model_config": model_config,
        }
        
        # Determine save path
        if is_best:
            save_path = self.best_checkpoint_path
            logger.info(f"Saving best checkpoint at epoch {epoch} (val_loss={val_loss:.4f})")
        else:
            save_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            logger.info(f"Saving checkpoint at epoch {epoch}")
        
        # Save checkpoint
        torch.save(checkpoint, save_path)
        
        # Update resume info
        self._update_resume_info(
            epoch=epoch,
            global_step=global_step,
            best_val_loss=best_val_loss,
            wandb_run_id=checkpoint["wandb_run_id"],
            checkpoint_path=str(save_path),
        )
        
        # Cleanup old checkpoints (keep best + last N)
        if not is_best:
            self._cleanup_old_checkpoints()
        
        # Upload to W&B as artifact (if enabled)
        if self.wandb_run and is_best:
            self._upload_to_wandb(save_path, epoch, val_loss)
        
        return save_path
    
    def load_checkpoint(
        self,
        checkpoint_path: Optional[Path | str] = None,
        device: Optional[torch.device] = None,
    ) -> Optional[CheckpointState]:
        """
        Load checkpoint for resumption.
        
        Args:
            checkpoint_path: Path to checkpoint (defaults to best if not specified)
            device: Device to load tensors to
            
        Returns:
            CheckpointState if checkpoint exists, None otherwise
        """
        if checkpoint_path is None:
            # Try to load from resume info
            if self.resume_info_path.exists():
                with open(self.resume_info_path, "r") as f:
                    resume_info = json.load(f)
                checkpoint_path = resume_info.get("latest_checkpoint")
            
            # Fall back to best checkpoint
            if checkpoint_path is None or not Path(checkpoint_path).exists():
                checkpoint_path = self.best_checkpoint_path
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            logger.warning(f"No checkpoint found at {checkpoint_path}")
            return None
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
        
        # Restore RNG states
        if "rng_state" in checkpoint:
            rng_state = checkpoint["rng_state"]
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch"])
            if rng_state["torch_cuda"] and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
            logger.info("Restored RNG states for exact resumption")
        
        # Create checkpoint state
        state = CheckpointState(
            epoch=checkpoint["epoch"],
            global_step=checkpoint["global_step"],
            model_state_dict=checkpoint["model_state_dict"],
            optimizer_state_dict=checkpoint["optimizer_state_dict"],
            optimizer_other_state_dict=checkpoint.get("optimizer_other_state_dict"),
            best_val_loss=checkpoint.get("best_val_loss", float("inf")),
            wandb_run_id=checkpoint.get("wandb_run_id"),
            rng_state=checkpoint.get("rng_state", {}),
            model_config=checkpoint.get("model_config", {}),
        )
        
        logger.info(
            f"Loaded checkpoint: epoch={state.epoch}, "
            f"step={state.global_step}, "
            f"best_val_loss={state.best_val_loss:.4f}"
        )
        
        return state
    
    def has_checkpoint(self) -> bool:
        """
        Check if any checkpoint exists for resumption.
        
        Returns:
            True if checkpoint exists, False otherwise
        """
        # Check resume info first
        if self.resume_info_path.exists():
            with open(self.resume_info_path, "r") as f:
                resume_info = json.load(f)
            checkpoint_path = resume_info.get("latest_checkpoint")
            if checkpoint_path and Path(checkpoint_path).exists():
                return True
        
        # Check best checkpoint
        return self.best_checkpoint_path.exists()
    
    def _update_resume_info(
        self,
        epoch: int,
        global_step: int,
        best_val_loss: float,
        wandb_run_id: Optional[str],
        checkpoint_path: str,
    ) -> None:
        """
        Update resume info JSON file.
        
        Args:
            epoch: Current epoch
            global_step: Global step
            best_val_loss: Best validation loss
            wandb_run_id: W&B run ID
            checkpoint_path: Path to checkpoint
        """
        resume_info = {
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "wandb_run_id": wandb_run_id,
            "latest_checkpoint": checkpoint_path,
        }
        
        with open(self.resume_info_path, "w") as f:
            json.dump(resume_info, f, indent=2)
    
    def _cleanup_old_checkpoints(self) -> None:
        """
        Keep only best + last N checkpoints, delete older ones.
        """
        # Get all checkpoint files (excluding best)
        checkpoint_files = sorted(
            [f for f in self.checkpoint_dir.glob("checkpoint_epoch_*.pt")],
            key=lambda x: x.stat().st_mtime,
            reverse=True,  # Most recent first
        )
        
        # Keep last N, delete rest
        to_delete = checkpoint_files[self.keep_last_n:]
        
        for checkpoint_file in to_delete:
            logger.info(f"Deleting old checkpoint: {checkpoint_file.name}")
            checkpoint_file.unlink()
    
    def _upload_to_wandb(
        self,
        checkpoint_path: Path,
        epoch: int,
        val_loss: float,
    ) -> None:
        """
        Upload checkpoint as W&B artifact.
        
        Args:
            checkpoint_path: Path to checkpoint file
            epoch: Epoch number
            val_loss: Validation loss
        """
        if not self.wandb_run:
            return
        
        try:
            artifact = wandb.Artifact(
                name="model_checkpoint",
                type="model",
                description=f"Model checkpoint at epoch {epoch}",
                metadata={
                    "epoch": epoch,
                    "val_loss": float(val_loss),
                    "checkpoint_type": "best",
                },
            )
            
            artifact.add_file(str(checkpoint_path))
            
            # Log artifact with automatic versioning
            self.wandb_run.log_artifact(artifact, aliases=["best", "latest"])
            
            logger.info(f"Uploaded checkpoint to W&B: {artifact.name}")
            
        except Exception as e:
            logger.warning(f"Failed to upload checkpoint to W&B: {e}")
    
    def get_resume_run_id(self) -> Optional[str]:
        """
        Get W&B run ID for resumption.
        
        Returns:
            W&B run ID if available, None otherwise
        """
        if self.resume_info_path.exists():
            with open(self.resume_info_path, "r") as f:
                resume_info = json.load(f)
            return resume_info.get("wandb_run_id")
        return None

