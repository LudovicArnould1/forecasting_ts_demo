"""
Reproducibility utilities for experiment tracking and deterministic behavior.

Provides functions to:
- Set seeds for reproducible results
- Capture system and environment information
- Track git provenance
- Log comprehensive metadata to W&B
"""
from __future__ import annotations
import random
import platform
from pathlib import Path
from typing import Dict, Any
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms (may be slower)
    
    Note:
        Deterministic mode ensures full reproducibility but may reduce performance.
        Some operations have no deterministic implementation and will raise errors.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Use deterministic algorithms where possible
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
    
    logger.info(f"Set random seed to {seed} (deterministic={deterministic})")



def get_system_info() -> Dict[str, Any]:
    """
    Capture system and environment information.
    
    Returns:
        Dictionary with Python version, OS, GPU info, and library versions
    """
    system_info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "hostname": platform.node(),
    }
    
    # PyTorch info
    system_info["pytorch_version"] = torch.__version__
    system_info["cuda_available"] = torch.cuda.is_available()
    
    if torch.cuda.is_available():
        system_info["cuda_version"] = torch.version.cuda
        system_info["cudnn_version"] = torch.backends.cudnn.version()
        system_info["gpu_count"] = torch.cuda.device_count()
        system_info["gpu_names"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        
        # Memory info for first GPU
        if torch.cuda.device_count() > 0:
            mem = torch.cuda.get_device_properties(0).total_memory
            system_info["gpu_memory_gb"] = mem / (1024**3)
    
    # NumPy version
    system_info["numpy_version"] = np.__version__
    
    return system_info


def get_package_versions() -> Dict[str, str]:
    """
    Get versions of key packages from pyproject.toml environment.
    
    Returns:
        Dictionary mapping package names to versions
    """
    versions = {}
    
    try:
        import scanpy
        versions["scanpy"] = scanpy.__version__
    except Exception:
        pass
    
    try:
        import sklearn
        versions["scikit-learn"] = sklearn.__version__
    except Exception:
        pass
    
    try:
        import anndata
        versions["anndata"] = anndata.__version__
    except Exception:
        pass
    
    try:
        import wandb
        versions["wandb"] = wandb.__version__
    except Exception:
        pass
    
    return versions


def get_environment_info() -> Dict[str, Any]:
    """
    Comprehensive environment information for reproducibility.
    
    Combines git, system, and package information into a single dict
    suitable for logging to W&B.
    
    Returns:
        Dictionary with all environment metadata
    """
    env_info = {
        "system": get_system_info(),
        "packages": get_package_versions(),
        "cwd": str(Path.cwd()),
    }
    
    return env_info


def log_environment_to_wandb(wandb_run: Any) -> None:
    """
    Log comprehensive environment information to W&B run.
    
    Args:
        wandb_run: Active wandb run object
    """
    if wandb_run is None:
        logger.warning("No active W&B run; skipping environment logging")
        return
    
    env_info = get_environment_info()
    
    # Log system info
    wandb_run.config.update(
        {f"system/{k}": v for k, v in env_info["system"].items()},
        allow_val_change=True,
    )
    
    # Log package versions
    wandb_run.config.update(
        {f"packages/{k}": v for k, v in env_info["packages"].items()},
        allow_val_change=True,
    )
    
    logger.info("Logged environment info to W&B")

