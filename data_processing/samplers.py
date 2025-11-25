"""Sampling strategies for time series windows."""

import logging
import random
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class RandomWindowSampler:
    """Random window sampler for training.
    
    Samples random windows from time series, following Moirai's approach:
    - Maximum window length of 128 patches
    - Randomly sample context_length and prediction_length within this window
    - context_length + prediction_length <= max_window_length
    
    Args:
        max_window_length: Maximum total window length in patches (default: 128)
        min_context_length: Minimum context length in patches (default: 16)
        max_context_length: Maximum context length in patches (default: 96)
        min_prediction_length: Minimum prediction length in patches (default: 4)
        max_prediction_length: Maximum prediction length in patches (default: 32)
        
    Examples:
        >>> sampler = RandomWindowSampler()
        >>> sample = {"patches": np.random.randn(1, 200, 128)}
        >>> result = sampler.sample(sample)
        >>> print(result.keys())
        dict_keys(['context_patches', 'target_patches', 'context_length', ...])
    """
    
    def __init__(
        self,
        max_window_length: int = 128,
        min_context_length: int = 16,
        max_context_length: int = 96,
        min_prediction_length: int = 4,
        max_prediction_length: int = 32,
    ):
        """Initialize random window sampler."""
        self.max_window_length = max_window_length
        self.min_context_length = min_context_length
        self.max_context_length = max_context_length
        self.min_prediction_length = min_prediction_length
        self.max_prediction_length = max_prediction_length
        
        # Validate parameters
        if min_context_length + min_prediction_length > max_window_length:
            raise ValueError(
                f"min_context_length ({min_context_length}) + "
                f"min_prediction_length ({min_prediction_length}) > "
                f"max_window_length ({max_window_length})"
            )
    
    def sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Sample a random window from the time series.
        
        Args:
            sample: Dictionary containing:
                - patches: array of shape (num_variates, num_patches, patch_size)
                - num_patches: number of patches
                - other metadata fields
                
        Returns:
            Dictionary with:
            - context_patches: array of shape (num_variates, context_length, patch_size)
            - target_patches: array of shape (num_variates, prediction_length, patch_size)
            - context_length: number of context patches
            - prediction_length: number of prediction patches
            - original metadata
        """
        patches = sample["patches"]
        num_variates, num_patches, patch_size = patches.shape
        
        # Step 1: Randomly sample context_length and prediction_length
        # such that their sum <= max_window_length
        
        # First, determine feasible range for context_length
        max_feasible_context = min(
            self.max_context_length,
            self.max_window_length - self.min_prediction_length,
            num_patches - self.min_prediction_length
        )
        
        if max_feasible_context < self.min_context_length:
            raise ValueError(
                f"Cannot sample window: max_feasible_context={max_feasible_context} < "
                f"min_context_length={self.min_context_length}"
            )
        
        context_length = random.randint(self.min_context_length, max_feasible_context)
        
        # Then determine feasible range for prediction_length
        max_pred_len = min(
            self.max_prediction_length,
            self.max_window_length - context_length,
            num_patches - context_length
        )
        
        if max_pred_len < self.min_prediction_length:
            raise ValueError(
                f"Cannot sample window: max_pred_len={max_pred_len} < "
                f"min_prediction_length={self.min_prediction_length}"
            )
        
        prediction_length = random.randint(self.min_prediction_length, max_pred_len)
        
        total_window = context_length + prediction_length
        
        # Step 2: Randomly sample starting position
        if num_patches < total_window:
            raise ValueError(
                f"Series has {num_patches} patches, but need {total_window} "
                f"(context={context_length} + prediction={prediction_length})"
            )
        
        max_start = num_patches - total_window
        start_idx = random.randint(0, max_start)
        
        # Step 3: Extract context and target windows
        context_patches = patches[:, start_idx:start_idx + context_length, :]
        target_patches = patches[
            :, start_idx + context_length:start_idx + total_window, :
        ]
        
        # Create result dictionary
        result = sample.copy()
        result.update({
            "context_patches": context_patches,
            "target_patches": target_patches,
            "context_length": context_length,
            "prediction_length": prediction_length,
            "window_start_idx": start_idx,
        })
        
        return result


class SequentialWindowSampler:
    """Sequential window sampler for validation/testing.
    
    Generates non-overlapping windows sequentially from time series.
    Uses fixed context and prediction lengths.
    
    Args:
        context_length: Fixed context length in patches (default: 64)
        prediction_length: Fixed prediction length in patches (default: 16)
        
    Examples:
        >>> sampler = SequentialWindowSampler(context_length=64, prediction_length=16)
        >>> sample = {"patches": np.random.randn(1, 200, 128)}
        >>> windows = list(sampler.sample_all(sample))
        >>> len(windows)
        2  # floor((200 - 64) / 16)
    """
    
    def __init__(self, context_length: int = 64, prediction_length: int = 16):
        """Initialize sequential window sampler."""
        self.context_length = context_length
        self.prediction_length = prediction_length
    
    def sample_all(self, sample: dict[str, Any]) -> list[dict[str, Any]]:
        """Generate all sequential windows from a time series.
        
        Args:
            sample: Dictionary containing:
                - patches: array of shape (num_variates, num_patches, patch_size)
                - num_patches: number of patches
                - other metadata fields
                
        Returns:
            List of dictionaries, each with:
            - context_patches: array of shape (num_variates, context_length, patch_size)
            - target_patches: array of shape (num_variates, prediction_length, patch_size)
            - context_length: number of context patches
            - prediction_length: number of prediction patches
            - window_idx: index of this window
            - original metadata
        """
        patches = sample["patches"]
        num_variates, num_patches, patch_size = patches.shape
        
        windows = []
        window_idx = 0
        
        # Generate windows with stride = prediction_length (no overlap)
        start_idx = 0
        while start_idx + self.context_length + self.prediction_length <= num_patches:
            context_patches = patches[
                :, start_idx:start_idx + self.context_length, :
            ]
            target_patches = patches[
                :,
                start_idx + self.context_length:
                start_idx + self.context_length + self.prediction_length,
                :
            ]
            
            result = sample.copy()
            result.update({
                "context_patches": context_patches,
                "target_patches": target_patches,
                "context_length": self.context_length,
                "prediction_length": self.prediction_length,
                "window_start_idx": start_idx,
                "window_idx": window_idx,
            })
            
            windows.append(result)
            window_idx += 1
            
            # Move to next window (stride = prediction_length)
            start_idx += self.prediction_length
        
        return windows
