"""Preprocessing functions for time series data."""

import logging
from typing import Any

import numpy as np
import torch

from data_processing.utils import get_patch_size_for_freq

logger = logging.getLogger(__name__)


class TimeSeriesPreprocessor:
    """Preprocessor for time series data.
    
    Handles:
    - Patching: Breaking time series into non-overlapping patches
    - No normalization (for now, will add instance norm later)
    
    Args:
        max_patch_length: Maximum number of patches to use (default: 128)
        
    Examples:
        >>> preprocessor = TimeSeriesPreprocessor(max_patch_length=128)
        >>> sample = {"target": np.array([[1,2,3,4,5,6]]), "freq": "H"}
        >>> result = preprocessor.process(sample)
        >>> print(result["patches"].shape)  # (1, num_patches, patch_size)
    """
    
    def __init__(self, max_patch_length: int = 128):
        """Initialize preprocessor.
        
        Args:
            max_patch_length: Maximum number of patches to keep
        """
        self.max_patch_length = max_patch_length
    
    def process(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Process a single time series sample.
        
        Args:
            sample: Dictionary containing at minimum:
                - target: numpy array of shape (num_variates, length)
                - freq: frequency string
                
        Returns:
            Dictionary with added fields:
            - patches: numpy array of shape (num_variates, num_patches, patch_size)
            - patch_size: int, size of each patch
            - num_patches: int, number of patches
            - original_length: int, original series length
        """
        target = sample["target"]  # Shape: (num_variates, length)
        freq = sample["freq"]
        
        # Get patch size for this frequency
        patch_size = get_patch_size_for_freq(freq)
        
        # Apply patching
        patches, num_patches = self._apply_patching(target, patch_size)
        
        # Return sample with added preprocessing info
        result = sample.copy()
        result.update({
            "patches": patches,  # (num_variates, num_patches, patch_size)
            "patch_size": patch_size,
            "num_patches": num_patches,
            "original_length": target.shape[1],
        })
        
        return result
    
    def _apply_patching(
        self, target: np.ndarray, patch_size: int
    ) -> tuple[np.ndarray, int]:
        """Break time series into non-overlapping patches.
        
        Args:
            target: Array of shape (num_variates, length)
            patch_size: Size of each patch
            
        Returns:
            Tuple of (patches, num_patches):
            - patches: Array of shape (num_variates, num_patches, patch_size)
            - num_patches: Number of patches created
            
        Examples:
            >>> target = np.array([[1, 2, 3, 4, 5, 6]])  # (1, 6)
            >>> patches, n = _apply_patching(target, patch_size=2)
            >>> patches.shape
            (1, 3, 2)  # 3 patches of size 2
            >>> patches
            array([[[1, 2], [3, 4], [5, 6]]])
        """
        num_variates, length = target.shape
        
        # Calculate number of complete patches
        num_patches = length // patch_size
        
        if num_patches == 0:
            raise ValueError(
                f"Series length {length} is shorter than patch size {patch_size}"
            )
        
        # Truncate to fit exact number of patches (no overlap)
        truncated_length = num_patches * patch_size
        target_truncated = target[:, :truncated_length]
        
        # Reshape to (num_variates, num_patches, patch_size)
        patches = target_truncated.reshape(num_variates, num_patches, patch_size)
        
        return patches, num_patches
    
    def to_tensor(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Convert numpy arrays to PyTorch tensors.
        
        Args:
            sample: Dictionary containing preprocessed data
            
        Returns:
            Same dictionary with patches converted to tensors
        """
        result = sample.copy()
        # Convert various array fields to tensors
        array_fields = [
            "patches", "target", "context_patches", "target_patches"
        ]
        for field in array_fields:
            if field in result and isinstance(result[field], np.ndarray):
                result[field] = torch.from_numpy(result[field])
        return result
