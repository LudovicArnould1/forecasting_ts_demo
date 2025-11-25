"""Tests for preprocessing module."""

import numpy as np
import pytest
import torch

from data_processing.preprocessing import TimeSeriesPreprocessor


class TestTimeSeriesPreprocessor:
    """Test preprocessing functionality."""
    
    def test_basic_patching(self):
        """Test basic patching functionality."""
        preprocessor = TimeSeriesPreprocessor()
        
        # Create univariate series with enough timesteps for daily patch size (128)
        target = np.arange(256, dtype=np.float32).reshape(1, 256)
        sample = {"target": target, "freq": "D"}
        
        result = preprocessor.process(sample)
        
        assert "patches" in result
        assert "patch_size" in result
        assert "num_patches" in result
        assert result["patch_size"] == 128  # Daily frequency
        assert result["num_patches"] == 2  # 256 / 128 = 2
        
    def test_patching_shape(self):
        """Test that patching produces correct shapes."""
        preprocessor = TimeSeriesPreprocessor()
        
        # Create series with 256 timesteps (2 patches of size 128)
        target = np.arange(256, dtype=np.float32).reshape(1, 256)
        sample = {"target": target, "freq": "D"}
        
        result = preprocessor.process(sample)
        patches = result["patches"]
        
        # Should be (1 variate, 2 patches, 128 patch_size)
        assert patches.shape == (1, 2, 128)
        assert result["num_patches"] == 2
    
    def test_multivariate_patching(self):
        """Test patching with multivariate time series."""
        preprocessor = TimeSeriesPreprocessor()
        
        # 3 variates, 384 timesteps -> 3 patches of 128
        target = np.arange(3 * 384, dtype=np.float32).reshape(3, 384)
        sample = {"target": target, "freq": "H"}
        
        result = preprocessor.process(sample)
        patches = result["patches"]
        
        # Should be (3 variates, 3 patches, 128 patch_size)
        assert patches.shape == (3, 3, 128)
    
    def test_no_overlap(self):
        """Test that patches don't overlap."""
        preprocessor = TimeSeriesPreprocessor()
        
        # Create series [0, 1, 2, 3, 4, 5, 6, 7]
        target = np.arange(8, dtype=np.float32).reshape(1, 8)
        sample = {"target": target, "freq": "Y"}  # Patch size = 8
        
        result = preprocessor.process(sample)
        patches = result["patches"]
        
        # Should have 1 patch: [0,1,2,3,4,5,6,7]
        assert patches.shape == (1, 1, 8)
        np.testing.assert_array_equal(patches[0, 0], np.arange(8))
    
    def test_truncation(self):
        """Test that series is truncated to fit exact patches."""
        preprocessor = TimeSeriesPreprocessor()
        
        # 10 timesteps with patch_size=8 -> 1 patch, 2 timesteps dropped
        target = np.arange(10, dtype=np.float32).reshape(1, 10)
        sample = {"target": target, "freq": "Y"}  # Patch size = 8
        
        result = preprocessor.process(sample)
        patches = result["patches"]
        
        assert patches.shape == (1, 1, 8)
        # Only first 8 values should be kept
        np.testing.assert_array_equal(patches[0, 0], np.arange(8))
    
    def test_to_tensor(self):
        """Test conversion to PyTorch tensors."""
        preprocessor = TimeSeriesPreprocessor()
        
        target = np.arange(256, dtype=np.float32).reshape(1, 256)
        sample = {"target": target, "freq": "D"}
        
        result = preprocessor.process(sample)
        result = preprocessor.to_tensor(result)
        
        assert isinstance(result["patches"], torch.Tensor)
        assert isinstance(result["target"], torch.Tensor)
    
    def test_frequency_adaptive_patch_size(self):
        """Test that patch size adapts to frequency."""
        preprocessor = TimeSeriesPreprocessor()
        
        # Test different frequencies
        frequencies = [
            ("Y", 8),
            ("M", 32),
            ("D", 128),
            ("H", 128),
            ("15T", 256),
        ]
        
        for freq, expected_patch_size in frequencies:
            target = np.arange(1000, dtype=np.float32).reshape(1, 1000)
            sample = {"target": target, "freq": freq}
            result = preprocessor.process(sample)
            assert result["patch_size"] == expected_patch_size

