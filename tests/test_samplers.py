"""Tests for sampling strategies."""

import numpy as np
import pytest

from data_processing.samplers import RandomWindowSampler, SequentialWindowSampler


class TestRandomWindowSampler:
    """Test random window sampling."""
    
    def test_basic_sampling(self):
        """Test basic random window sampling."""
        sampler = RandomWindowSampler(
            max_window_length=128,
            min_context_length=16,
            max_context_length=96,
            min_prediction_length=4,
            max_prediction_length=32,
        )
        
        # Create patches: (1 variate, 200 patches, 128 patch_size)
        patches = np.random.randn(1, 200, 128).astype(np.float32)
        sample = {"patches": patches, "num_patches": 200, "patch_size": 128}
        
        result = sampler.sample(sample)
        
        assert "context_patches" in result
        assert "target_patches" in result
        assert "context_length" in result
        assert "prediction_length" in result
    
    def test_sampled_shapes(self):
        """Test that sampled windows have correct shapes."""
        sampler = RandomWindowSampler()
        
        patches = np.random.randn(1, 200, 128).astype(np.float32)
        sample = {"patches": patches, "num_patches": 200, "patch_size": 128}
        
        result = sampler.sample(sample)
        
        context = result["context_patches"]
        target = result["target_patches"]
        
        # Context and target should have correct shapes
        assert context.shape[0] == 1  # num_variates
        assert context.shape[1] == result["context_length"]
        assert context.shape[2] == 128  # patch_size
        
        assert target.shape[0] == 1
        assert target.shape[1] == result["prediction_length"]
        assert target.shape[2] == 128
    
    def test_window_length_constraint(self):
        """Test that context + prediction <= max_window_length."""
        sampler = RandomWindowSampler(max_window_length=64)
        
        patches = np.random.randn(1, 200, 128).astype(np.float32)
        sample = {"patches": patches, "num_patches": 200, "patch_size": 128}
        
        # Sample multiple times to test
        for _ in range(10):
            result = sampler.sample(sample)
            total = result["context_length"] + result["prediction_length"]
            assert total <= 64
    
    def test_multivariate_sampling(self):
        """Test sampling with multivariate time series."""
        sampler = RandomWindowSampler()
        
        # 3 variates
        patches = np.random.randn(3, 200, 128).astype(np.float32)
        sample = {"patches": patches, "num_patches": 200, "patch_size": 128}
        
        result = sampler.sample(sample)
        
        assert result["context_patches"].shape[0] == 3
        assert result["target_patches"].shape[0] == 3


class TestSequentialWindowSampler:
    """Test sequential window sampling."""
    
    def test_basic_sequential_sampling(self):
        """Test basic sequential window sampling."""
        sampler = SequentialWindowSampler(context_length=64, prediction_length=16)
        
        # 200 patches
        patches = np.random.randn(1, 200, 128).astype(np.float32)
        sample = {"patches": patches, "num_patches": 200, "patch_size": 128}
        
        windows = sampler.sample_all(sample)
        
        assert len(windows) > 0
        assert all("context_patches" in w for w in windows)
        assert all("target_patches" in w for w in windows)
    
    def test_fixed_window_sizes(self):
        """Test that sequential windows have fixed sizes."""
        sampler = SequentialWindowSampler(context_length=64, prediction_length=16)
        
        patches = np.random.randn(1, 200, 128).astype(np.float32)
        sample = {"patches": patches, "num_patches": 200, "patch_size": 128}
        
        windows = sampler.sample_all(sample)
        
        for window in windows:
            assert window["context_length"] == 64
            assert window["prediction_length"] == 16
            assert window["context_patches"].shape == (1, 64, 128)
            assert window["target_patches"].shape == (1, 16, 128)
    
    def test_non_overlapping_windows(self):
        """Test that sequential windows don't overlap."""
        sampler = SequentialWindowSampler(context_length=32, prediction_length=16)
        
        patches = np.random.randn(1, 200, 128).astype(np.float32)
        sample = {"patches": patches, "num_patches": 200, "patch_size": 128}
        
        windows = sampler.sample_all(sample)
        
        # Check that each window starts after the previous one's context
        start_indices = [w["window_start_idx"] for w in windows]
        
        for i in range(1, len(start_indices)):
            # Next window should start prediction_length ahead
            assert start_indices[i] == start_indices[i-1] + 16
    
    def test_number_of_windows(self):
        """Test correct number of sequential windows."""
        sampler = SequentialWindowSampler(context_length=64, prediction_length=16)
        
        # With 200 patches, context=64, pred=16:
        # First window needs 64+16=80 patches starting at 0
        # Windows at positions: 0, 16, 32, ..., until no more room
        # Last valid start: 200 - 80 = 120
        # Number of windows: (120 - 0) / 16 + 1 = 8.5 -> 8 windows
        patches = np.random.randn(1, 200, 128).astype(np.float32)
        sample = {"patches": patches, "num_patches": 200, "patch_size": 128}
        
        windows = sampler.sample_all(sample)
        
        # Verify we get the expected number
        expected = (200 - 64 - 16) // 16 + 1
        assert len(windows) == expected
    
    def test_multivariate_sequential(self):
        """Test sequential sampling with multivariate series."""
        sampler = SequentialWindowSampler(context_length=32, prediction_length=8)
        
        patches = np.random.randn(3, 100, 128).astype(np.float32)
        sample = {"patches": patches, "num_patches": 100, "patch_size": 128}
        
        windows = sampler.sample_all(sample)
        
        assert len(windows) > 0
        for window in windows:
            assert window["context_patches"].shape[0] == 3
            assert window["target_patches"].shape[0] == 3

