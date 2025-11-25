"""Tests for TimeSeriesDataset."""

import numpy as np
import pytest

from data_processing.dataset import TimeSeriesDataset


class TestTimeSeriesDataset:
    """Test TimeSeriesDataset loading."""
    
    def test_load_dataset(self):
        """Test loading a real dataset."""
        # Use one of the smaller datasets for testing
        dataset = TimeSeriesDataset(
            "data/training/favorita_sales",
            min_length=100
        )
        
        assert len(dataset) > 0, "Dataset should contain at least one series"
        
        # Get first sample
        sample = dataset[0]
        assert "target" in sample
        assert "start" in sample
        assert "freq" in sample
        assert "item_id" in sample
        assert "dataset_name" in sample
        assert "length" in sample
    
    def test_target_shape(self):
        """Test that targets have correct shape."""
        dataset = TimeSeriesDataset(
            "data/training/favorita_sales",
            min_length=100
        )
        
        sample = dataset[0]
        target = sample["target"]
        
        # Target should be 2D: (num_variates, length)
        assert target.ndim == 2
        assert target.shape[0] >= 1  # At least one variate
        assert target.shape[1] >= 100  # At least min_length
        assert target.dtype == np.float32
    
    def test_min_length_filter(self):
        """Test that min_length filtering works."""
        dataset_short = TimeSeriesDataset(
            "data/training/favorita_sales",
            min_length=10
        )
        dataset_long = TimeSeriesDataset(
            "data/training/favorita_sales",
            min_length=10000
        )
        
        # Shorter min_length should include more series
        assert len(dataset_short) >= len(dataset_long)

