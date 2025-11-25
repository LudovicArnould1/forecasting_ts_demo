"""Integration tests for the complete data processing pipeline."""

import logging

import pytest
import torch

from data_processing.dataloader import (
    TimeSeriesTrainingDataset,
    TimeSeriesValidationDataset,
    collate_fn,
    create_train_dataloader,
    create_val_dataloader,
)
from data_processing.preprocessing import TimeSeriesPreprocessor
from data_processing.samplers import RandomWindowSampler, SequentialWindowSampler

# Configure logging to see info messages during tests
logging.basicConfig(level=logging.INFO)


class TestDataProcessingPipeline:
    """Test complete data processing pipeline."""
    
    def test_training_dataset_creation(self):
        """Test creating a training dataset."""
        preprocessor = TimeSeriesPreprocessor()
        sampler = RandomWindowSampler()
        
        dataset = TimeSeriesTrainingDataset(
            data_dir="data/training/favorita_sales",
            preprocessor=preprocessor,
            sampler=sampler,
            min_length=1000,
        )
        
        assert len(dataset) > 0
    
    def test_training_dataset_getitem(self):
        """Test getting items from training dataset."""
        preprocessor = TimeSeriesPreprocessor()
        # Use smaller min/max lengths suitable for daily data with ~13 patches max
        sampler = RandomWindowSampler(
            min_context_length=4,
            max_context_length=8,
            min_prediction_length=2,
            max_prediction_length=4,
            max_window_length=12,
        )
        
        dataset = TimeSeriesTrainingDataset(
            data_dir="data/training/favorita_sales",
            preprocessor=preprocessor,
            sampler=sampler,
            min_length=1500,  # At least 11-12 patches
        )
        
        sample = dataset[0]
        
        # Check required fields
        assert "context_patches" in sample
        assert "target_patches" in sample
        assert isinstance(sample["context_patches"], torch.Tensor)
        assert isinstance(sample["target_patches"], torch.Tensor)
    
    def test_validation_dataset_creation(self):
        """Test creating a validation dataset."""
        preprocessor = TimeSeriesPreprocessor()
        # Use smaller window sizes for daily data
        sampler = SequentialWindowSampler(context_length=6, prediction_length=2)
        
        dataset = TimeSeriesValidationDataset(
            data_dir="data/training/favorita_sales",
            preprocessor=preprocessor,
            sampler=sampler,
            min_length=1500,
        )
        
        # Should have multiple windows
        assert len(dataset) > 0
    
    def test_validation_dataset_getitem(self):
        """Test getting items from validation dataset."""
        preprocessor = TimeSeriesPreprocessor()
        sampler = SequentialWindowSampler(context_length=6, prediction_length=2)
        
        dataset = TimeSeriesValidationDataset(
            data_dir="data/training/favorita_sales",
            preprocessor=preprocessor,
            sampler=sampler,
            min_length=1500,
        )
        
        sample = dataset[0]
        
        assert "context_patches" in sample
        assert "target_patches" in sample
        assert sample["context_length"] == 6
        assert sample["prediction_length"] == 2


class TestCollateFunction:
    """Test collate function for batching."""
    
    def test_collate_same_lengths(self):
        """Test collating samples with same lengths."""
        # Create mock samples
        batch = [
            {
                "context_patches": torch.randn(1, 32, 128),
                "target_patches": torch.randn(1, 8, 128),
                "context_length": 32,
                "prediction_length": 8,
                "freq": "D",
                "dataset_name": "test",
                "item_id": "0",
            }
            for _ in range(4)
        ]
        
        result = collate_fn(batch)
        
        assert result["context_patches"].shape == (4, 1, 32, 128)
        assert result["target_patches"].shape == (4, 1, 8, 128)
        assert result["context_mask"].shape == (4, 32)
        assert result["target_mask"].shape == (4, 8)
        assert torch.all(result["context_mask"] == 1)
        assert torch.all(result["target_mask"] == 1)
    
    def test_collate_variable_lengths(self):
        """Test collating samples with variable lengths."""
        batch = [
            {
                "context_patches": torch.randn(1, 32, 128),
                "target_patches": torch.randn(1, 8, 128),
                "context_length": 32,
                "prediction_length": 8,
                "freq": "D",
                "dataset_name": "test",
                "item_id": "0",
            },
            {
                "context_patches": torch.randn(1, 16, 128),
                "target_patches": torch.randn(1, 4, 128),
                "context_length": 16,
                "prediction_length": 4,
                "freq": "D",
                "dataset_name": "test",
                "item_id": "1",
            },
        ]
        
        result = collate_fn(batch)
        
        # Should pad to max lengths (32 and 8)
        assert result["context_patches"].shape == (2, 1, 32, 128)
        assert result["target_patches"].shape == (2, 1, 8, 128)
        
        # Check masks
        assert torch.all(result["context_mask"][0] == 1)
        assert torch.all(result["context_mask"][1, :16] == 1)
        assert torch.all(result["context_mask"][1, 16:] == 0)


@pytest.mark.slow
class TestDataLoaders:
    """Test DataLoader creation and iteration."""
    
    def test_create_train_dataloader(self):
        """Test creating training dataloader."""
        loader = create_train_dataloader(
            data_dir="data/training/favorita_sales",
            batch_size=4,
            num_workers=0,  # Use 0 for testing
            min_length=1000,
        )
        
        assert loader is not None
        assert loader.batch_size == 4
    
    def test_train_dataloader_iteration(self):
        """Test iterating through training dataloader."""
        loader = create_train_dataloader(
            data_dir="data/training/favorita_sales",
            batch_size=4,
            num_workers=0,
            max_window_length=12,  # Suitable for daily data
            min_context_length=4,
            max_context_length=8,
            min_prediction_length=2,
            max_prediction_length=4,
            min_length=1500,
        )
        
        # Get one batch
        batch = next(iter(loader))
        
        assert "context_patches" in batch
        assert "target_patches" in batch
        assert "context_mask" in batch
        assert "target_mask" in batch
        assert batch["context_patches"].shape[0] <= 4  # batch size
        
        # Check basic properties (NaNs are OK in real data)
        assert batch["context_patches"].dtype == torch.float32
        assert batch["target_patches"].dtype == torch.float32
    
    def test_create_val_dataloader(self):
        """Test creating validation dataloader."""
        loader = create_val_dataloader(
            data_dir="data/training/favorita_sales",
            batch_size=4,
            num_workers=0,
            context_length=32,
            prediction_length=8,
            min_length=1000,
        )
        
        assert loader is not None
    
    def test_val_dataloader_iteration(self):
        """Test iterating through validation dataloader."""
        loader = create_val_dataloader(
            data_dir="data/training/favorita_sales",
            batch_size=4,
            num_workers=0,
            context_length=6,
            prediction_length=2,
            min_length=1500,
        )
        
        # Get one batch
        batch = next(iter(loader))
        
        assert "context_patches" in batch
        assert "target_patches" in batch
        
        # All should have same length in validation
        assert torch.all(batch["context_length"] == 6)
        assert torch.all(batch["prediction_length"] == 2)
    
    def test_multivariate_data(self):
        """Test with multivariate dataset (beijing air quality)."""
        loader = create_train_dataloader(
            data_dir="GiftEvalPretrain/beijing_air_quality",
            batch_size=2,
            num_workers=0,
            min_length=5000,
        )
        
        batch = next(iter(loader))
        
        # Check that we have multiple variates
        num_variates = batch["context_patches"].shape[1]
        assert num_variates > 1, "Beijing air quality should be multivariate"
        
        # Both context and target should have same number of variates
        assert batch["context_patches"].shape[1] == batch["target_patches"].shape[1]


class TestEndToEnd:
    """End-to-end tests with real data."""
    
    def test_full_training_pipeline(self):
        """Test complete training pipeline."""
        # Create loader
        loader = create_train_dataloader(
            data_dir="data/training/favorita_sales",
            batch_size=2,
            num_workers=0,
            max_window_length=12,
            min_context_length=4,
            max_context_length=8,
            min_prediction_length=2,
            max_prediction_length=4,
            min_length=1500,
        )
        
        # Iterate through a few batches
        num_batches = 3
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            
            # Verify batch structure
            assert batch["context_patches"].dtype == torch.float32
            assert batch["target_patches"].dtype == torch.float32
            assert batch["context_mask"].dtype == torch.float32
            
            # NaNs are expected in real data (favorita_sales has missing values)
            
            print(f"Batch {i}: context={batch['context_patches'].shape}, "
                  f"target={batch['target_patches'].shape}")
    
    def test_full_validation_pipeline(self):
        """Test complete validation pipeline."""
        loader = create_val_dataloader(
            data_dir="data/training/favorita_sales",
            batch_size=2,
            num_workers=0,
            context_length=6,
            prediction_length=2,
            min_length=1500,
        )
        
        # Iterate through a few batches
        num_batches = 3
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            
            # All validation batches should have fixed sizes
            assert batch["context_patches"].shape[2] == 6
            assert batch["target_patches"].shape[2] == 2
            
            print(f"Val batch {i}: context={batch['context_patches'].shape}, "
                  f"target={batch['target_patches'].shape}")

