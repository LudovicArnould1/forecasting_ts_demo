"""DataLoader construction for time series training."""

import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from data_processing.dataset import TimeSeriesDataset
from data_processing.preprocessing import TimeSeriesPreprocessor
from data_processing.samplers import RandomWindowSampler, SequentialWindowSampler

logger = logging.getLogger(__name__)


class DatasetAwareBatchSampler(Sampler):
    """Sampler that implements Strategy 1: Dataset-first sampling with pure batches.
    
    This sampler:
    1. First samples which dataset to use (based on probabilities)
    2. Then samples a batch of series from that dataset
    3. Ensures each batch contains only series from one dataset (pure batches)
    
    This is important for efficiency when datasets have different frequencies,
    as it avoids padding overhead from mixed patch sizes.
    
    Args:
        dataset: Base TimeSeriesDataset
        batch_size: Batch size
        temperature: Sampling temperature for datasets
            - 1.0 = proportional to dataset size
            - 0.5 = square root sampling
            - 0.0 = uniform sampling
        drop_last: Whether to drop the last incomplete batch
        
    Examples:
        >>> base_dataset = TimeSeriesDataset("data/chunks/train")
        >>> sampler = DatasetAwareBatchSampler(base_dataset, batch_size=32)
        >>> loader = DataLoader(dataset, batch_sampler=sampler)
    """
    
    def __init__(
        self,
        dataset: TimeSeriesDataset,
        batch_size: int,
        temperature: float = 1.0,
        drop_last: bool = False,
        seed: int | None = None,
    ):
        """Initialize dataset-aware batch sampler."""
        self.batch_size = batch_size
        self.temperature = temperature
        self.drop_last = drop_last
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Group series indices by dataset_name
        self.dataset_groups = defaultdict(list)
        
        # Check if this is a wrapped dataset with base_dataset attribute
        base_dataset = getattr(dataset, 'base_dataset', None)
        if base_dataset is not None and hasattr(dataset, 'valid_indices'):
            # This is a filtered dataset - use valid indices only
            for idx in range(len(dataset)):
                # Get the corresponding series from base dataset
                base_idx = dataset.valid_indices[idx]
                sample = base_dataset[base_idx]
                dataset_name = sample['dataset_name']
                self.dataset_groups[dataset_name].append(idx)
        else:
            # This is a regular dataset
            for idx in range(len(dataset)):
                sample = dataset[idx]
                dataset_name = sample['dataset_name']
                self.dataset_groups[dataset_name].append(idx)
        
        # Compute dataset sampling probabilities
        self.dataset_names = list(self.dataset_groups.keys())
        dataset_sizes = {
            name: len(indices) for name, indices in self.dataset_groups.items()
        }
        self.sampling_probs = self._compute_probs(dataset_sizes, temperature)
        
        # Log dataset composition
        logger.info(f"Dataset-aware sampler initialized with {len(self.dataset_names)} datasets:")
        for name in sorted(self.dataset_names):
            count = len(self.dataset_groups[name])
            prob = self.sampling_probs[name]
            logger.info(f"  {name}: {count:,} series (sampling prob: {prob:.4f})")
    
    def _compute_probs(
        self, dataset_sizes: dict[str, int], temperature: float
    ) -> dict[str, float]:
        """Compute sampling probabilities based on dataset sizes."""
        if temperature == 0:
            # Uniform sampling
            n = len(dataset_sizes)
            return {name: 1.0 / n for name in dataset_sizes}
        
        # Temperature-based sampling
        powered = {name: size ** temperature for name, size in dataset_sizes.items()}
        total = sum(powered.values())
        return {name: p / total for name, p in powered.items()}
    
    def __iter__(self):
        """Generate batches by first sampling dataset, then series within dataset."""
        # Prepare probabilities array for numpy
        probs = np.array([self.sampling_probs[name] for name in self.dataset_names])
        
        # Estimate total number of batches
        total_series = sum(len(indices) for indices in self.dataset_groups.values())
        num_batches = total_series // self.batch_size
        
        for _ in range(num_batches):
            # Step 1: Sample which dataset to use
            dataset_idx = np.random.choice(len(self.dataset_names), p=probs)
            dataset_name = self.dataset_names[dataset_idx]
            
            # Step 2: Sample batch_size series from that dataset
            available_indices = self.dataset_groups[dataset_name]
            
            if len(available_indices) < self.batch_size:
                # Sample with replacement if not enough series
                batch_indices = np.random.choice(
                    available_indices, size=self.batch_size, replace=True
                )
            else:
                # Sample without replacement
                batch_indices = np.random.choice(
                    available_indices, size=self.batch_size, replace=False
                )
            
            yield batch_indices.tolist()
    
    def __len__(self) -> int:
        """Return number of batches per epoch."""
        total_series = sum(len(indices) for indices in self.dataset_groups.values())
        if self.drop_last:
            return total_series // self.batch_size
        else:
            return (total_series + self.batch_size - 1) // self.batch_size


class TimeSeriesTrainingDataset(Dataset):
    """Wraps TimeSeriesDataset with preprocessing and random sampling for training.
    
    Args:
        data_dir: Directory containing parquet files
        preprocessor: TimeSeriesPreprocessor instance
        sampler: RandomWindowSampler instance
        min_length: Minimum series length
        
    Examples:
        >>> preprocessor = TimeSeriesPreprocessor()
        >>> sampler = RandomWindowSampler()
        >>> dataset = TimeSeriesTrainingDataset("data/training", preprocessor, sampler)
        >>> sample = dataset[0]
    """
    
    def __init__(
        self,
        data_dir: str | Path,
        preprocessor: TimeSeriesPreprocessor,
        sampler: RandomWindowSampler,
        min_length: int = 256,
    ):
        """Initialize training dataset and filter series with insufficient patches."""
        base_dataset = TimeSeriesDataset(data_dir, min_length=min_length)
        self.preprocessor = preprocessor
        self.sampler = sampler
        
        # Calculate minimum number of patches needed
        min_patches_needed = sampler.min_context_length + sampler.min_prediction_length
        
        # Filter series that will have enough patches
        logger.info(f"Filtering series with at least {min_patches_needed} patches...")
        self.valid_indices = []
        
        for idx in range(len(base_dataset)):
            sample = base_dataset[idx]
            try:
                # Apply preprocessing to check patch count
                processed = preprocessor.process(sample)
                if processed["num_patches"] >= min_patches_needed:
                    self.valid_indices.append(idx)
            except ValueError as e:
                # Skip series that can't be patched (too short for patch size)
                logger.debug(f"Skipping series {idx}: {e}")
                continue
        
        self.base_dataset = base_dataset
        
        logger.info(
            f"Filtered to {len(self.valid_indices)}/{len(base_dataset)} series "
            f"with sufficient patches"
        )
        
        if len(self.valid_indices) == 0:
            raise ValueError(
                f"No series have enough patches! Need at least {min_patches_needed} patches. "
                f"Consider lowering min_length or adjusting sampler parameters."
            )
    
    def __len__(self) -> int:
        """Return number of valid series in dataset."""
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a preprocessed and sampled window from a series.
        
        Args:
            idx: Index in valid_indices (not base dataset)
            
        Returns:
            Dictionary containing sampled window data
        """
        # Map to base dataset index
        base_idx = self.valid_indices[idx]
        
        # Get base sample
        sample = self.base_dataset[base_idx]
        
        # Apply preprocessing (patching)
        sample = self.preprocessor.process(sample)
        
        # Apply random sampling
        sample = self.sampler.sample(sample)
        
        # Convert to tensors
        sample = self.preprocessor.to_tensor(sample)
        
        return sample


class TimeSeriesValidationDataset(Dataset):
    """Wraps TimeSeriesDataset with preprocessing and sequential sampling for validation.
    
    For validation, we generate multiple windows per series sequentially.
    
    Args:
        data_dir: Directory containing parquet files
        preprocessor: TimeSeriesPreprocessor instance
        sampler: SequentialWindowSampler instance
        min_length: Minimum series length
        
    Examples:
        >>> preprocessor = TimeSeriesPreprocessor()
        >>> sampler = SequentialWindowSampler()
        >>> dataset = TimeSeriesValidationDataset("data/val", preprocessor, sampler)
        >>> sample = dataset[0]
    """
    
    def __init__(
        self,
        data_dir: str | Path,
        preprocessor: TimeSeriesPreprocessor,
        sampler: SequentialWindowSampler,
        min_length: int = 256,
    ):
        """Initialize validation dataset and filter series with insufficient patches."""
        self.base_dataset = TimeSeriesDataset(data_dir, min_length=min_length)
        self.preprocessor = preprocessor
        self.sampler = sampler
        
        # Calculate minimum number of patches needed
        min_patches_needed = sampler.context_length + sampler.prediction_length
        
        # Pre-generate all windows from all series
        self._build_window_index(min_patches_needed)
    
    def _build_window_index(self, min_patches_needed: int) -> None:
        """Pre-generate all sequential windows from all series."""
        self.windows = []
        skipped_count = 0
        
        logger.info(f"Building validation window index (need at least {min_patches_needed} patches)...")
        for idx in range(len(self.base_dataset)):
            sample = self.base_dataset[idx]
            
            # Apply preprocessing
            try:
                sample = self.preprocessor.process(sample)
                
                # Check if series has enough patches
                if sample["num_patches"] < min_patches_needed:
                    skipped_count += 1
                    continue
                
                # Generate all sequential windows
                windows = self.sampler.sample_all(sample)
                self.windows.extend(windows)
            except ValueError as e:
                logger.warning(
                    f"Skipping series {sample.get('item_id', idx)}: {e}"
                )
                skipped_count += 1
                continue
        
        logger.info(
            f"Generated {len(self.windows)} validation windows "
            f"({skipped_count} series skipped due to insufficient patches)"
        )
    
    def __len__(self) -> int:
        """Return total number of windows across all series."""
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a preprocessed window.
        
        Args:
            idx: Index of the window
            
        Returns:
            Dictionary containing window data
        """
        sample = self.windows[idx]
        
        # Convert to tensors
        sample = self.preprocessor.to_tensor(sample)
        
        return sample


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Collate function for batching variable-length sequences.
    
    Since context_length and prediction_length can vary (for training),
    we need to pad sequences to the max length in the batch.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Dictionary with batched tensors:
        - context_patches: (batch, num_variates, max_context_len, patch_size)
        - target_patches: (batch, num_variates, max_pred_len, patch_size)
        - context_mask: (batch, max_context_len) - 1 for valid, 0 for padding
        - target_mask: (batch, max_pred_len) - 1 for valid, 0 for padding
        - context_length: (batch,) - actual context lengths
        - prediction_length: (batch,) - actual prediction lengths
        - Other metadata fields
    """
    batch_size = len(batch)
    
    # Find max lengths in this batch
    max_context_len = max(s["context_length"] for s in batch)
    max_pred_len = max(s["prediction_length"] for s in batch)
    
    # Get dimensions from first sample
    num_variates, _, patch_size = batch[0]["context_patches"].shape
    
    # Initialize padded tensors
    context_batch = torch.zeros(
        batch_size, num_variates, max_context_len, patch_size
    )
    target_batch = torch.zeros(
        batch_size, num_variates, max_pred_len, patch_size
    )
    context_mask = torch.zeros(batch_size, max_context_len)
    target_mask = torch.zeros(batch_size, max_pred_len)
    
    context_lengths = []
    prediction_lengths = []
    
    for i, sample in enumerate(batch):
        ctx_len = sample["context_length"]
        pred_len = sample["prediction_length"]
        
        # Fill in actual values
        context_batch[i, :, :ctx_len, :] = sample["context_patches"]
        target_batch[i, :, :pred_len, :] = sample["target_patches"]
        
        # Mark valid positions in mask
        context_mask[i, :ctx_len] = 1
        target_mask[i, :pred_len] = 1
        
        context_lengths.append(ctx_len)
        prediction_lengths.append(pred_len)
    
    return {
        "context_patches": context_batch,
        "target_patches": target_batch,
        "context_mask": context_mask,
        "target_mask": target_mask,
        "context_length": torch.tensor(context_lengths),
        "prediction_length": torch.tensor(prediction_lengths),
        "freq": [s["freq"] for s in batch],
        "dataset_name": [s["dataset_name"] for s in batch],
        "item_id": [s["item_id"] for s in batch],
    }


def create_train_dataloader(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    max_window_length: int = 128,
    min_length: int = 256,
    min_context_length: int = 16,
    max_context_length: int = 96,
    min_prediction_length: int = 4,
    max_prediction_length: int = 32,
    use_dataset_sampling: bool = True,
    temperature: float = 1.0,
) -> DataLoader:
    """Create training dataloader.
    
    Args:
        data_dir: Directory with training data
        batch_size: Batch size
        num_workers: Number of workers for data loading
        max_window_length: Maximum window length in patches
        min_length: Minimum series length in timesteps
        min_context_length: Minimum context length in patches
        max_context_length: Maximum context length in patches
        min_prediction_length: Minimum prediction length in patches
        max_prediction_length: Maximum prediction length in patches
        use_dataset_sampling: If True, use Strategy 1 (dataset-first sampling)
        temperature: Sampling temperature for datasets (1.0=proportional, 0.5=sqrt)
        
    Returns:
        DataLoader for training
        
    Examples:
        >>> train_loader = create_train_dataloader("data/chunks/train", batch_size=32)
        >>> for batch in train_loader:
        ...     print(batch["context_patches"].shape)
        ...     break
    """
    preprocessor = TimeSeriesPreprocessor(max_patch_length=max_window_length)
    sampler = RandomWindowSampler(
        max_window_length=max_window_length,
        min_context_length=min_context_length,
        max_context_length=max_context_length,
        min_prediction_length=min_prediction_length,
        max_prediction_length=max_prediction_length,
    )
    
    dataset = TimeSeriesTrainingDataset(
        data_dir=data_dir,
        preprocessor=preprocessor,
        sampler=sampler,
        min_length=min_length,
    )
    
    logger.info(f"Created training dataset with {len(dataset)} series")
    
    if use_dataset_sampling:
        # Strategy 1: Dataset-first sampling with pure batches
        logger.info("Using dataset-aware sampling (Strategy 1)")
        batch_sampler = DatasetAwareBatchSampler(
            dataset=dataset,  # Pass the wrapper dataset, not base_dataset
            batch_size=batch_size,
            temperature=temperature,
            drop_last=False,
        )
        
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    else:
        # Standard uniform sampling
        logger.info("Using standard uniform sampling")
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )


def create_val_dataloader(
    data_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    context_length: int = 64,
    prediction_length: int = 16,
    min_length: int = 256,
) -> DataLoader:
    """Create validation dataloader.
    
    Args:
        data_dir: Directory with validation data
        batch_size: Batch size
        num_workers: Number of workers for data loading
        context_length: Fixed context length in patches
        prediction_length: Fixed prediction length in patches
        min_length: Minimum series length in timesteps
        
    Returns:
        DataLoader for validation
        
    Examples:
        >>> val_loader = create_val_dataloader("data/val", batch_size=32)
        >>> for batch in val_loader:
        ...     print(batch["context_patches"].shape)
        ...     break
    """
    preprocessor = TimeSeriesPreprocessor(max_patch_length=128)
    sampler = SequentialWindowSampler(
        context_length=context_length,
        prediction_length=prediction_length
    )
    
    dataset = TimeSeriesValidationDataset(
        data_dir=data_dir,
        preprocessor=preprocessor,
        sampler=sampler,
        min_length=min_length,
    )
    
    logger.info(f"Created validation dataset with {len(dataset)} windows")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

