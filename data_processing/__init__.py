"""Data processing module for time series forecasting.

This module provides a complete pipeline for processing time series data
following the Moirai paper's approach:

- Frequency-adaptive patching
- Variable-length window sampling (training)
- Fixed-length sequential sampling (validation)
- Automatic batching with padding and masking

Quick Start:
    >>> from data_processing import create_train_dataloader, create_val_dataloader
    >>> 
    >>> train_loader = create_train_dataloader("data/training", batch_size=32)
    >>> val_loader = create_val_dataloader("data/val", batch_size=32)
    >>> 
    >>> for batch in train_loader:
    ...     context = batch['context_patches']
    ...     target = batch['target_patches']
    ...     # Your model here...

See DATA_PROCESSING_README.md for detailed documentation.
"""

from data_processing.dataloader import (
    TimeSeriesTrainingDataset,
    TimeSeriesValidationDataset,
    create_train_dataloader,
    create_val_dataloader,
)
from data_processing.dataset import TimeSeriesDataset
from data_processing.preprocessing import TimeSeriesPreprocessor
from data_processing.samplers import RandomWindowSampler, SequentialWindowSampler
from data_processing.utils import get_patch_size_for_freq, parse_frequency

__all__ = [
    # High-level API (recommended)
    "create_train_dataloader",
    "create_val_dataloader",
    # Low-level components
    "TimeSeriesDataset",
    "TimeSeriesPreprocessor",
    "RandomWindowSampler",
    "SequentialWindowSampler",
    "TimeSeriesTrainingDataset",
    "TimeSeriesValidationDataset",
    # Utilities
    "get_patch_size_for_freq",
    "parse_frequency",
]

__version__ = "0.1.0"

