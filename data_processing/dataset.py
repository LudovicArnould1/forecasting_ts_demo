"""Dataset class for loading time series data from parquet files."""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)




class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for loading time series from parquet files.
    
    This dataset loads time series data stored in parquet format following
    the GiftEvalPretrain schema:
    - dataset_name: Name of the dataset
    - item_id: Unique identifier for each series
    - start: Start timestamp of the series
    - freq: Frequency string (e.g., 'H', '15T', 'D')
    - target: Array of target values, shape (num_variates, length) or (length,)
    - past_feat_dynamic_real: Optional covariates (not used in light version)
    
    NaN Handling:
    - Any series containing NaN values is discarded
    
    Args:
        data_dir: Path to directory containing parquet files (can have subdirs)
        min_length: Minimum length of time series to include (default: 128)
        
    Examples:
        >>> dataset = TimeSeriesDataset("data/training")
        >>> sample = dataset[0]
        >>> print(sample.keys())
        dict_keys(['target', 'start', 'freq', 'item_id', 'dataset_name'])
    """
    
    def __init__(self, data_dir: str | Path, min_length: int = 128):
        """Initialize dataset by loading all parquet files from directory.
        
        Args:
            data_dir: Path to directory containing parquet files
            min_length: Minimum series length to include
        """
        self.data_dir = Path(data_dir)
        self.min_length = min_length
        self.series_list: list[dict[str, Any]] = []
        
        # Statistics for logging
        self.stats = {
            "total_processed": 0,
            "discarded_too_short": 0,
            "discarded_nan": 0,
            "clean": 0,
        }
        
        # Timing statistics
        self.timing = {
            "total_time": 0.0,
            "load_time": 0.0,
            "processing_time": 0.0,
        }

        logger.info(f"Loading dataset from {self.data_dir}")
        start_time = time.time()
        
        self._load_all_parquet_files()
        
        self.timing["total_time"] = time.time() - start_time
        
        # Log statistics
        logger.info(
            f"Loaded {len(self.series_list)} time series from {self.data_dir}"
        )
        logger.info(f"Dataset loading statistics:")
        logger.info(f"  Total series processed: {self.stats['total_processed']}")
        logger.info(f"  Clean series (no NaN): {self.stats['clean']}")
        logger.info(f"  Discarded (too short): {self.stats['discarded_too_short']}")
        logger.info(f"  Discarded (contains NaN): {self.stats['discarded_nan']}")
        if self.stats['total_processed'] > 0:
            acceptance_rate = len(self.series_list) / self.stats['total_processed'] * 100
            logger.info(f"  Acceptance rate: {acceptance_rate:.2f}%")
        
        # Log timing
        logger.info(f"Timing statistics:")
        logger.info(f"  Total time: {self.timing['total_time']:.2f}s")
        logger.info(f"  Load time: {self.timing['load_time']:.2f}s")
        logger.info(f"  Processing time: {self.timing['processing_time']:.2f}s")
        if len(self.series_list) > 0:
            logger.info(f"  Time per series: {self.timing['total_time'] / len(self.series_list) * 1000:.2f}ms")
    
    def _load_all_parquet_files(self) -> None:
        """Recursively load all parquet files from data_dir."""
        parquet_files = list(self.data_dir.rglob("*.parquet"))
        
        if not parquet_files:
            raise ValueError(f"No parquet files found in {self.data_dir}")
        
        for parquet_file in parquet_files:
            self._load_parquet_file(parquet_file)
    
    def _process_target_array(self, target: Any) -> np.ndarray | None:
        """Process a single target array into standardized format.
        
        Args:
            target: Target array in various formats
            
        Returns:
            Numpy array of shape (num_variates, length) or None if invalid
        """
        target = np.array(target)
        
        # Handle different target formats
        # Case 1: target is array of shape (N,) where each element is an array
        # This is the multivariate case OR univariate stored as single element
        if target.dtype == object:
            # Convert to list of arrays, then stack
            try:
                arrays = [np.array(x, dtype=np.float32) for x in target]
                target = np.stack(arrays, axis=0)  # Shape: (num_variates, length)
            except (ValueError, TypeError):
                return None
        # Case 2: target is already numeric array of shape (length,)
        elif target.ndim == 1:
            # Univariate: reshape to (1, length)
            target = target.reshape(1, -1)
        # Case 3: target is already 2D array
        elif target.ndim == 2:
            # Already correct shape (num_variates, length)
            pass
        else:
            return None
        
        return target.astype(np.float32)
    
    def _load_parquet_file(self, filepath: Path) -> None:
        """Load a single parquet file and add valid series to the list.
        
        Uses vectorized operations for significantly faster processing.
        
        Args:
            filepath: Path to parquet file
        """
        try:
            # Time the loading
            load_start = time.time()
            table = pq.read_table(filepath)
            df = table.to_pandas()
            load_time = time.time() - load_start
            self.timing["load_time"] += load_time
            
            logger.info(f"Loaded {len(df)} time series from {filepath} in {load_time:.2f}s")
            
            # Time the processing
            process_start = time.time()
            
            initial_count = len(df)
            self.stats["total_processed"] += initial_count
            
            # Step 1: Process all target arrays vectorized
            logger.info("Processing target arrays...")
            targets = df["target"].apply(self._process_target_array)
            
            # Filter out invalid targets
            valid_mask = targets.notna()
            df = df[valid_mask].copy()
            targets = targets[valid_mask]
            
            if len(df) == 0:
                logger.warning(f"No valid targets in {filepath}")
                return
            
            # Step 2: Vectorized length calculation and filtering
            logger.info("Filtering by length...")
            lengths = targets.apply(lambda x: x.shape[1])
            length_mask = lengths >= self.min_length
            
            discarded_short = (~length_mask).sum()
            self.stats["discarded_too_short"] += discarded_short
            
            df = df[length_mask].copy()
            targets = targets[length_mask]
            lengths = lengths[length_mask]
            
            if len(df) == 0:
                logger.warning(f"No series meet minimum length in {filepath}")
                return
            
            # Step 3: Vectorized NaN filtering (simply discard any series with NaN)
            logger.info("Filtering series with NaN values...")
            
            # Check for NaN in each series - much faster than interpolation
            has_nan = targets.apply(lambda x: np.isnan(x).any())
            no_nan_mask = ~has_nan
            
            # Update statistics
            num_with_nan = has_nan.sum()
            self.stats["discarded_nan"] += num_with_nan
            self.stats["clean"] += no_nan_mask.sum()
            
            if num_with_nan > 0:
                logger.info(f"Discarding {num_with_nan} series containing NaN values")
            
            # Filter to series without NaN
            df = df[no_nan_mask].copy()
            targets = targets[no_nan_mask]
            lengths = lengths[no_nan_mask]
            
            if len(df) == 0:
                logger.warning(f"No series passed NaN filtering in {filepath}")
                return
            
            # Step 4: Build series list efficiently
            logger.info(f"Building series list for {len(df)} series...")
            
            # Build list of dictionaries efficiently
            for idx, (_, row) in enumerate(df.iterrows()):
                self.series_list.append({
                    "target": targets.iloc[idx],
                    "start": row["start"],
                    "freq": row["freq"],
                    "item_id": row["item_id"],
                    "dataset_name": row["dataset_name"],
                    "length": int(lengths.iloc[idx]),
                })
            
            process_time = time.time() - process_start
            self.timing["processing_time"] += process_time
            
            logger.info(
                f"Processed {len(df)} valid series from {filepath} in {process_time:.2f}s "
                f"({len(df)/process_time:.0f} series/s)"
            )
        
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
    
    def __len__(self) -> int:
        """Return number of time series in dataset.
        
        Returns:
            Number of time series
        """
        return len(self.series_list)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a time series by index.
        
        Args:
            idx: Index of the series
            
        Returns:
            Dictionary containing:
            - target: numpy array of shape (num_variates, length)
            - start: timestamp of series start
            - freq: frequency string
            - item_id: unique identifier
            - dataset_name: name of the dataset
            - length: length of the series
        """
        return self.series_list[idx]
