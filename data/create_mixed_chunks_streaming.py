"""Create mixed-dataset training chunks with streaming (memory-efficient).

This script implements Strategy 1: Pre-Shuffled Chunking
- Uses a streaming approach to handle large datasets
- Samples series proportionally without loading all data into memory
- Creates 10GB chunks with mixed datasets
"""

import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetSampler:
    """Efficient sampler that streams through datasets."""
    
    def __init__(
        self,
        dataset_files: dict[str, list[Path]],
        sampling_probs: dict[str, float],
        seed: int = 42
    ):
        """Initialize sampler.
        
        Args:
            dataset_files: Mapping of dataset_name -> list of parquet files
            sampling_probs: Sampling probability for each dataset
            seed: Random seed
        """
        self.dataset_files = dataset_files
        self.sampling_probs = sampling_probs
        self.seed = seed
        
        # Create iterators for each dataset
        self.reset_iterators()
    
    def reset_iterators(self):
        """Reset all dataset iterators."""
        self.iterators = {}
        self.current_tables = {}
        self.current_indices = {}
        
        for dataset_name, files in self.dataset_files.items():
            # Shuffle files for this dataset
            shuffled_files = files.copy()
            random.Random(self.seed).shuffle(shuffled_files)
            self.iterators[dataset_name] = iter(shuffled_files)
            self.current_tables[dataset_name] = None
            self.current_indices[dataset_name] = []
    
    def _load_next_file(self, dataset_name: str) -> bool:
        """Load next file for a dataset.
        
        Returns:
            True if file loaded successfully, False if no more files
        """
        try:
            next_file = next(self.iterators[dataset_name])
            logger.debug(f"Loading {dataset_name}: {next_file.name}")
            
            # Read parquet file
            table = pq.read_table(next_file)
            df = table.to_pandas()
            
            # Create shuffled indices for this file
            indices = list(range(len(df)))
            random.shuffle(indices)
            
            self.current_tables[dataset_name] = df
            self.current_indices[dataset_name] = indices
            
            return True
        except StopIteration:
            return False
    
    def sample_series(self, dataset_name: str) -> pd.Series | None:
        """Sample one series from a dataset.
        
        Args:
            dataset_name: Name of dataset to sample from
            
        Returns:
            Pandas Series representing one time series, or None if dataset exhausted
        """
        # Check if we need to load a new file
        if (self.current_tables[dataset_name] is None or 
            len(self.current_indices[dataset_name]) == 0):
            if not self._load_next_file(dataset_name):
                return None  # Dataset exhausted
        
        # Sample from current file
        idx = self.current_indices[dataset_name].pop()
        series = self.current_tables[dataset_name].iloc[idx]
        
        return series


def get_dataset_info(train_dir: Path) -> tuple[dict[str, int], dict[str, list[Path]]]:
    """Get dataset sizes and parquet files.
    
    Args:
        train_dir: Path to training data directory
        
    Returns:
        Tuple of (dataset_sizes, dataset_files)
    """
    dataset_sizes = {}
    dataset_files = {}
    
    logger.info("Scanning datasets...")
    for dataset_dir in sorted(train_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        
        dataset_name = dataset_dir.name
        parquet_files = sorted(dataset_dir.glob("*.parquet"))
        
        if not parquet_files:
            continue
        
        # Compute total size
        total_size = sum(f.stat().st_size for f in parquet_files)
        
        dataset_sizes[dataset_name] = total_size
        dataset_files[dataset_name] = parquet_files
        
        logger.info(
            f"  {dataset_name}: {total_size / 1e9:.2f} GB, "
            f"{len(parquet_files)} files"
        )
    
    return dataset_sizes, dataset_files


def compute_sampling_probs(
    dataset_sizes: dict[str, int],
    temperature: float = 1.0
) -> dict[str, float]:
    """Compute sampling probabilities.
    
    Args:
        dataset_sizes: Dataset sizes in bytes
        temperature: Sampling temperature
            
    Returns:
        Sampling probabilities
    """
    if temperature == 0:
        n = len(dataset_sizes)
        return {name: 1.0 / n for name in dataset_sizes}
    
    powered = {name: size ** temperature for name, size in dataset_sizes.items()}
    total = sum(powered.values())
    probs = {name: p / total for name, p in powered.items()}
    
    logger.info("\nSampling probabilities:")
    for name, prob in sorted(probs.items(), key=lambda x: -x[1]):
        logger.info(f"  {name}: {prob:.4f}")
    
    return probs


def create_mixed_chunks_streaming(
    train_dir: Path,
    output_dir: Path,
    target_chunk_size_gb: float = 10.0,
    target_series_per_chunk: int = 200000,
    temperature: float = 1.0,
    seed: int = 42,
) -> None:
    """Create mixed chunks using streaming approach.
    
    Args:
        train_dir: Path to training data directory
        output_dir: Output directory for chunks
        target_chunk_size_gb: Target size per chunk in GB
        target_series_per_chunk: Target number of series per chunk
        temperature: Sampling temperature
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    
    logger.info(f"Creating mixed chunks from {train_dir}")
    logger.info(f"Target chunk size: {target_chunk_size_gb} GB")
    logger.info(f"Target series per chunk: {target_series_per_chunk:,}")
    logger.info(f"Temperature: {temperature}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get dataset info
    logger.info("\n" + "="*80)
    logger.info("Step 1: Analyzing datasets")
    logger.info("="*80)
    dataset_sizes, dataset_files = get_dataset_info(train_dir)
    
    total_size_gb = sum(dataset_sizes.values()) / 1e9
    logger.info(f"\nTotal size: {total_size_gb:.2f} GB")
    
    # Compute sampling probabilities
    sampling_probs = compute_sampling_probs(dataset_sizes, temperature)
    
    # Expected number of chunks
    num_chunks_estimate = int(np.ceil(total_size_gb / target_chunk_size_gb))
    logger.info(f"\nEstimated chunks: {num_chunks_estimate}")
    
    # Create sampler
    logger.info("\n" + "="*80)
    logger.info("Step 2: Initializing sampler")
    logger.info("="*80)
    sampler = DatasetSampler(dataset_files, sampling_probs, seed)
    
    # Create chunks
    logger.info("\n" + "="*80)
    logger.info("Step 3: Creating chunks")
    logger.info("="*80)
    
    chunk_idx = 0
    total_series_written = 0
    dataset_names = list(sampling_probs.keys())
    probs = [sampling_probs[name] for name in dataset_names]
    
    # Keep creating chunks until all data is processed
    while True:
        logger.info(f"\nCreating chunk {chunk_idx:04d}...")
        
        chunk_data = []
        dataset_counts = defaultdict(int)
        attempts_without_data = 0
        max_attempts = len(dataset_names) * 10
        
        # Sample series for this chunk
        for _ in range(target_series_per_chunk):
            # Sample which dataset to use
            dataset_name = np.random.choice(dataset_names, p=probs)
            
            # Try to get a series from that dataset
            series = sampler.sample_series(dataset_name)
            
            if series is not None:
                chunk_data.append(series)
                dataset_counts[dataset_name] += 1
                attempts_without_data = 0
            else:
                attempts_without_data += 1
                
                # If we've failed too many times, all datasets might be exhausted
                if attempts_without_data >= max_attempts:
                    logger.info("  All datasets appear to be exhausted")
                    break
        
        # Check if we got any data
        if not chunk_data:
            logger.info("No more data available, stopping")
            break
        
        # Shuffle chunk data
        random.shuffle(chunk_data)
        
        # Create DataFrame and write
        chunk_df = pd.DataFrame(chunk_data)
        output_file = output_dir / f"train_chunk_{chunk_idx:04d}.parquet"
        chunk_df.to_parquet(output_file, index=False, compression='snappy')
        
        actual_size = output_file.stat().st_size / 1e9
        total_series_written += len(chunk_data)
        
        logger.info(f"  ✓ Chunk {chunk_idx:04d} complete:")
        logger.info(f"    Size: {actual_size:.2f} GB")
        logger.info(f"    Series: {len(chunk_data):,}")
        logger.info("    Composition:")
        for name in sorted(dataset_counts.keys()):
            count = dataset_counts[name]
            percentage = 100 * count / len(chunk_data)
            logger.info(f"      {name}: {count:,} ({percentage:.1f}%)")
        
        chunk_idx += 1
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Total chunks created: {chunk_idx}")
    logger.info(f"Total series written: {total_series_written:,}")
    logger.info(f"Output directory: {output_dir}")
    
    # List all chunks
    logger.info("\nChunk files:")
    total_output_size = 0
    for chunk_file in sorted(output_dir.glob("train_chunk_*.parquet")):
        size = chunk_file.stat().st_size / 1e9
        total_output_size += size
        logger.info(f"  {chunk_file.name}: {size:.2f} GB")
    
    logger.info(f"\nTotal output size: {total_output_size:.2f} GB")
    logger.info("="*80)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create mixed-dataset training chunks (streaming)"
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("data/splits/train"),
        help="Path to training data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/chunks/train"),
        help="Path to output directory"
    )
    parser.add_argument(
        "--chunk-size",
        type=float,
        default=10.0,
        help="Target chunk size in GB"
    )
    parser.add_argument(
        "--series-per-chunk",
        type=int,
        default=200000,
        help="Target series per chunk (adjust based on your data)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature: 1.0=proportional, 0.5=sqrt, 0.0=uniform"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    create_mixed_chunks_streaming(
        train_dir=args.train_dir,
        output_dir=args.output_dir,
        target_chunk_size_gb=args.chunk_size,
        target_series_per_chunk=args.series_per_chunk,
        temperature=args.temperature,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()


