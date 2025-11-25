"""Create train/validation/test splits for time series data.

This script performs temporal splitting of time series data with a 70/15/15 ratio.
For each time series, the first 70% of timesteps go to training, the next 15% to
validation, and the final 15% to testing. This approach is appropriate for time
series forecasting as it preserves temporal order and avoids data leakage.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def split_time_series(
    target: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    min_points_per_split: int = 10,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Split a single time series into train/val/test sets temporally.

    Args:
        target: Time series array to split.
        train_ratio: Proportion of data for training.
        val_ratio: Proportion of data for validation.
        test_ratio: Proportion of data for testing.
        min_points_per_split: Minimum number of points required in each split.

    Returns:
        Tuple of (train, val, test) arrays. Returns (None, None, None) if
        the series is too short.

    Raises:
        ValueError: If ratios don't sum to 1.0.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )

    length = len(target)

    # Check minimum length requirement
    min_length = min_points_per_split * 3  # At least min_points in each split
    if length < min_length:
        return None, None, None

    # Calculate split points
    train_end = int(length * train_ratio)
    val_end = int(length * (train_ratio + val_ratio))

    # Ensure each split has at least min_points_per_split
    if (
        train_end < min_points_per_split
        or (val_end - train_end) < min_points_per_split
        or (length - val_end) < min_points_per_split
    ):
        return None, None, None

    # Split the series
    train = target[:train_end]
    val = target[train_end:val_end]
    test = target[val_end:]

    return train, val, test


def process_parquet_file(
    input_path: Path,
    output_base_dir: Path,
    dataset_name: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    min_points_per_split: int = 10,
) -> dict[str, int]:
    """Process a single parquet file and create splits.

    Args:
        input_path: Path to input parquet file.
        output_base_dir: Base directory for output files (should be data/splits).
        dataset_name: Name of the dataset being processed.
        train_ratio: Proportion of data for training.
        val_ratio: Proportion of data for validation.
        test_ratio: Proportion of data for testing.
        min_points_per_split: Minimum number of points required in each split.

    Returns:
        Dictionary with statistics about the processing.
    """
    # Read parquet in batches to avoid "List index overflow" with large nested arrays
    parquet_file = pq.ParquetFile(input_path)
    
    train_rows = []
    val_rows = []
    test_rows = []
    skipped = 0

    # Process in batches
    for batch in parquet_file.iter_batches(batch_size=1000):
        df = batch.to_pandas()
        
        for idx, row in df.iterrows():
            # Extract the target time series
            target = row["target"][0] if len(row["target"]) > 0 else np.array([])

            if len(target) == 0:
                skipped += 1
                continue

            # Split the time series
            train_split, val_split, test_split = split_time_series(
                target,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                min_points_per_split=min_points_per_split,
            )

            if train_split is None:
                skipped += 1
                continue

            # Create new rows for each split
            # Convert row to dict to avoid Series issues
            row_dict = row.to_dict()
            
            # For train split
            train_row = row_dict.copy()
            train_row["target"] = [[float(x) for x in train_split]]
            train_rows.append(train_row)

            # For val split
            val_row = row_dict.copy()
            val_row["target"] = [[float(x) for x in val_split]]
            val_rows.append(val_row)

            # For test split
            test_row = row_dict.copy()
            test_row["target"] = [[float(x) for x in test_split]]
            test_rows.append(test_row)

    # Save the splits to parquet files
    stats = {
        "total_series": len(train_rows) + skipped,
        "processed_series": len(train_rows),
        "skipped_series": skipped,
    }

    if train_rows:
        # Get relative path structure for organizing outputs
        train_df = pd.DataFrame(train_rows)
        val_df = pd.DataFrame(val_rows)
        test_df = pd.DataFrame(test_rows)

        # Save each split
        for split_type, split_df in [
            ("train", train_df),
            ("val", val_df),
            ("test", test_df),
        ]:
            output_path = output_base_dir / split_type / dataset_name / input_path.name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            split_df.to_parquet(output_path, index=False)

        stats["train_length_mean"] = (
            train_df["target"].apply(lambda x: len(x[0]) if len(x) > 0 else 0).mean()
        )
        stats["val_length_mean"] = (
            val_df["target"].apply(lambda x: len(x[0]) if len(x) > 0 else 0).mean()
        )
        stats["test_length_mean"] = (
            test_df["target"].apply(lambda x: len(x[0]) if len(x) > 0 else 0).mean()
        )

    return stats


def process_dataset(
    dataset_dir: Path,
    output_base_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    min_points_per_split: int = 10,
) -> dict[str, Any]:
    """Process all parquet files in a dataset directory.

    Args:
        dataset_dir: Directory containing parquet files for a dataset.
        output_base_dir: Base directory for output files.
        train_ratio: Proportion of data for training.
        val_ratio: Proportion of data for validation.
        test_ratio: Proportion of data for testing.
        min_points_per_split: Minimum number of points required in each split.

    Returns:
        Dictionary with aggregated statistics.
    """
    parquet_files = sorted(dataset_dir.glob("*.parquet"))

    if not parquet_files:
        logger.warning(f"No parquet files found in {dataset_dir}")
        return {}

    dataset_name = dataset_dir.name
    logger.info(f"Processing dataset: {dataset_name} ({len(parquet_files)} files)")

    total_stats = {
        "dataset_name": dataset_name,
        "num_files": len(parquet_files),
        "total_series": 0,
        "processed_series": 0,
        "skipped_series": 0,
        "train_lengths": [],
        "val_lengths": [],
        "test_lengths": [],
    }

    for parquet_file in tqdm(parquet_files, desc=f"  {dataset_name}"):
        file_stats = process_parquet_file(
            parquet_file,
            output_base_dir,
            dataset_name=dataset_name,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            min_points_per_split=min_points_per_split,
        )

        total_stats["total_series"] += file_stats.get("total_series", 0)
        total_stats["processed_series"] += file_stats.get("processed_series", 0)
        total_stats["skipped_series"] += file_stats.get("skipped_series", 0)

        if "train_length_mean" in file_stats:
            total_stats["train_lengths"].append(file_stats["train_length_mean"])
            total_stats["val_lengths"].append(file_stats["val_length_mean"])
            total_stats["test_lengths"].append(file_stats["test_length_mean"])

    # Compute aggregated statistics
    if total_stats["train_lengths"]:
        total_stats["avg_train_length"] = np.mean(total_stats["train_lengths"])
        total_stats["avg_val_length"] = np.mean(total_stats["val_lengths"])
        total_stats["avg_test_length"] = np.mean(total_stats["test_lengths"])

    return total_stats


def create_splits(
    input_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    min_points_per_split: int = 10,
) -> None:
    """Create train/val/test splits for all datasets.

    Args:
        input_dir: Directory containing training data organized by dataset.
        output_dir: Directory where splits will be saved.
        train_ratio: Proportion of data for training.
        val_ratio: Proportion of data for validation.
        test_ratio: Proportion of data for testing.
        min_points_per_split: Minimum number of points required in each split.
    """
    logger.info("=" * 80)
    logger.info("Creating Train/Val/Test Splits for Time Series Data")
    logger.info("=" * 80)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(
        f"Split ratios: train={train_ratio:.0%}, val={val_ratio:.0%}, "
        f"test={test_ratio:.0%}"
    )
    logger.info(f"Minimum points per split: {min_points_per_split}")
    logger.info("=" * 80)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all dataset directories
    dataset_dirs = [d for d in input_dir.iterdir() if d.is_dir()]

    if not dataset_dirs:
        logger.error(f"No dataset directories found in {input_dir}")
        return

    logger.info(f"\nFound {len(dataset_dirs)} datasets to process\n")

    all_stats = []

    for dataset_dir in sorted(dataset_dirs):
        stats = process_dataset(
            dataset_dir,
            output_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            min_points_per_split=min_points_per_split,
        )
        if stats:
            all_stats.append(stats)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    total_processed = sum(s["processed_series"] for s in all_stats)
    total_skipped = sum(s["skipped_series"] for s in all_stats)
    total_series = sum(s["total_series"] for s in all_stats)

    logger.info(f"\nTotal series: {total_series:,}")
    logger.info(f"Successfully split: {total_processed:,} ({total_processed/total_series*100:.1f}%)")
    logger.info(f"Skipped (too short): {total_skipped:,} ({total_skipped/total_series*100:.1f}%)")

    logger.info("\nPer-dataset statistics:")
    logger.info("-" * 80)
    logger.info(
        f"{'Dataset':<30} {'Files':>8} {'Series':>10} {'Processed':>10} {'Skipped':>10}"
    )
    logger.info("-" * 80)

    for stats in all_stats:
        logger.info(
            f"{stats['dataset_name']:<30} "
            f"{stats['num_files']:>8} "
            f"{stats['total_series']:>10,} "
            f"{stats['processed_series']:>10,} "
            f"{stats['skipped_series']:>10,}"
        )

    logger.info("\nAverage lengths per split:")
    logger.info("-" * 80)
    logger.info(f"{'Dataset':<30} {'Train':>12} {'Val':>12} {'Test':>12}")
    logger.info("-" * 80)

    for stats in all_stats:
        if "avg_train_length" in stats:
            logger.info(
                f"{stats['dataset_name']:<30} "
                f"{stats['avg_train_length']:>12.1f} "
                f"{stats['avg_val_length']:>12.1f} "
                f"{stats['avg_test_length']:>12.1f}"
            )

    logger.info("\n" + "=" * 80)
    logger.info(f"Splits saved to: {output_dir}")
    logger.info("=" * 80)


def main() -> None:
    """Main entry point."""
    # Set paths
    project_root = Path(__file__).parent.parent
    input_dir = project_root / "data" / "training"
    output_dir = project_root / "data" / "splits"

    # Create splits with default 70/15/15 ratio
    create_splits(
        input_dir=input_dir,
        output_dir=output_dir,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        min_points_per_split=10,
    )


if __name__ == "__main__":
    main()
