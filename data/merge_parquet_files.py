"""Merge small parquet files into larger chunks (~10GB each) per dataset.

This script consolidates many small parquet files into fewer, larger files
to improve I/O efficiency during training. Files are grouped by dataset
and merged into chunks of approximately 10GB each.
"""

import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Target chunk size in bytes (10 GB)
TARGET_CHUNK_SIZE = 10 * 1024**3  # 10 GB
# Threshold below which we don't split (if total size < this, make 1 file)
MIN_SPLIT_THRESHOLD = 2 * 1024**3  # 2 GB


def get_file_size(file_path: Path) -> int:
    """Get file size in bytes.

    Args:
        file_path: Path to file.

    Returns:
        File size in bytes.
    """
    return file_path.stat().st_size


def merge_parquet_files(
    input_files: list[Path],
    output_file: Path,
) -> dict[str, Any]:
    """Merge multiple parquet files into a single file.

    Args:
        input_files: List of input parquet file paths.
        output_file: Output parquet file path.

    Returns:
        Dictionary with merge statistics.
    """
    logger.info(f"Merging {len(input_files)} files into {output_file.name}")

    # Read and concatenate all files
    dfs = []
    total_rows = 0

    for file_path in tqdm(input_files, desc=f"  Reading files", leave=False):
        df = pd.read_parquet(file_path)
        dfs.append(df)
        total_rows += len(df)

    # Concatenate all dataframes
    logger.info(f"  Concatenating {len(dfs)} dataframes...")
    merged_df = pd.concat(dfs, ignore_index=True)

    # Write merged file
    logger.info(f"  Writing merged file ({total_rows:,} rows)...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(output_file, index=False)

    output_size = get_file_size(output_file)

    return {
        "num_input_files": len(input_files),
        "total_rows": total_rows,
        "output_size_gb": output_size / (1024**3),
    }


def merge_dataset(
    dataset_dir: Path,
    output_dir: Path,
    target_chunk_size: int = TARGET_CHUNK_SIZE,
    min_split_threshold: int = MIN_SPLIT_THRESHOLD,
) -> dict[str, Any]:
    """Merge parquet files for a single dataset into chunks.

    Args:
        dataset_dir: Directory containing parquet files for the dataset.
        output_dir: Output directory for merged files.
        target_chunk_size: Target size for each chunk in bytes.
        min_split_threshold: If total size is below this, create single file.

    Returns:
        Dictionary with merge statistics.
    """
    dataset_name = dataset_dir.name
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing dataset: {dataset_name}")
    logger.info(f"{'='*80}")

    # Get all parquet files sorted by name
    parquet_files = sorted(dataset_dir.glob("*.parquet"))

    if not parquet_files:
        logger.warning(f"No parquet files found in {dataset_dir}")
        return {}

    # Calculate file sizes
    file_sizes = [(f, get_file_size(f)) for f in parquet_files]
    total_size = sum(size for _, size in file_sizes)
    total_size_gb = total_size / (1024**3)

    logger.info(f"Found {len(parquet_files)} files, total size: {total_size_gb:.2f} GB")

    # Determine chunking strategy
    if total_size < min_split_threshold:
        logger.info(
            f"Dataset size < {min_split_threshold/(1024**3):.1f} GB, "
            f"creating single merged file"
        )
        num_chunks = 1
    else:
        num_chunks = max(1, int(total_size / target_chunk_size) + 1)
        logger.info(f"Creating {num_chunks} chunks of ~10 GB each")

    # Group files into chunks
    chunks = []
    current_chunk = []
    current_chunk_size = 0

    for file_path, file_size in file_sizes:
        current_chunk.append(file_path)
        current_chunk_size += file_size

        # Check if we should start a new chunk
        if len(chunks) < num_chunks - 1 and current_chunk_size >= target_chunk_size:
            chunks.append(current_chunk)
            current_chunk = []
            current_chunk_size = 0

    # Add remaining files to last chunk
    if current_chunk:
        chunks.append(current_chunk)

    # Merge each chunk
    output_dataset_dir = output_dir / dataset_name
    merge_stats = []

    for i, chunk_files in enumerate(chunks):
        chunk_name = f"{dataset_name}_chunk_{i:04d}.parquet"
        output_file = output_dataset_dir / chunk_name

        stats = merge_parquet_files(chunk_files, output_file)
        stats["chunk_index"] = i
        merge_stats.append(stats)

    # Summary
    total_output_size = sum(s["output_size_gb"] for s in merge_stats)
    logger.info(f"\n✓ Merged {len(parquet_files)} files into {len(chunks)} chunks")
    logger.info(f"  Input size: {total_size_gb:.2f} GB")
    logger.info(f"  Output size: {total_output_size:.2f} GB")

    return {
        "dataset_name": dataset_name,
        "num_input_files": len(parquet_files),
        "num_output_chunks": len(chunks),
        "input_size_gb": total_size_gb,
        "output_size_gb": total_output_size,
        "chunks": merge_stats,
    }


def merge_all_datasets(
    input_dir: Path,
    output_dir: Path,
    target_chunk_size: int = TARGET_CHUNK_SIZE,
    min_split_threshold: int = MIN_SPLIT_THRESHOLD,
) -> None:
    """Merge parquet files for all datasets.

    Args:
        input_dir: Directory containing dataset subdirectories.
        output_dir: Output directory for merged files.
        target_chunk_size: Target size for each chunk in bytes.
        min_split_threshold: If total size is below this, create single file.
    """
    logger.info("="*80)
    logger.info("MERGING PARQUET FILES INTO CHUNKS")
    logger.info("="*80)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Target chunk size: {target_chunk_size/(1024**3):.1f} GB")
    logger.info(f"Min split threshold: {min_split_threshold/(1024**3):.1f} GB")
    logger.info("="*80)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all dataset directories
    dataset_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])

    if not dataset_dirs:
        logger.error(f"No dataset directories found in {input_dir}")
        return

    logger.info(f"\nFound {len(dataset_dirs)} datasets to process\n")

    all_stats = []

    for dataset_dir in dataset_dirs:
        try:
            stats = merge_dataset(
                dataset_dir,
                output_dir,
                target_chunk_size=target_chunk_size,
                min_split_threshold=min_split_threshold,
            )
            if stats:
                all_stats.append(stats)
        except Exception as e:
            logger.error(f"Error processing {dataset_dir.name}: {e}", exc_info=True)
            continue

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)

    total_input_files = sum(s["num_input_files"] for s in all_stats)
    total_output_chunks = sum(s["num_output_chunks"] for s in all_stats)
    total_input_size = sum(s["input_size_gb"] for s in all_stats)
    total_output_size = sum(s["output_size_gb"] for s in all_stats)

    logger.info(f"\nOverall:")
    logger.info(f"  Total input files: {total_input_files:,}")
    logger.info(f"  Total output chunks: {total_output_chunks:,}")
    logger.info(f"  Total input size: {total_input_size:.2f} GB")
    logger.info(f"  Total output size: {total_output_size:.2f} GB")

    logger.info("\nPer-dataset summary:")
    logger.info("-"*80)
    logger.info(
        f"{'Dataset':<35} {'Input Files':>12} {'Output Chunks':>14} {'Size (GB)':>12}"
    )
    logger.info("-"*80)

    for stats in all_stats:
        logger.info(
            f"{stats['dataset_name']:<35} "
            f"{stats['num_input_files']:>12,} "
            f"{stats['num_output_chunks']:>14} "
            f"{stats['output_size_gb']:>12.2f}"
        )

    logger.info("\n" + "="*80)
    logger.info(f"Merged files saved to: {output_dir}")
    logger.info("="*80)


def main() -> None:
    """Main entry point."""
    # Set paths
    project_root = Path(__file__).parent.parent
    input_dir = project_root / "data" / "training"
    output_dir = project_root / "data" / "training_merged"

    # Check if output directory exists
    if output_dir.exists():
        logger.warning(f"\nOutput directory already exists: {output_dir}")
        response = input("Do you want to overwrite it? (yes/no): ").strip().lower()
        if response != "yes":
            logger.info("Aborted by user")
            return
        logger.info("Removing existing output directory...")
        shutil.rmtree(output_dir)

    # Merge all datasets
    merge_all_datasets(
        input_dir=input_dir,
        output_dir=output_dir,
        target_chunk_size=TARGET_CHUNK_SIZE,
        min_split_threshold=MIN_SPLIT_THRESHOLD,
    )


if __name__ == "__main__":
    main()


