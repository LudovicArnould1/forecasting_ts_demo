"""Missing data pattern analysis across datasets.

This script analyzes the presence, patterns, and characteristics of missing
values to inform handling strategies.
"""

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")


def load_sample_series(
    file_path: Path, max_series: int = 100, max_length: int = 10000
) -> dict[str, Any]:
    """Load a sample of time series from a parquet file.

    Args:
        file_path: Path to parquet file.
        max_series: Maximum number of series to load.
        max_length: Maximum length per series to keep.

    Returns:
        Dictionary with metadata and sampled series.
    """
    table = pq.read_table(file_path)
    df = table.to_pandas()

    if len(df) > max_series:
        df = df.sample(n=max_series, random_state=42)

    series_data = []
    for _, row in df.iterrows():
        target = row["target"]
        if len(target) > 0:
            ts = np.array(target[0], dtype=np.float32)
            if len(ts) > max_length:
                ts = ts[:max_length]
            series_data.append(ts)

    return {
        "series": series_data,
        "freq": df["freq"].iloc[0] if len(df) > 0 else "unknown",
        "num_series": len(series_data),
    }


def scan_datasets(base_path: Path, split: str = "training") -> dict[str, dict[str, Any]]:
    """Scan and sample datasets from directory.

    Args:
        base_path: Root data directory.
        split: Either 'training' or 'test'.

    Returns:
        Dictionary mapping dataset names to sampled data.
    """
    datasets = {}
    split_dir = base_path / split

    dataset_dirs = {}
    for parquet_file in split_dir.rglob("*.parquet"):
        relative_path = parquet_file.relative_to(split_dir)
        dataset_name = str(relative_path.parent)

        if dataset_name not in dataset_dirs:
            dataset_dirs[dataset_name] = parquet_file.parent

    for dataset_name, dataset_dir in sorted(dataset_dirs.items()):
        parquet_files = sorted(dataset_dir.glob("*.parquet"))
        if parquet_files:
            print(f"   Loading {dataset_name}...")
            data = load_sample_series(parquet_files[0])
            datasets[dataset_name] = data

    return datasets


def analyze_missing_data(datasets: dict[str, dict[str, Any]]) -> dict[str, dict]:
    """Analyze missing data patterns.

    Args:
        datasets: Dictionary of dataset names to data.

    Returns:
        Dictionary of missing data statistics.
    """
    missing_stats = {}

    for name, data in datasets.items():
        total_values = 0
        missing_values = 0
        max_consecutive = 0
        gaps = []

        for ts in data["series"]:
            total_values += len(ts)
            nan_mask = np.isnan(ts)
            missing_values += nan_mask.sum()

            # Find consecutive gaps
            if nan_mask.any():
                consecutive = 0
                for is_nan in nan_mask:
                    if is_nan:
                        consecutive += 1
                    else:
                        if consecutive > 0:
                            gaps.append(consecutive)
                            max_consecutive = max(max_consecutive, consecutive)
                        consecutive = 0
                if consecutive > 0:
                    gaps.append(consecutive)

        missing_stats[name] = {
            "missing_pct": (missing_values / total_values * 100) if total_values > 0 else 0,
            "max_consecutive": max_consecutive,
            "avg_gap_length": np.mean(gaps) if gaps else 0,
            "num_gaps": len(gaps),
        }

    return missing_stats


def print_missing_analysis(
    train_stats: dict[str, dict], eval_stats: dict[str, dict]
) -> None:
    """Print missing data analysis.

    Args:
        train_stats: Training dataset statistics.
        eval_stats: Evaluation dataset statistics.
    """
    print("Training Datasets:")
    print(f"{'Dataset':<40} {'Missing %':<12} {'Max Gap':<12} {'# Gaps':<10}")
    print("-" * 80)

    for name in sorted(train_stats.keys()):
        stats = train_stats[name]
        print(
            f"{name:<40} {stats['missing_pct']:>10.2f}% {stats['max_consecutive']:>10} "
            f"{stats['num_gaps']:>10}"
        )

    print("\n\nEvaluation Datasets:")
    print(f"{'Dataset':<40} {'Missing %':<12} {'Max Gap':<12} {'# Gaps':<10}")
    print("-" * 80)

    for name in sorted(eval_stats.keys()):
        stats = eval_stats[name]
        print(
            f"{name:<40} {stats['missing_pct']:>10.2f}% {stats['max_consecutive']:>10} "
            f"{stats['num_gaps']:>10}"
        )


def print_key_insights(train_stats: dict[str, dict], eval_stats: dict[str, dict]) -> None:
    """Print key insights about missing data.

    Args:
        train_stats: Training dataset statistics.
        eval_stats: Evaluation dataset statistics.
    """
    all_stats = {**train_stats, **eval_stats}

    # Find datasets with significant missing data
    high_missing = [
        (name, stats["missing_pct"])
        for name, stats in all_stats.items()
        if stats["missing_pct"] > 5
    ]

    print("\n\n🔍 Key Insights:")
    if high_missing:
        print(f"\n   Datasets with >5% missing data:")
        for name, pct in sorted(high_missing, key=lambda x: x[1], reverse=True):
            print(f"      {name}: {pct:.2f}%")
    else:
        print("\n   ✓ All datasets have <5% missing data")

    # Check for large gaps
    large_gaps = [
        (name, stats["max_consecutive"])
        for name, stats in all_stats.items()
        if stats["max_consecutive"] > 100
    ]

    if large_gaps:
        print(f"\n   Datasets with gaps >100 timesteps:")
        for name, gap in sorted(large_gaps, key=lambda x: x[1], reverse=True):
            print(f"      {name}: {gap} timesteps")


def main() -> None:
    """Main analysis workflow."""
    data_dir = Path("data")

    print("\n" + "=" * 80)
    print(" MISSING DATA ANALYSIS ".center(80, "="))
    print("=" * 80 + "\n")

    print("📂 Loading training datasets...")
    train_datasets = scan_datasets(data_dir, "training")

    print("\n📂 Loading evaluation datasets...")
    eval_datasets = scan_datasets(data_dir, "test")

    print("\n📊 Analyzing missing data patterns...")
    train_stats = analyze_missing_data(train_datasets)
    eval_stats = analyze_missing_data(eval_datasets)

    print("\n")
    print_missing_analysis(train_stats, eval_stats)
    print_key_insights(train_stats, eval_stats)

    print("\n" + "=" * 80)
    print(" ANALYSIS COMPLETE ".center(80, "="))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

