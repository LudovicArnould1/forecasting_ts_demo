"""Analyze temporal relationships between time series within datasets.

This script examines:
1. How start times relate across different series in the same dataset
2. Whether series are consecutive, overlapping, or have gaps
3. Whether each series covers the full time span between consecutive starts
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_frequency(freq: str) -> pd.Timedelta:
    """Parse frequency string to timedelta.

    Args:
        freq: Frequency string (e.g., '1H', '15T', '1D', 'H', 'D').

    Returns:
        Pandas Timedelta object.
    """
    # Normalize deprecated frequency codes
    freq_map = {
        'T': 'min',
        'H': 'h',
        'D': 'd',
        'W': 'w',
        'M': 'ME',  # Month end
        'Y': 'YE',  # Year end
    }
    
    # Replace single-letter codes with modern equivalents
    if freq in freq_map:
        freq = '1' + freq_map[freq]
    # Also handle patterns like '15T' -> '15min'
    for old_code, new_code in freq_map.items():
        if old_code in freq and not freq.isalpha():
            freq = freq.replace(old_code, new_code)
            break
    
    try:
        return pd.Timedelta(freq)
    except (ValueError, AttributeError):
        # For complex frequencies, use offset and convert to timedelta
        offset = pd.tseries.frequencies.to_offset(freq)
        # Convert to nanoseconds then to Timedelta
        return pd.Timedelta(offset.nanos, unit='ns')


def analyze_dataset_temporal_structure(
    dataset_path: Path, max_series: int = 1000
) -> dict[str, Any]:
    """Analyze temporal relationships between series in a dataset.

    Args:
        dataset_path: Path to dataset directory containing parquet files.
        max_series: Maximum number of series to analyze per dataset.

    Returns:
        Dictionary with analysis results.
    """
    parquet_files = sorted(dataset_path.glob("*.parquet"))
    if not parquet_files:
        return {}

    # Load first parquet file and sample series
    df = pd.read_parquet(parquet_files[0])
    df = df.head(max_series)

    results = {
        "dataset_name": dataset_path.name,
        "num_series_analyzed": len(df),
        "frequencies": [],
        "start_times": [],
        "end_times": [],
        "series_lengths": [],
    }

    # Extract temporal information for each series
    for idx, row in df.iterrows():
        target = row["target"][0] if len(row["target"]) > 0 else np.array([])
        if len(target) == 0:
            continue

        start = pd.Timestamp(row["start"])
        freq = parse_frequency(row["freq"])
        length = len(target)
        end = start + (length - 1) * freq

        results["start_times"].append(start)
        results["end_times"].append(end)
        results["series_lengths"].append(length)
        results["frequencies"].append(freq)

    if not results["start_times"]:
        return results

    # Convert to arrays for analysis
    start_times = np.array(results["start_times"])
    end_times = np.array(results["end_times"])
    frequencies = results["frequencies"]

    # Sort by start time
    sort_idx = np.argsort(start_times)
    start_times = start_times[sort_idx]
    end_times = end_times[sort_idx]

    # Analyze relationships between consecutive series
    results["unique_frequencies"] = list(set(str(f) for f in frequencies))
    results["num_unique_frequencies"] = len(results["unique_frequencies"])

    # Check if all series have the same frequency
    results["uniform_frequency"] = results["num_unique_frequencies"] == 1

    # Analyze start time patterns
    if len(start_times) > 1:
        # Check if all start times are identical
        results["all_same_start"] = len(set(start_times)) == 1

        if not results["all_same_start"]:
            # Calculate gaps between consecutive start times
            start_gaps = np.diff(start_times)
            results["min_start_gap"] = str(np.min(start_gaps))
            results["max_start_gap"] = str(np.max(start_gaps))
            results["median_start_gap"] = str(np.median(start_gaps))

            # Check if start times are evenly spaced
            unique_gaps = set(start_gaps)
            results["evenly_spaced_starts"] = len(unique_gaps) == 1
            results["num_unique_start_gaps"] = len(unique_gaps)

            # Check for overlaps
            overlaps = 0
            gaps = 0
            consecutive = 0

            for i in range(len(start_times) - 1):
                gap_between = start_times[i + 1] - end_times[i]

                if gap_between < pd.Timedelta(0):
                    overlaps += 1
                elif gap_between == frequencies[sort_idx[i]]:
                    consecutive += 1
                else:
                    gaps += 1

            results["num_overlapping"] = overlaps
            results["num_consecutive"] = consecutive
            results["num_with_gaps"] = gaps
            results["pct_overlapping"] = (
                overlaps / (len(start_times) - 1) * 100 if len(start_times) > 1 else 0
            )
            results["pct_consecutive"] = (
                consecutive / (len(start_times) - 1) * 100
                if len(start_times) > 1
                else 0
            )
            results["pct_with_gaps"] = (
                gaps / (len(start_times) - 1) * 100 if len(start_times) > 1 else 0
            )
        else:
            results["all_same_start"] = True
            results["interpretation"] = (
                "All series start at the same time (likely different entities/locations)"
            )
    else:
        results["all_same_start"] = None

    # Time span coverage
    results["earliest_start"] = str(start_times[0])
    results["latest_end"] = str(end_times[-1])
    results["total_time_span"] = str(end_times[-1] - start_times[0])

    # Average series duration
    durations = [end - start for start, end in zip(start_times, end_times)]
    # Convert to total seconds to avoid overflow with large timedeltas
    duration_seconds = [d.total_seconds() for d in durations]
    results["mean_series_duration"] = str(
        pd.Timedelta(seconds=np.mean(duration_seconds))
    )
    results["median_series_duration"] = str(
        pd.Timedelta(seconds=np.median(duration_seconds))
    )

    return results


def print_dataset_summary(results: dict[str, Any]) -> None:
    """Print a formatted summary of temporal structure analysis.

    Args:
        results: Analysis results dictionary.
    """
    print(f"\n{'='*80}")
    print(f"Dataset: {results['dataset_name']}")
    print(f"{'='*80}")

    print(f"\nBasic Info:")
    print(f"  Series analyzed: {results['num_series_analyzed']:,}")
    print(f"  Frequencies: {', '.join(results.get('unique_frequencies', []))}")
    print(f"  Uniform frequency: {results.get('uniform_frequency', 'N/A')}")

    print(f"\nTemporal Coverage:")
    print(f"  Earliest start: {results.get('earliest_start', 'N/A')}")
    print(f"  Latest end: {results.get('latest_end', 'N/A')}")
    print(f"  Total span: {results.get('total_time_span', 'N/A')}")
    print(f"  Mean series duration: {results.get('mean_series_duration', 'N/A')}")

    if results.get("all_same_start"):
        print(f"\n📊 Start Time Pattern:")
        print(f"  ✓ All series start at the SAME time")
        print(
            f"  → Interpretation: Different entities observed simultaneously"
        )
        print(f"    (e.g., multiple sensors, locations, or products at same time)")
    elif results.get("all_same_start") is False:
        print(f"\n📊 Start Time Pattern:")
        print(f"  ✓ Series have DIFFERENT start times")
        print(f"  Min gap between starts: {results.get('min_start_gap', 'N/A')}")
        print(f"  Median gap: {results.get('median_start_gap', 'N/A')}")
        print(f"  Max gap: {results.get('max_start_gap', 'N/A')}")

        if results.get("evenly_spaced_starts"):
            print(f"  ✓ Start times are EVENLY SPACED")
        else:
            print(f"  ✗ Start times are IRREGULARLY SPACED")
            print(
                f"    ({results.get('num_unique_start_gaps', 0)} different gap sizes)"
            )

        print(f"\n📊 Series Relationships (consecutive pairs):")
        print(
            f"  Overlapping: {results.get('num_overlapping', 0)} "
            f"({results.get('pct_overlapping', 0):.1f}%)"
        )
        print(
            f"  Consecutive: {results.get('num_consecutive', 0)} "
            f"({results.get('pct_consecutive', 0):.1f}%)"
        )
        print(
            f"  With gaps: {results.get('num_with_gaps', 0)} "
            f"({results.get('pct_with_gaps', 0):.1f}%)"
        )

        # Interpretation
        print(f"\n💡 Interpretation:")
        if results.get("pct_overlapping", 0) > 50:
            print(
                f"  → Series OVERLAP significantly (same time periods, different entities)"
            )
        elif results.get("pct_consecutive", 0) > 50:
            print(f"  → Series are CONSECUTIVE (end-to-end coverage)")
        elif results.get("pct_with_gaps", 0) > 50:
            print(f"  → Series have GAPS between them (sparse temporal coverage)")
        else:
            print(f"  → Mixed pattern of overlaps, gaps, and consecutive series")


def main() -> None:
    """Analyze temporal structure of all datasets."""
    data_dir = Path(__file__).parent.parent / "training"

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return

    dataset_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    if not dataset_dirs:
        print(f"No dataset directories found in {data_dir}")
        return

    print("="*80)
    print("TEMPORAL RELATIONSHIPS ANALYSIS")
    print("="*80)
    print(f"\nAnalyzing {len(dataset_dirs)} datasets...\n")

    all_results = []

    for dataset_dir in tqdm(dataset_dirs, desc="Processing datasets"):
        results = analyze_dataset_temporal_structure(dataset_dir, max_series=1000)
        if results:
            all_results.append(results)

    # Print individual summaries
    for results in all_results:
        print_dataset_summary(results)

    # Print overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}\n")

    print(f"{'Dataset':<35} {'Start Pattern':<25} {'Relationship':<20}")
    print(f"{'-'*80}")

    for results in all_results:
        dataset = results["dataset_name"][:34]

        if results.get("all_same_start"):
            start_pattern = "Same start time"
            relationship = "Parallel series"
        elif results.get("all_same_start") is False:
            if results.get("evenly_spaced_starts"):
                start_pattern = "Evenly spaced"
            else:
                start_pattern = "Irregular spacing"

            if results.get("pct_overlapping", 0) > 50:
                relationship = "Overlapping"
            elif results.get("pct_consecutive", 0) > 50:
                relationship = "Consecutive"
            else:
                relationship = "Mixed/Gaps"
        else:
            start_pattern = "N/A"
            relationship = "N/A"

        print(f"{dataset:<35} {start_pattern:<25} {relationship:<20}")

    print("\n" + "="*80)
    print("KEY INSIGHTS FOR MODEL DESIGN")
    print("="*80)
    print("""
1. **Same Start Time Pattern**:
   - Multiple entities observed simultaneously (e.g., sensors, stores, products)
   - Each series is an independent entity with its own dynamics
   - Model should handle multi-entity forecasting

2. **Different Start Times**:
   - Could be sequential measurements or different observation periods
   - Check if consecutive (end-to-end) or with gaps

3. **Overlapping Series**:
   - Different entities/channels measured during overlapping periods
   - Each series represents a different dimension/entity

4. **Consecutive Series**:
   - Could be a single long time series split into chunks
   - Or sequential observations with continuous coverage

5. **Gaps Between Series**:
   - Sparse temporal coverage
   - Missing data or intentional sampling gaps
    """)


if __name__ == "__main__":
    main()

