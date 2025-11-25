"""Value statistics and distribution analysis across datasets.

This script analyzes value ranges, scales, distributions, and variability
to inform normalization and preprocessing decisions.
"""

import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid")


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


def analyze_value_statistics(datasets: dict[str, dict[str, Any]]) -> dict[str, dict]:
    """Compute value statistics for each dataset.

    Args:
        datasets: Dictionary of dataset names to data.

    Returns:
        Dictionary of statistics per dataset.
    """
    stats = {}

    for name, data in datasets.items():
        all_values = np.concatenate([ts[~np.isnan(ts)] for ts in data["series"]])

        if len(all_values) == 0:
            continue

        stats[name] = {
            "mean": np.mean(all_values),
            "std": np.std(all_values),
            "min": np.min(all_values),
            "max": np.max(all_values),
            "median": np.median(all_values),
            "q25": np.percentile(all_values, 25),
            "q75": np.percentile(all_values, 75),
            "positive_pct": (all_values > 0).mean() * 100,
            "negative_pct": (all_values < 0).mean() * 100,
            "zero_pct": (all_values == 0).mean() * 100,
            "range": np.max(all_values) - np.min(all_values),
            "cv": (
                np.std(all_values) / np.mean(all_values)
                if np.mean(all_values) != 0
                else np.inf
            ),
        }

    return stats


def plot_value_distributions(
    train_datasets: dict[str, dict[str, Any]],
    eval_datasets: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Create distribution plots comparing datasets.

    Args:
        train_datasets: Training datasets.
        eval_datasets: Evaluation datasets.
        output_path: Directory to save plots.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    all_datasets = {**train_datasets, **eval_datasets}

    # Plot 1: Scale comparison (log scale)
    ax1 = fig.add_subplot(gs[0, :])
    dataset_names = []
    means = []
    stds = []

    for name, data in sorted(all_datasets.items()):
        all_vals = np.concatenate([ts[~np.isnan(ts)] for ts in data["series"]])
        if len(all_vals) > 0:
            dataset_names.append(name.split("/")[-1][:20])
            means.append(np.abs(np.mean(all_vals)) + 1e-10)
            stds.append(np.std(all_vals) + 1e-10)

    x_pos = np.arange(len(dataset_names))
    ax1.bar(x_pos, means, alpha=0.6, label="Mean (abs)", color="steelblue")
    ax1.bar(x_pos, stds, alpha=0.6, label="Std Dev", color="coral")
    ax1.set_yscale("log")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(dataset_names, rotation=45, ha="right")
    ax1.set_ylabel("Value (log scale)")
    ax1.set_title("Dataset Scale Comparison (Log Scale)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2 & 3: Distribution samples
    for idx, (split_name, datasets) in enumerate(
        [("Training", train_datasets), ("Eval", eval_datasets)]
    ):
        ax = fig.add_subplot(gs[1, idx])

        for i, (name, data) in enumerate(list(datasets.items())[:5]):
            all_vals = np.concatenate([ts[~np.isnan(ts)] for ts in data["series"]])
            if len(all_vals) > 1000:
                all_vals = np.random.choice(all_vals, 1000, replace=False)

            ax.hist(
                all_vals,
                bins=50,
                alpha=0.5,
                label=name.split("/")[-1][:15],
                density=True,
            )

        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.set_title(f"{split_name} Datasets - Value Distributions")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Plot 4: Sign analysis
    ax4 = fig.add_subplot(gs[2, 0])
    pos_pcts = []
    labels = []
    for name, data in list(all_datasets.items())[:8]:
        all_vals = np.concatenate([ts[~np.isnan(ts)] for ts in data["series"]])
        if len(all_vals) > 0:
            pos_pct = (all_vals > 0).mean() * 100
            neg_pct = (all_vals < 0).mean() * 100
            zero_pct = (all_vals == 0).mean() * 100
            pos_pcts.append([pos_pct, neg_pct, zero_pct])
            labels.append(name.split("/")[-1][:15])

    pos_pcts = np.array(pos_pcts)
    x_pos = np.arange(len(labels))
    ax4.barh(x_pos, pos_pcts[:, 0], label="Positive", color="green", alpha=0.7)
    ax4.barh(
        x_pos,
        pos_pcts[:, 1],
        left=pos_pcts[:, 0],
        label="Negative",
        color="red",
        alpha=0.7,
    )
    ax4.barh(
        x_pos,
        pos_pcts[:, 2],
        left=pos_pcts[:, 0] + pos_pcts[:, 1],
        label="Zero",
        color="gray",
        alpha=0.7,
    )
    ax4.set_yticks(x_pos)
    ax4.set_yticklabels(labels)
    ax4.set_xlabel("Percentage")
    ax4.set_title("Value Sign Distribution")
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="x")

    # Plot 5: Coefficient of Variation
    ax5 = fig.add_subplot(gs[2, 1])
    cvs = []
    labels = []
    for name, data in sorted(all_datasets.items()):
        all_vals = np.concatenate([ts[~np.isnan(ts)] for ts in data["series"]])
        if len(all_vals) > 0:
            mean_val = np.mean(all_vals)
            cv = np.std(all_vals) / mean_val if mean_val != 0 else 0
            if np.isfinite(cv):
                cvs.append(abs(cv))
                labels.append(name.split("/")[-1][:20])

    x_pos = np.arange(len(labels))
    ax5.barh(x_pos, cvs, color="purple", alpha=0.7)
    ax5.set_yticks(x_pos)
    ax5.set_yticklabels(labels, fontsize=8)
    ax5.set_xlabel("Coefficient of Variation (|std/mean|)")
    ax5.set_title("Relative Variability Across Datasets")
    ax5.grid(True, alpha=0.3, axis="x")

    plt.savefig(output_path / "value_statistics.png", dpi=150, bbox_inches="tight")
    print(f"   Saved: {output_path / 'value_statistics.png'}")
    plt.close()


def print_statistics_table(
    train_stats: dict[str, dict], eval_stats: dict[str, dict]
) -> None:
    """Print formatted statistics tables.

    Args:
        train_stats: Training dataset statistics.
        eval_stats: Evaluation dataset statistics.
    """
    print("Training Datasets:")
    print(f"{'Dataset':<40} {'Mean':<12} {'Std':<12} {'Range':<15} {'CV':<10}")
    print("-" * 90)
    for name in sorted(train_stats.keys()):
        s = train_stats[name]
        print(
            f"{name:<40} {s['mean']:>11.2f} {s['std']:>11.2f} "
            f"[{s['min']:.1f}, {s['max']:.1f}] {s['cv']:>9.2f}"
        )

    print("\n\nEvaluation Datasets:")
    print(f"{'Dataset':<40} {'Mean':<12} {'Std':<12} {'Range':<15} {'CV':<10}")
    print("-" * 90)
    for name in sorted(eval_stats.keys()):
        s = eval_stats[name]
        print(
            f"{name:<40} {s['mean']:>11.2f} {s['std']:>11.2f} "
            f"[{s['min']:.1f}, {s['max']:.1f}] {s['cv']:>9.2f}"
        )


def main() -> None:
    """Main analysis workflow."""
    data_dir = Path("data")
    output_dir = Path("data/analysis/analysis_output")
    output_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 80)
    print(" VALUE STATISTICS ANALYSIS ".center(80, "="))
    print("=" * 80 + "\n")

    print("📂 Loading training datasets...")
    train_datasets = scan_datasets(data_dir, "training")

    print("\n📂 Loading evaluation datasets...")
    eval_datasets = scan_datasets(data_dir, "test")

    print("\n📊 Computing statistics...")
    train_stats = analyze_value_statistics(train_datasets)
    eval_stats = analyze_value_statistics(eval_datasets)

    print("\n")
    print_statistics_table(train_stats, eval_stats)

    print("\n📊 Generating visualizations...")
    plot_value_distributions(train_datasets, eval_datasets, output_dir)

    print("\n" + "=" * 80)
    print(" ANALYSIS COMPLETE ".center(80, "="))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

