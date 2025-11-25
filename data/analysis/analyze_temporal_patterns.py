"""Temporal patterns and seasonality analysis.

This script analyzes autocorrelation, trends, and variance stability
to understand temporal dependencies and seasonality.
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


def compute_autocorrelation(ts: np.ndarray, max_lag: int = 100) -> np.ndarray:
    """Compute autocorrelation function.

    Args:
        ts: Time series array.
        max_lag: Maximum lag to compute.

    Returns:
        Array of autocorrelation values.
    """
    ts_clean = ts[~np.isnan(ts)]
    if len(ts_clean) < max_lag:
        return np.array([])

    ts_centered = ts_clean - np.mean(ts_clean)
    c0 = np.dot(ts_centered, ts_centered) / len(ts_centered)

    acf = np.zeros(max_lag)
    for lag in range(max_lag):
        if lag < len(ts_centered):
            c_lag = (
                np.dot(ts_centered[: -lag or None], ts_centered[lag:]) / len(ts_centered)
            )
            acf[lag] = c_lag / c0 if c0 != 0 else 0

    return acf


def plot_raw_series(
    datasets: dict[str, dict[str, Any]], output_path: Path, title: str
) -> None:
    """Plot raw time series samples with trend lines.

    Args:
        datasets: Dictionary of datasets.
        output_path: Directory to save plots.
        title: Plot title.
    """
    n_datasets = len(datasets)
    ncols = 3
    nrows = (n_datasets + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.5 * nrows))
    axes = axes.flatten() if n_datasets > 1 else [axes]
    
    for idx, (name, data) in enumerate(sorted(datasets.items())):
        ax = axes[idx]
        if len(data["series"]) > 0:
            # Plot first series with trend line
            ts = data["series"][0]
            time = np.arange(len(ts))
            ts_clean = ts[~np.isnan(ts)]
            time_clean = time[~np.isnan(ts)]
            
            ax.plot(time_clean, ts_clean, linewidth=0.7, alpha=0.7, label="Series")
            
            # Add trend line
            if len(time_clean) > 10:
                z = np.polyfit(time_clean, ts_clean, 1)
                p = np.poly1d(z)
                ax.plot(time_clean, p(time_clean), "r--", linewidth=1.5, 
                       alpha=0.8, label=f"Trend (slope={z[0]:.3f})")
            
            ax.set_title(f"{name.split('/')[-1]}\nFreq: {data['freq']}", fontsize=9)
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_datasets, len(axes)):
        axes[idx].axis("off")
    
    plt.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path / "temporal_raw_series.png", dpi=150, bbox_inches="tight")
    print(f"   Saved: {output_path / 'temporal_raw_series.png'}")
    plt.close()


def plot_normalized_series(datasets: dict[str, dict[str, Any]], output_path: Path) -> None:
    """Plot normalized series grouped by frequency.

    Args:
        datasets: Dictionary of datasets.
        output_path: Directory to save plots.
    """
    # Group by frequency
    freq_groups = {}
    for name, data in datasets.items():
        freq = data["freq"]
        if freq not in freq_groups:
            freq_groups[freq] = []
        freq_groups[freq].append((name, data))
    
    n_freqs = len(freq_groups)
    fig, axes = plt.subplots(n_freqs, 1, figsize=(16, 4 * n_freqs))
    axes = [axes] if n_freqs == 1 else axes
    
    for idx, (freq, datasets_list) in enumerate(sorted(freq_groups.items())):
        ax = axes[idx]
        
        for name, data in datasets_list:
            if len(data["series"]) > 0:
                ts = data["series"][0]
                ts_clean = ts[~np.isnan(ts)]
                
                # Normalize: z-score
                ts_norm = (ts_clean - np.mean(ts_clean)) / (np.std(ts_clean) + 1e-8)
                time = np.arange(len(ts_norm))
                
                ax.plot(time, ts_norm, linewidth=0.8, alpha=0.6, 
                       label=name.split("/")[-1][:20])
        
        # Add average trend
        ax.axhline(y=0, color="k", linestyle="-", linewidth=1.5, alpha=0.5, label="Zero line")
        
        ax.set_title(f"Normalized Series - Frequency: {freq}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time")
        ax.set_ylabel("Z-score")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Normalized Time Series by Frequency", fontsize=14, y=0.998)
    plt.tight_layout()
    plt.savefig(output_path / "temporal_normalized_series.png", dpi=150, bbox_inches="tight")
    print(f"   Saved: {output_path / 'temporal_normalized_series.png'}")
    plt.close()


def plot_cross_dataset_analysis(
    train_datasets: dict[str, dict[str, Any]],
    eval_datasets: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot cross-dataset temporal characteristics.

    Args:
        train_datasets: Training datasets.
        eval_datasets: Evaluation datasets.
        output_path: Directory to save plots.
    """
    all_datasets = {**train_datasets, **eval_datasets}
    
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # ACF: High-frequency
    ax1 = fig.add_subplot(gs[0, 0])
    for name, data in all_datasets.items():
        if data["freq"] in ["5T", "15T", "T"]:
            acfs = [compute_autocorrelation(ts, max_lag=200) 
                   for ts in data["series"][:5]]
            acfs = [acf for acf in acfs if len(acf) > 0]
            if acfs:
                mean_acf = np.mean(acfs, axis=0)
                ax1.plot(mean_acf, alpha=0.7, label=f"{name.split('/')[-1][:15]}", linewidth=2)
    
    ax1.axhline(y=0, color="k", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("Lag")
    ax1.set_ylabel("Autocorrelation")
    ax1.set_title("ACF: High-Freq (5T, 15T, T)")
    ax1.legend(fontsize=6)
    ax1.grid(True, alpha=0.3)
    
    # ACF: Low-frequency
    ax2 = fig.add_subplot(gs[0, 1])
    for name, data in all_datasets.items():
        if data["freq"] in ["H", "D"]:
            acfs = [compute_autocorrelation(ts, max_lag=100) 
                   for ts in data["series"][:5]]
            acfs = [acf for acf in acfs if len(acf) > 0]
            if acfs:
                mean_acf = np.mean(acfs, axis=0)
                ax2.plot(mean_acf, alpha=0.7, label=f"{name.split('/')[-1][:15]}", linewidth=2)
    
    ax2.axhline(y=0, color="k", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Lag")
    ax2.set_ylabel("Autocorrelation")
    ax2.set_title("ACF: Low-Freq (H, D)")
    ax2.legend(fontsize=6)
    ax2.grid(True, alpha=0.3)
    
    # Variance stability
    ax3 = fig.add_subplot(gs[0, 2])
    for name, data in list(all_datasets.items())[:8]:
        if len(data["series"]) > 0:
            ts = data["series"][0]
            ts_clean = ts[~np.isnan(ts)]
            if len(ts_clean) > 100:
                window = min(len(ts_clean) // 10, 100)
                rolling_std = [np.std(ts_clean[i:i + window]) 
                              for i in range(0, len(ts_clean) - window, window)]
                ax3.plot(rolling_std, alpha=0.6, label=name.split("/")[-1][:12], linewidth=1.5)
    
    ax3.set_xlabel("Window")
    ax3.set_ylabel("Rolling Std Dev")
    ax3.set_title("Variance Stability")
    ax3.legend(fontsize=6)
    ax3.grid(True, alpha=0.3)
    
    # Average ACF decay rate
    ax4 = fig.add_subplot(gs[1, 0])
    decay_rates = []
    labels = []
    for name, data in all_datasets.items():
        acfs = [compute_autocorrelation(ts, max_lag=50) 
               for ts in data["series"][:5]]
        acfs = [acf for acf in acfs if len(acf) > 10]
        if acfs:
            mean_acf = np.mean(acfs, axis=0)
            # Find where ACF drops below 0.5
            decay_idx = np.where(mean_acf < 0.5)[0]
            decay_rate = decay_idx[0] if len(decay_idx) > 0 else len(mean_acf)
            decay_rates.append(decay_rate)
            labels.append(name.split("/")[-1][:15])
    
    colors = ["blue"] * len(train_datasets) + ["red"] * len(eval_datasets)
    ax4.barh(range(len(labels)), decay_rates, color=colors, alpha=0.7)
    ax4.set_yticks(range(len(labels)))
    ax4.set_yticklabels(labels, fontsize=7)
    ax4.set_xlabel("Lags until ACF < 0.5")
    ax4.set_title("Memory Length (ACF Decay)")
    ax4.grid(True, alpha=0.3, axis="x")
    
    # Trend strength comparison
    ax5 = fig.add_subplot(gs[1, 1])
    trend_strengths = []
    labels = []
    for name, data in all_datasets.items():
        if len(data["series"]) > 0:
            ts = data["series"][0]
            ts_clean = ts[~np.isnan(ts)]
            if len(ts_clean) > 10:
                x = np.arange(len(ts_clean))
                z = np.polyfit(x, ts_clean, 1)
                # Normalize by std
                trend_strength = abs(z[0]) * len(ts_clean) / (np.std(ts_clean) + 1e-8)
                trend_strengths.append(trend_strength)
                labels.append(name.split("/")[-1][:15])
    
    colors = ["blue"] * len(train_datasets) + ["red"] * len(eval_datasets)
    ax5.barh(range(len(labels)), trend_strengths, color=colors, alpha=0.7)
    ax5.set_yticks(range(len(labels)))
    ax5.set_yticklabels(labels, fontsize=7)
    ax5.set_xlabel("Trend Strength (normalized)")
    ax5.set_title("Linear Trend Magnitude")
    ax5.grid(True, alpha=0.3, axis="x")
    
    # Seasonality strength (spectral analysis simple)
    ax6 = fig.add_subplot(gs[1, 2])
    seasonality_scores = []
    labels = []
    for name, data in all_datasets.items():
        if len(data["series"]) > 0:
            acfs = [compute_autocorrelation(ts, max_lag=100) 
                   for ts in data["series"][:3]]
            acfs = [acf for acf in acfs if len(acf) > 10]
            if acfs:
                mean_acf = np.mean(acfs, axis=0)
                # Seasonality: sum of periodic peaks in ACF
                seasonality = np.sum(np.abs(mean_acf[10:50]))
                seasonality_scores.append(seasonality)
                labels.append(name.split("/")[-1][:15])
    
    colors = ["blue"] * len(train_datasets) + ["red"] * len(eval_datasets)
    ax6.barh(range(len(labels)), seasonality_scores, color=colors, alpha=0.7)
    ax6.set_yticks(range(len(labels)))
    ax6.set_yticklabels(labels, fontsize=7)
    ax6.set_xlabel("Seasonality Score")
    ax6.set_title("Periodicity Strength")
    ax6.grid(True, alpha=0.3, axis="x")
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="blue", alpha=0.7, label="Train"),
                      Patch(facecolor="red", alpha=0.7, label="Test")]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=9)
    
    plt.suptitle("Cross-Dataset Temporal Characteristics", fontsize=14, y=0.998)
    plt.savefig(output_path / "temporal_cross_analysis.png", dpi=150, bbox_inches="tight")
    print(f"   Saved: {output_path / 'temporal_cross_analysis.png'}")
    plt.close()


def main() -> None:
    """Main analysis workflow."""
    data_dir = Path("data")
    output_dir = Path("data/analysis/analysis_output")
    output_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 80)
    print(" TEMPORAL & SEASONALITY ANALYSIS ".center(80, "="))
    print("=" * 80 + "\n")

    print("📂 Loading training datasets...")
    train_datasets = scan_datasets(data_dir, "training")

    print("\n📂 Loading evaluation datasets...")
    eval_datasets = scan_datasets(data_dir, "test")

    all_datasets = {**train_datasets, **eval_datasets}

    print("\n📊 Generating visualizations...")
    print("   1/3: Raw time series with trends...")
    plot_raw_series(all_datasets, output_dir, "Time Series Samples Across Datasets")
    
    print("   2/3: Normalized series by frequency...")
    plot_normalized_series(all_datasets, output_dir)
    
    print("   3/3: Cross-dataset temporal analysis...")
    plot_cross_dataset_analysis(train_datasets, eval_datasets, output_dir)

    print("\n" + "=" * 80)
    print(" ANALYSIS COMPLETE ".center(80, "="))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

