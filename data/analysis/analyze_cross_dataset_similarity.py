"""Cross-dataset similarity analysis.

This script computes feature-based similarity metrics between datasets
to validate OOD categories and understand dataset relationships.
"""

import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

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


def compute_dataset_features(data: dict[str, Any]) -> np.ndarray:
    """Compute feature vector for a dataset.

    Args:
        data: Dataset with series and metadata.

    Returns:
        Feature vector: [mean, std, skew, kurtosis, acf_mean, trend, ...].
    """
    all_vals = []
    for ts in data["series"]:
        ts_clean = ts[~np.isnan(ts)]
        if len(ts_clean) > 0:
            all_vals.append(ts_clean)

    if not all_vals:
        return np.zeros(10)

    # Concatenate all values
    concat_vals = np.concatenate(all_vals)

    # Statistical features
    mean = np.mean(concat_vals)
    std = np.std(concat_vals)
    median = np.median(concat_vals)
    q25 = np.percentile(concat_vals, 25)
    q75 = np.percentile(concat_vals, 75)

    # Scale-invariant features
    cv = std / mean if mean != 0 else 0

    # ACF features (averaged over series)
    acf_feats = []
    for ts in all_vals[:10]:  # Sample 10 series
        acf = compute_autocorrelation(ts, max_lag=20)
        if len(acf) > 0:
            acf_feats.append(acf[:10])

    acf_mean = np.mean(acf_feats, axis=0) if acf_feats else np.zeros(10)

    # Trend feature (simple linear fit on first series)
    if len(all_vals[0]) > 10:
        x = np.arange(len(all_vals[0]))
        trend = np.polyfit(x, all_vals[0], 1)[0]
    else:
        trend = 0

    # Combine features
    features = np.array(
        [
            np.log(abs(mean) + 1),
            np.log(std + 1),
            cv,
            (q75 - q25) / (median + 1),
            trend,
            *acf_mean[:5],
        ]
    )

    return features


def plot_cross_dataset_similarity(
    train_datasets: dict[str, dict[str, Any]],
    eval_datasets: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Analyze and visualize cross-dataset similarity.

    Args:
        train_datasets: Training datasets.
        eval_datasets: Evaluation datasets.
        output_path: Directory to save plots.
    """
    all_datasets = {**train_datasets, **eval_datasets}

    # Compute feature vectors
    dataset_names = []
    features = []
    is_train = []

    for name, data in all_datasets.items():
        feat = compute_dataset_features(data)
        if np.all(np.isfinite(feat)):
            dataset_names.append(name.split("/")[-1])
            features.append(feat)
            is_train.append(name in train_datasets)

    features = np.array(features)

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Compute distance matrix
    distances = squareform(pdist(features_scaled, metric="euclidean"))

    # Create plots
    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(1, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Plot 1: Distance matrix heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(distances, cmap="YlOrRd", aspect="auto")
    ax1.set_xticks(range(len(dataset_names)))
    ax1.set_yticks(range(len(dataset_names)))
    ax1.set_xticklabels(
        [n[:15] for n in dataset_names], rotation=45, ha="right", fontsize=8
    )
    ax1.set_yticklabels([n[:15] for n in dataset_names], fontsize=8)
    ax1.set_title("Cross-Dataset Distance Matrix\n(Lower = More Similar)")
    plt.colorbar(im, ax=ax1)

    # Plot 2: 2D projection using MDS
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    coords = mds.fit_transform(distances)

    ax2 = fig.add_subplot(gs[0, 1])
    for i, (name, is_tr) in enumerate(zip(dataset_names, is_train)):
        color = "blue" if is_tr else "red"
        marker = "o" if is_tr else "^"
        label = "Train" if is_tr and i == 0 else ("Eval" if not is_tr and i == 1 else "")
        ax2.scatter(
            coords[i, 0],
            coords[i, 1],
            c=color,
            marker=marker,
            s=100,
            alpha=0.6,
            label=label,
        )
        ax2.annotate(
            name[:12],
            (coords[i, 0], coords[i, 1]),
            fontsize=7,
            alpha=0.8,
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax2.set_xlabel("MDS Dimension 1")
    ax2.set_ylabel("MDS Dimension 2")
    ax2.set_title("Dataset Similarity Map (2D Projection)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.savefig(
        output_path / "cross_dataset_similarity.png", dpi=150, bbox_inches="tight"
    )
    print(f"   Saved: {output_path / 'cross_dataset_similarity.png'}")
    plt.close()

    # Print similarity insights
    print("\n🔍 Similarity Insights:")

    # Find most similar train-eval pairs
    print("\n   Most similar Train-Eval pairs:")
    for i, name_i in enumerate(dataset_names):
        if is_train[i]:
            for j, name_j in enumerate(dataset_names):
                if not is_train[j]:
                    dist = distances[i, j]
                    print(f"      {name_i[:20]} <-> {name_j[:20]}: {dist:.2f}")

    # Find most/least similar overall
    upper_tri_indices = np.triu_indices_from(distances, k=1)
    upper_tri_dists = distances[upper_tri_indices]
    min_idx = np.argmin(upper_tri_dists)
    max_idx = np.argmax(upper_tri_dists)

    min_i, min_j = upper_tri_indices[0][min_idx], upper_tri_indices[1][min_idx]
    max_i, max_j = upper_tri_indices[0][max_idx], upper_tri_indices[1][max_idx]

    print(f"\n   Most similar overall:")
    print(
        f"      {dataset_names[min_i]} <-> {dataset_names[min_j]}: "
        f"{distances[min_i, min_j]:.2f}"
    )

    print(f"\n   Most dissimilar overall:")
    print(
        f"      {dataset_names[max_i]} <-> {dataset_names[max_j]}: "
        f"{distances[max_i, max_j]:.2f}"
    )


def main() -> None:
    """Main analysis workflow."""
    data_dir = Path("data")
    output_dir = Path("data/analysis/analysis_output")
    output_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 80)
    print(" CROSS-DATASET SIMILARITY ANALYSIS ".center(80, "="))
    print("=" * 80 + "\n")

    print("📂 Loading training datasets...")
    train_datasets = scan_datasets(data_dir, "training")

    print("\n📂 Loading evaluation datasets...")
    eval_datasets = scan_datasets(data_dir, "test")

    print("\n📊 Computing similarity metrics and generating plots...")
    plot_cross_dataset_similarity(train_datasets, eval_datasets, output_dir)

    print("\n" + "=" * 80)
    print(" ANALYSIS COMPLETE ".center(80, "="))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

