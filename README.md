# Time Series Forecasting Prototype

A prototype forecasting model trained on the GiftEvalPretrain dataset, implementing a simplified Moirai-inspired architecture with a training pipeline for multivariate time series.

> **DISCLAIMER**: This prototype is designed for illustration purposes only. Many components may be incomplete or require improvement. See `todo.md` for a list of planned enhancements.

## Project Structure

- **`config/`** - YAML configuration files for data, model, and training parameters
- **`data_processing/`** - Dataset classes, dataloaders, preprocessing utilities, and sampling strategies
- **`model/`** - Moirai-inspired transformer architecture for time series forecasting
- **`train/`** - Training pipeline, training scripts, and optimization logic
- **`monitor_and_setup/`** - Checkpointing, reproducibility utilities, and training monitoring
- **`tests/`** - Unit and integration tests for data processing and model components
- **`checkpoints/`** - Saved model checkpoints and training state

## Quick Start

Configure training parameters in `config/`, then run:

```bash
python -m train.run_train
```

## Data Processing

**Dataset Preparation:**
- Data downloaded from Hugging Face using Git Large File Storage (LFS)
- Selected datasets manually curated to ensure variety in frequencies, sizes, and temporal profiles
- Training sets merged into 10 GB chunks with dataset identification columns (see `data/merge_parquet_files.py` and `data/create_mixed_chunks_streaming.py`)
- Data split into train, validation, and test sets (`data/create_train_val_test_splits.py`)
- Additional test directory contains both in-distribution and out-of-distribution datasets

A data analysis is available in `data/analysis/`.

**Preprocessing Pipeline:**

Following Moirai's approach, data is patchified using patch lengths that depend on time series frequency (lower frequency → smaller patch length). During training, windows of varying lengths are randomly sampled. Minimum and maximum window lengths are enforced, filtering out series that are too short (see `data_processing/preprocessing.py`). Series containing NaN values are also filtered out.

Dataset and dataloader objects are constructed using standard PyTorch patterns (`data_processing/dataloader.py`, `data_processing/dataset.py`).

## Model

This implements a minimal version of the Moirai model (`model/moirai.py`). The architecture is a transformer with a specialized attention mechanism that handles multivariate time series by flattening them into a single sequence. Key simplifications include:
- Removed output distribution sampling in favor of single-value predictions per patch
- Implemented RevIN normalization for improved stability

The architecture made me think of a 2023 NeurIPS [paper](https://arxiv.org/abs/2306.06156) applied to DNA sequence generation from "multivariate" (groups of) sequences. Both use similar flattening strategies with attention and positional encoding schemes designed to preserve causal relationships.

## Training

The training pipeline is implemented in `train/train_minimal.py`. It follows a standard training loop that:
- Iterates over the dataloader
- Performs forward and backward passes for each batch
- Trains on the training set and evaluates on the validation set

**Note**: Current evaluation is not properly implemented in terms of both conception and implementation (see `todo.md` for details).

## Monitoring & Testing

- **Experiment Tracking**: Basic visualization and tracking implemented using Weights & Biases for checkpoints and metrics (`monitor_and_setup/checkpoint.py`, `monitor_and_setup/reproducibility.py`)
- **Unit Tests**: Test suite included in the `tests/` directory covering data processing and model components
