"""Test dataset-aware sampling with the new dataloader."""

import logging
from collections import Counter

from data_processing.dataloader import create_train_dataloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_dataset_sampling():
    """Test that dataset-aware sampling creates pure batches."""
    
    logger.info("="*80)
    logger.info("Testing Dataset-Aware Sampling")
    logger.info("="*80)
    
    # Create dataloader with dataset-aware sampling
    # Use more lenient parameters for series with fewer patches
    train_loader = create_train_dataloader(
        data_dir="data/chunks/train_sample",
        batch_size=32,
        num_workers=4,
        use_dataset_sampling=True,
        temperature=1.0,  # Proportional to size
        max_window_length=64,  # Reduced from 128
        min_context_length=4,   # Reduced from 16
        max_context_length=48,  # Reduced from 96
        min_prediction_length=2, # Reduced from 4
        max_prediction_length=16, # Reduced from 32
        min_length=256,
    )
    
    logger.info(f"\nTotal batches: {len(train_loader)}")
    
    # Test a few batches
    logger.info("\n" + "="*80)
    logger.info("Sampling 10 batches to verify pure batch property")
    logger.info("="*80)
    
    dataset_counter = Counter()
    pure_batch_count = 0
    mixed_batch_count = 0
    
    for i, batch in enumerate(train_loader):
        if i >= 10:
            break
        
        # Get unique datasets in this batch
        datasets_in_batch = set(batch['dataset_name'])
        frequencies_in_batch = set(batch['freq'])
        
        # Count datasets seen
        for ds in datasets_in_batch:
            dataset_counter[ds] += 1
        
        # Check if pure batch
        is_pure = len(datasets_in_batch) == 1
        if is_pure:
            pure_batch_count += 1
        else:
            mixed_batch_count += 1
        
        logger.info(f"\nBatch {i}:")
        logger.info(f"  Datasets: {datasets_in_batch}")
        logger.info(f"  Frequencies: {frequencies_in_batch}")
        logger.info(f"  Pure batch: {'✓ Yes' if is_pure else '✗ No (UNEXPECTED!)'}")
        logger.info(f"  Context shape: {batch['context_patches'].shape}")
        logger.info(f"  Target shape: {batch['target_patches'].shape}")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("Summary")
    logger.info("="*80)
    logger.info(f"Pure batches: {pure_batch_count}/10")
    logger.info(f"Mixed batches: {mixed_batch_count}/10")
    
    if pure_batch_count == 10:
        logger.info("✓ SUCCESS: All batches are pure!")
    else:
        logger.warning(f"⚠ WARNING: Found {mixed_batch_count} mixed batches")
    
    logger.info("\nDataset distribution in sampled batches:")
    for dataset, count in dataset_counter.most_common():
        logger.info(f"  {dataset}: {count} batches")


if __name__ == "__main__":
    test_dataset_sampling()

