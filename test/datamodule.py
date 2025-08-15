import os
import sys
from tqdm import tqdm
import psutil
import torch
import pytest
import numpy as np
from pathlib import Path
from memory_profiler import profile
from emotionlinmult.train.datamodule import MultiDatasetModule

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Test configuration matching the one in the datamodule
TEST_CONFIG = {
    'batch_size': 16,  # Smaller batch size for testing
    'num_workers': 4,  # Fewer workers for testing
    'feature_list': ['wavlm_baseplus', 'clip', 'xml_roberta'],
    'target_list': ['tmm_wavlm_baseplus', 'tmm_clip', 'tmm_xml_roberta'],
    'datasets': {
        'train': ['mosei'],  # Using a single dataset for testing
        'valid': ['mosei'],
        'test': ['mosei']
    },
    'block_dropout': 1.0,
    'block_length': 15,
    'num_block_drops': 2,
    'min_gap_between_blocks': 5,
}

def get_memory_usage():
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB

def test_datamodule_memory_leak():
    """Test for memory leaks in the datamodule."""
    # Initialize datamodule
    datamodule = MultiDatasetModule(TEST_CONFIG)
    datamodule.setup(stage='fit')
    
    # Get dataloader
    train_dataloader = datamodule.train_dataloader()
    
    # Track memory usage
    initial_memory = get_memory_usage()
    memory_readings = [initial_memory]
    
    print("\n=== Starting memory leak test ===")
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Run multiple epochs to detect memory growth
    num_epochs = 3
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        
        # Track memory at the start of each batch
        epoch_memory = []
        
        for batch_idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            
            if batch_idx > 200: break

            # Move batch to CPU and force garbage collection
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.cpu()
            
            # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Get memory after processing batch
            current_memory = get_memory_usage()
            epoch_memory.append(current_memory)
            
            print(f"Batch {batch_idx + 1} - Memory: {current_memory:.2f} MB")
            
            # Explicitly delete batch and force garbage collection
            del batch

        # Calculate memory growth during this epoch
        if epoch_memory:
            memory_growth = max(epoch_memory) - min(epoch_memory)
            print(f"Memory growth this epoch: {memory_growth:.2f} MB")

        # Store memory at the end of epoch
        memory_readings.append(get_memory_usage())

    # Calculate total memory growth
    final_memory = get_memory_usage()
    total_growth = final_memory - initial_memory

    print("\n=== Memory Test Results ===")
    print(f"Initial memory: {initial_memory:.2f} MB")
    print(f"Final memory: {final_memory:.2f} MB")
    print(f"Total memory growth: {total_growth:.2f} MB")
    print(f"Memory readings: {memory_readings}")
    
    # Check for significant memory growth (more than 100MB)
    assert total_growth < 100, f"Significant memory growth detected: {total_growth:.2f} MB"
    print("Test passed: No significant memory leak detected")

if __name__ == "__main__":
    test_datamodule_memory_leak()

    #=== Memory Test Results ===
    #Initial memory: 591.42 MB
    #Final memory: 607.34 MB
    #Total memory growth: 15.93 MB
    #Memory readings: [591.41796875, 607.19140625, 607.28515625, 607.34375]
    #Test passed: No significant memory leak detected