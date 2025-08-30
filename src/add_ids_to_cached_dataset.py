#!/usr/bin/env python3
"""
Script to add IDs to existing cached logprob dataset by matching order.
Since both datasets were created from the same shuffled source, they should have identical ordering.
"""

import datasets
import os
from tqdm import tqdm
import config

def add_ids_to_cached_dataset():
    print("Loading datasets...")
    
    # Load your new dataset with IDs (same shuffle order as original)
    source_dataset = datasets.load_from_disk("/scratch/klambert/dataset/tulu-3-sft-mixture-pretokenized_clean")
    
    # Load existing cached logprob dataset
    cached_dataset_path = os.path.join(config.logprob_cache_path, "teacher_logprobs")
    if not os.path.exists(cached_dataset_path):
        print(f"ERROR: Cached dataset not found at {cached_dataset_path}")
        return
    
    cached_dataset = datasets.load_from_disk(cached_dataset_path)
    
    print(f"Source dataset sizes: Train={len(source_dataset['train'])}, Test={len(source_dataset['test'])}")
    print(f"Cached dataset sizes: Train={len(cached_dataset['train'])}, Test={len(cached_dataset['test'])}")
    
    breakpoint()

    # Verify sizes match
    for split in ['train', 'test']:
        if len(source_dataset[split]) != len(cached_dataset[split]):
            print(f"ERROR: Size mismatch for {split} split!")
            print(f"Source: {len(source_dataset[split])}, Cached: {len(cached_dataset[split])}")
            return
    
    print("Sizes match! Adding IDs...")
    
    # Add IDs to cached dataset
    updated_splits = {}
    for split in ['train', 'test']:
        print(f"Processing {split} split...")
        
        # Extract IDs from source dataset in order
        source_ids = [source_dataset[split][i]['id'] for i in tqdm(range(len(source_dataset[split])))]
        
        # Add IDs to cached dataset
        cached_data = cached_dataset[split].to_dict()
        cached_data['id'] = source_ids
        
        updated_splits[split] = datasets.Dataset.from_dict(cached_data)
    
    # Create updated dataset
    updated_dataset = datasets.DatasetDict(updated_splits)
    
    # Save with backup
    backup_path = cached_dataset_path + "_backup"
    if os.path.exists(backup_path):
        import shutil
        shutil.rmtree(backup_path)
    
    print(f"Creating backup at {backup_path}...")
    os.rename(cached_dataset_path, backup_path)
    
    print(f"Saving updated dataset with IDs...")
    updated_dataset.save_to_disk(cached_dataset_path)
    
    print("✅ Successfully added IDs to cached dataset!")
    print(f"Backup saved at: {backup_path}")
    
    # Show final dataset structure
    print(f"\nFinal dataset structure:")
    print(f"Train features: {list(updated_dataset['train'].features.keys())}")
    print(f"Test features: {list(updated_dataset['test'].features.keys())}")
    
    # Verify a few samples
    print("\nVerification - First 3 samples:")
    for i in range(3):
        src_id = source_dataset['train'][i]['id'] 
        cached_id = updated_dataset['train'][i]['id']
        print(f"Sample {i}: Source ID = {src_id}, Cached ID = {cached_id}, Match = {src_id == cached_id}")
    
    print(f"\n✅ Dataset now includes 'id' column and is saved at: {cached_dataset_path}")

if __name__ == "__main__":
    add_ids_to_cached_dataset()
