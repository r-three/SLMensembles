"""
Simplified utilities for distillation training.
"""
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
from transformers import AutoTokenizer
import random
import numpy as np

from simple_config import config


def fix_seed(seed: int):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def main_print(*args, **kwargs):
    """Print only on main process."""
    if is_main_process():
        print(*args, **kwargs)


def get_dataset():
    """Load and tokenize the dataset."""
    # Load dataset
    dataset = load_dataset(config.dataset_name, split="train")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.student_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        # Tokenize the text
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=config.max_seq_length,
            return_tensors=None,
        )
        
        # Create labels (same as input_ids but with -100 for padding)
        labels = tokenized["input_ids"].copy()
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                if tokenized["attention_mask"][i][j] == 0:
                    labels[i][j] = -100
        
        tokenized["labels"] = labels
        return tokenized
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=8,
        remove_columns=dataset.column_names,
    )
    
    # Split into train/test
    split_dataset = tokenized_dataset.train_test_split(test_size=0.05, seed=config.seed)
    
    return split_dataset


def prepare_dataset(train_dataset, eval_dataset):
    """Prepare DataLoaders with DistributedSampler."""
    # Create samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size() if dist.is_initialized() else 1,
        rank=dist.get_rank() if dist.is_initialized() else 0,
        shuffle=True,
        seed=config.seed,
    )
    
    eval_sampler = DistributedSampler(
        eval_dataset,
        num_replicas=dist.get_world_size() if dist.is_initialized() else 1,
        rank=dist.get_rank() if dist.is_initialized() else 0,
        shuffle=False,
    )
    
    # Custom collate function to handle tensor conversion
    def collate_fn(batch):
        # Convert lists to tensors
        input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
        attention_mask = torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long)
        labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=config.eval_batch_size,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    
    return train_dataloader, eval_dataloader
