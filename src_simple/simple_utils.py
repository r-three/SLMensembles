"""
Simplified utilities for distillation training.
"""
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import datasets
from datasets import load_from_disk
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
    """Load dataset."""
    return datasets.load_from_disk(config.dataset_path)
    
class CustomPadCollator:
    def __init__(self, max_length, pad_token_id: int = -100, pad_label_id: int = -100, pad_attention_id: int = 0):
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.pad_label_id = pad_label_id
        self.pad_attention_id = pad_attention_id

    def __call__(self, batch):
        batch_padded = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }

        # Track other keys
        other_keys = [k for k in batch[0].keys() if k not in batch_padded]

        for item in batch:
            length = len(item["input_ids"])
            pad_len = self.max_length - length

            batch_padded["input_ids"].append(torch.cat([
                torch.tensor(item["input_ids"]),
                torch.full((pad_len,), self.pad_token_id, dtype=torch.tensor(item["input_ids"]).dtype)
            ]))

            batch_padded["attention_mask"].append(torch.cat([
                torch.tensor(item["attention_mask"]),
                torch.full((pad_len,), self.pad_attention_id, dtype=torch.tensor(item["attention_mask"]).dtype)
            ]))

            batch_padded["labels"].append(torch.cat([
                torch.tensor(item["labels"]),
                torch.full((pad_len,), self.pad_label_id, dtype=torch.tensor(item["labels"]).dtype)
            ]))

        # Stack padded fields
        for k in ["input_ids", "attention_mask", "labels"]:
            batch_padded[k] = torch.stack(batch_padded[k])

        # Add other keys without padding (just stack as-is)
        for k in other_keys:
            values = [item[k] for item in batch]
            try:
                batch_padded[k] = torch.stack(values)
            except:
                batch_padded[k] = values  # Leave as list if not stackable

        return batch_padded

def prepare_dataset(train_ds, eval_ds):
    """Prepare DataLoaders with DistributedSampler."""
    custom_collator = CustomPadCollator(1024, pad_token_id=100277) # pad_token_id for OLmo2

    train_sampler = DistributedSampler(
        train_ds,
        dist.get_world_size(),
        dist.get_rank(),
        shuffle=True,
        seed=config.seed,
        drop_last=True, 
    )
    test_sampler = DistributedSampler(
        eval_ds,
        dist.get_world_size(),
        dist.get_rank(),
        shuffle=False,
        drop_last=True, 
    )

    tokenizer = AutoTokenizer.from_pretrained(config.student_model_name)
    train_dataloader = DataLoader(
        train_ds,
        batch_size=config.per_device_train_batch_size,
        sampler=train_sampler,
        shuffle=False,
        collate_fn=custom_collator,
        num_workers=0,
        persistent_workers=False,
        drop_last=True
    )   
    eval_dataloader = DataLoader(
        eval_ds,
        batch_size=config.eval_batch_size,
        sampler=test_sampler,
        shuffle=False,
        collate_fn=custom_collator,
        num_workers=0,
        persistent_workers=False,
        drop_last=True  
    )

    dist.barrier()
    return train_dataloader, eval_dataloader
