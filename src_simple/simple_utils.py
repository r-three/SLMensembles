"""
Simplified utilities for distillation training.
"""
import math
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import datasets
from datasets import load_from_disk
import random
import numpy as np
import os, csv, time, glob, sys, atexit, threading
import torch
from tqdm import tqdm
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple
import queue
import pandas as pd
import numpy as np

from simple_config import config


# ==================================================
# Random Seed Utilities
# ==================================================
def fix_seed(seed: int):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==================================================
# Distributed Training Utilities
# ==================================================
def is_main_process():
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def main_print(*args, **kwargs):
    """Print only on main process."""
    if is_main_process():
        print(*args, **kwargs)

# ==================================================
# Custom Collator for Padding
# ==================================================
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

            # Convert to tensor if needed, otherwise clone
            input_ids = item["input_ids"] if isinstance(item["input_ids"], torch.Tensor) else torch.tensor(item["input_ids"])
            attention_mask = item["attention_mask"] if isinstance(item["attention_mask"], torch.Tensor) else torch.tensor(item["attention_mask"])
            labels = item["labels"] if isinstance(item["labels"], torch.Tensor) else torch.tensor(item["labels"])

            batch_padded["input_ids"].append(torch.cat([
                input_ids,
                torch.full((pad_len,), self.pad_token_id, dtype=input_ids.dtype)
            ]))

            batch_padded["attention_mask"].append(torch.cat([
                attention_mask,
                torch.full((pad_len,), self.pad_attention_id, dtype=attention_mask.dtype)
            ]))

            batch_padded["labels"].append(torch.cat([
                labels,
                torch.full((pad_len,), self.pad_label_id, dtype=labels.dtype)
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


# ==================================================
# Dataset Loading
# ==================================================
def get_dataset():
    """Load dataset."""
    return datasets.load_from_disk(config.dataset_path)


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

    train_dataloader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
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


# ==================================================
# Top k maximum loss
# ==================================================
# ---------------------- Helper functions ----------------------

def _to_scalar(x: Any):
    """Convert various types to Python scalar."""
    # Works for Python numbers, PyTorch tensors, numpy scalars
    try:
        import torch
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return x.detach().float().item()
            raise ValueError("Tensor must be scalar")
    except Exception:
        pass
    try:
        import numpy as np
        if isinstance(x, np.number):
            return float(x)
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        pass
    # Last resort
    return float(x)

class AsyncLossLogger:
    """
    Append-only JSONL writer with batching and periodic atomic snapshots.
    Designed to keep training non-blocking.
    """
    def __init__(self, log_path: str,
                 flush_interval_s: float = 30.0,
                 max_queue: int = 100_000):
        self.log_path = log_path
        self.flush_interval_s = flush_interval_s

        # Only main process creates the directory to avoid race conditions
        if is_main_process():
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        dist.barrier()
            
        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self._writer = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer.start()

    def write(self, record: Dict[str, Any]):
        """Non-blocking enqueue; drops if queue is full to avoid stalling."""
        try:
            self._q.put_nowait(record)
        except queue.Full:
            # If you want to block instead, use self._q.put(record)
            pass

    def update_and_write_many(self, ids, tr_step_loss, next_token_loss, kl_loss, valid_count):
        ids_list = list(ids)

        # Convert elementwise (each item can be a Python number, 0-d torch tensor, numpy scalar, etc.)
        tr_list = [_to_scalar(x) for x in tr_step_loss]
        nt_list = [_to_scalar(x) for x in next_token_loss]
        kl_list = [_to_scalar(x) for x in kl_loss]
        vc_list = [int(_to_scalar(x)) for x in valid_count]

        now = time.time()
        with self._lock:
            for id_, tr, nt, kl, vc in zip(ids_list, tr_list, nt_list, kl_list, vc_list):
                self.write({"id": id_, "tr": tr, "nt": nt, "kl": kl, "vc": vc, "t": now})


    def close(self, timeout: float = 10.0):
        self._stop.set()
        self._writer.join(timeout=timeout)

    def _writer_loop(self):
        buf = []
        last_flush = time.time()

        # Check if file exists to know if we need to write header
        file_exists = os.path.exists(self.log_path)

        # Open CSV file in append mode, line-buffered
        f = open(self.log_path, "a", newline="", buffering=1)
        writer = csv.DictWriter(f, fieldnames=["id", "tr", "nt", "kl", "vc", "t"])
        if not file_exists:
            writer.writeheader()

        try:
            while not self._stop.is_set() or not self._q.empty():
                try:
                    item = self._q.get(timeout=self.flush_interval_s)
                    buf.append(item)
                except queue.Empty:
                    pass

                now = time.time()

                # Flush buffer if enough time passed or buffer is large
                if buf and (now - last_flush >= self.flush_interval_s or len(buf) >= 2048):
                    writer.writerows(buf)
                    f.flush()
                    buf.clear()
                    last_flush = now

        finally:
            # Flush any remaining records
            if buf:
                writer.writerows(buf)
                f.flush()
            f.close()

    def get_top_n_ids(self, heading, k_percent: int):
        df = pd.read_csv(self.log_path)
        n = len(df)
        top_n = max(1, math.ceil(n * k_percent / 100.0))
        top_n_df = df.nlargest(top_n, heading)

        return top_n_df['id'].tolist()
