import os
import time
import torch
from datetime import datetime
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from trl import DataCollatorForCompletionOnlyLM
from utils import (CSVLogger, prepare_dataset, format_time_elapsed, 
                  is_main_process, main_print, check_batch_shape, fix_seed,
                  inspect_mixed_precision, inspect_model,
                  set_modules_to_forward_prefetch, set_modules_to_backward_prefetch)
from ensemble import ModelEnsemble
from checkpoint import Checkpointer, index_checkpoints, best_checkpoint
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from tqdm.auto import tqdm
from shard_weight import *
from utils import fix_seed
import atexit
from pathlib import Path
from datasets import Dataset, DatasetDict
from utils import DistillDataset, get_round_path
from checkpoint import Checkpoint
import wandb

def main():
    dataset = datasets.load_from_disk(config.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    teacher_model = AutoModelForCausalLM.from_pretrained(
        config.teacher_model_name,
        torch_dtype=torch.bfloat16,
    ).to('cuda')

    
