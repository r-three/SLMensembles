import argparse
import os
import time
import torch
import torch.distributed as dist
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from tqdm.auto import tqdm
import sys

from simple_config import config
from simple_trainer import Trainer
from simple_utils import prepare_dataset, get_dataset, is_main_process, main_print, fix_seed, AsyncLossLogger
from simple_checkpoint import SimpleCheckpointer


def main(args):
    for epoch in range(start_epoch, config.num_epochs):
        trainer.epoch = epoch
        main_print(f"\nEpoch {epoch}/{config.num_epochs-1}")
        
        # ----------------------------------
        # Epoch Setup
        # ----------------------------------
        if hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)
        
        # Initialize tracking variables
        epoch_train_loss = 0.0
        num_train_steps = 0
        eval_count = 0
        
        # ----------------------------------
        # Training Iteration
        # ----------------------------------
        progress_bar = tqdm(train_dataloader, disable=rank != 0, file=sys.stdout, desc=f"Training Epoch {epoch}")
        for batch_idx, batch in enumerate(progress_bar):
            # Debug mode: stop after max_steps
            if config.debug_mode and trainer.global_step >= config.debug_max_steps:
                main_print(f"[DEBUG MODE] Reached max steps ({config.debug_max_steps}), stopping training")
                break
            
            # Train step (handles gradient accumulation internally)
            loss = trainer.train_step(batch)
            epoch_train_loss += loss
            num_train_steps += 1
            
            # Update progress bar
            if rank == 0:
                progress_bar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'step': trainer.global_step
                })
            
            # ------ Periodic Evaluation ------
            if trainer.global_step > 0 and trainer.global_step % config.eval_steps == 0:
                dist.barrier()  # Sync before eval
                eval_loss = trainer.eval_step(eval_dataloader)
                main_print(f"Step {trainer.global_step}: eval_loss = {eval_loss:.4f}")
                eval_count += 1
                dist.barrier()  # Sync after eval
            
            # ------ Periodic Checkpointing ------
            # Skip in debug mode to avoid NCCL timeout
            if not config.debug_mode and trainer.global_step > 0 and trainer.global_step % config.save_steps == 0:
                dist.barrier()
                trainer.save_checkpoint(loss=None)
                dist.barrier()