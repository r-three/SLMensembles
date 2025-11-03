"""
Simplified checkpointing for distillation training.
"""
import os
import torch
import torch.distributed as dist
import glob
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from simple_utils import main_print, is_main_process


# ==================================================
# Checkpointer Class
# ==================================================
class SimpleCheckpointer:
    """Simple checkpoint manager for saving and loading model states."""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    # ----------------------------------
    # Save Checkpoint
    # ----------------------------------
    def save(self, model, optimizer, lr_scheduler, epoch, global_step, loss):
        """Save a checkpoint."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"checkpoint_epoch{epoch}_step{global_step}.pt"
        )
        
        # Synchronize before checkpoint save
        if dist.is_initialized():
            dist.barrier()
        
        # Gather full state dicts from all ranks (collective operations)
        state_dict_opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        model_state_dict = get_model_state_dict(model=model, options=state_dict_opts)
        optimizer_state_dict = get_optimizer_state_dict(model=model, optimizers=optimizer, options=state_dict_opts)
        
        # Only the main process should save to avoid corruption
        if is_main_process():
            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'loss': loss,
            }
            
            torch.save(checkpoint, checkpoint_path)
            main_print(f"✓ Saved checkpoint to {checkpoint_path}")
                    
            # Keep only the last 3 checkpoints
            self._cleanup_old_checkpoints(keep_last=3)
        
        # Synchronize after checkpoint save to ensure all ranks wait
        if dist.is_initialized():
            dist.barrier()

    
    # ----------------------------------
    # Load Checkpoint
    # ----------------------------------
    def load(self, model, optimizer, lr_scheduler):
        """Load the latest checkpoint if it exists."""
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_*.pt"))
        
        if not checkpoints:
            main_print("No checkpoints found.")
            return None
        
        # Sort by modification time and get the latest
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        main_print(f"Loading checkpoint from {latest_checkpoint}")
        
        checkpoint = None
        if is_main_process():
            checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        
        # Broadcast checkpoint metadata to all ranks
        if dist.is_initialized():
            checkpoint = dist.broadcast_object_list([checkpoint], src=0)[0]
        
        # Load state dicts using FSDP-aware functions
        state_dict_opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        
        set_model_state_dict(
            model=model,
            model_state_dict=checkpoint['model_state_dict'],
            options=state_dict_opts
        )
        
        set_optimizer_state_dict(
            model=model,
            optimizers=optimizer,
            optim_state_dict=checkpoint['optimizer_state_dict'],
            options=state_dict_opts
        )
        
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        return {
            'epoch': checkpoint['epoch'],
            'global_step': checkpoint['global_step'],
            'loss': checkpoint['loss'],
        }
    
    # ----------------------------------
    # Cleanup Old Checkpoints
    # ----------------------------------
    def _cleanup_old_checkpoints(self, keep_last=3):
        """Remove old checkpoints, keeping only the most recent ones."""
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_*.pt"))
        
        if len(checkpoints) <= keep_last:
            return
        
        # Identify the oldest checkpoint by creation time (when it was saved)
        oldest_checkpoint = min(checkpoints, key=os.path.getctime)
        os.remove(oldest_checkpoint)
        main_print(f"Removed oldest checkpoint: {oldest_checkpoint}")
