"""
Simplified checkpointing for distillation training.
"""
import os
import torch
import glob
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
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

        state_dict_opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        model_state_dict = get_model_state_dict(model=model, options=state_dict_opts)
        
        # Only rank 0 saves to disk
        if is_main_process():
            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'loss': loss,
            }
            
            torch.save(checkpoint, checkpoint_path)
            main_print(f"✓ Saved checkpoint to {checkpoint_path}")
            
            # Keep only the last 3 checkpoints
            self._cleanup_old_checkpoints(keep_last=3)
    
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
        
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        
        # Load state dicts
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
        
        # Sort by modification time
        checkpoints.sort(key=os.path.getmtime)
        
        # Remove older checkpoints
        for checkpoint in checkpoints[:-keep_last]:
            os.remove(checkpoint)
            main_print(f"Removed old checkpoint: {checkpoint}")
