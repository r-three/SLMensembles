import os
import re
import json
import math
import time
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint import save as dcp_save, load as dcp_load
from torch.distributed.checkpoint import FileSystemWriter, FileSystemReader

# --------------- Helper Functions ---------------

MODEL_CHECKPOINT = "model_state_dict.pt"
OPTIM_CHECKPOINT = "optim_state_dict.pt"
TRAIN_STATE = "training_state.pt"  
STEP_DIR_RE = re.compile(r"^step_(\d+)_loss_([0-9.]+)$")

DCP_SD_OPTS = StateDictOptions(full_state_dict=False, cpu_offload=False)
FULL_SD_OPTS = StateDictOptions(full_state_dict=True, cpu_offload=True)





def _fmt_step_dir(step: int, loss: float) -> str:
    """Format a step directory name."""
    return f"step_{step:08d}_loss_{loss:.4f}"


def _rng_capture() -> Dict[str, Any]:
    """Capture Random Number Generator states."""
    out = {"time": time.time()}
    try:
        out["python"] = None
        out["numpy"] = None
        out["torch"] = torch.get_rng_state()
        out["torch_cuda"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    except Exception:
        pass
    return out


def _rng_restore(rng: Dict[str, Any]) -> None:
    """Restore Random Number Generator states."""
    if not rng:
        return
    if "torch" in rng and rng["torch"] is not None:
        torch.set_rng_state(rng["torch"])
    if "torch_cuda" in rng and rng["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["torch_cuda"])



class Checkpointer:
    """
    Handles checkpoint saving and loading for distributed FSDP training.

    DCP-based training checkpoints in:
        <output_path>/checkpoints/<round_num>/<step_dir>/
    where <step_dir> == 'step_{step:08d}_loss_{loss:.4f}'.
    """

    def __init__(self, checkpoint_dir: str, max_checkpoints_per_round: int = 3):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints_per_round = max_checkpoints_per_round
        self.last_training_time = None
        self.last_checkpoint_path = None
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _find_latest_checkpoint(self):
        """Find the latest checkpoint in round-based directory structure."""
        try:
            latest_round = -1
            latest_checkpoint = None
            latest_step = -1
            
            for item in os.listdir(self.checkpoint_dir):
                round_path = os.path.join(self.checkpoint_dir, item)
                if not os.path.isdir(round_path):
                    continue
                try:
                    round_num = int(item)
                    if round_num > latest_round:
                        for ckpt_dir in os.listdir(round_path):
                            ckpt_path = os.path.join(round_path, ckpt_dir)
                            if os.path.isdir(ckpt_path) and 'step_' in ckpt_dir:
                                try:
                                    step_part = ckpt_dir.split('_')[1]
                                    step = int(step_part)
                                    if round_num > latest_round or (round_num == latest_round and step > latest_step):
                                        latest_round = round_num
                                        latest_step = step
                                        latest_checkpoint = ckpt_path
                                except (ValueError, IndexError):
                                    continue
                except ValueError:
                    continue
            self.last_checkpoint_path = latest_checkpoint
        except Exception:
            self.last_checkpoint_path = None

    def load_model(self, model: FSDPModule):
        """Load (DCP) model weights from the latest checkpoint into the given FSDP model."""
        if self.last_checkpoint_path is None:
            raise ValueError("No checkpoint found to load from")

        reader = FileSystemReader(self.last_checkpoint_path)

        model_sd = get_model_state_dict(model=model, options=DCP_SD_OPTS)
        dcp_load({"model": model_sd}, storage_reader=reader)
        set_model_state_dict(model=model, model_state_dict=model_sd, options=DCP_SD_OPTS)

    def load_optim(self, model: FSDPModule, optim: torch.optim.Optimizer):
        """Load (DCP) optimizer state from the latest checkpoint."""
        if self.last_checkpoint_path is None:
            raise ValueError("No checkpoint found to load from")

        reader = FileSystemReader(self.last_checkpoint_path)
        optim_sd = get_optimizer_state_dict(model=model, optimizer=optim, options=DCP_SD_OPTS)
        dcp_load({"optim": optim_sd}, storage_reader=reader)

        set_optimizer_state_dict(
            model=model, optimizer=optim, optimizer_state_dict=optim_sd, options=DCP_SD_OPTS
        )

    def save(self, model: FSDPModule, optim: torch.optim.Optimizer, round_num: int, step: int, current_loss: float, training_state=None):
        """Save checkpoint with standardized round-based directory structure and rotation."""
        ckpt_dir = os.path.join(self.checkpoint_dir, str(round_num), f"step_{step}_loss_{current_loss:.4f}")
        os.makedirs(ckpt_dir, exist_ok=True)

        writer = FileSystemWriter(ckpt_dir)
        state = {
            "model": get_model_state_dict(model=model, options=DCP_SD_OPTS),
            "optim": get_optimizer_state_dict(model=model, optimizer=optim, options=DCP_SD_OPTS),
        }
        dcp_save(state, storage_writer=writer)

        if training_state and dist.get_rank() == 0:
            torch.save(training_state, os.path.join(ckpt_dir, "training_state.pt"))
        self.last_checkpoint_path = ckpt_dir
        print(f"Saved DCP checkpoint: {ckpt_dir}")

        dist.barrier()
        self.rotate_checkpoints(round_num)

    
    def rotate_checkpoints(self, round_dir: str):
        """Keep only the latest max_checkpoints_per_round checkpoints in the round directory."""
        if not os.path.exists(round_dir):
            return
        
        checkpoints = []
        for item in os.listdir(round_dir):
            item_path = os.path.join(round_dir, item)
            if os.path.isdir(item_path) and item.startswith('step_'):
                try:
                    step_part = item.split('_')[1]
                    step = int(step_part)
                    checkpoints.append((step, item_path))
                except (ValueError, IndexError):
                    continue

        checkpoints.sort(key=lambda x: x[0], reverse=True)
        
        if len(checkpoints) > self.max_checkpoints_per_round:
            for _, old_checkpoint_path in checkpoints[self.max_checkpoints_per_round:]:
                try:
                    import shutil
                    shutil.rmtree(old_checkpoint_path)
                    print(f"Removed old checkpoint: {old_checkpoint_path}")
                except Exception as e:
                    print(f"Warning: Failed to remove old checkpoint {old_checkpoint_path}: {e}")

    @staticmethod
    def export_ensemble_member(model: FSDPModule, round: int, output_dir: str):
        """Gather a FULL state dict (CPU offload) and write a single .pt file for inference/ensemble."""
        os.makedirs(output_dir, exist_ok=True)
        # Gather full parameters on CPU (rank0 saves file; others just participate)
        save_dir = os.path.join(output_dir, f"round_{round}")
        os.makedirs(save_dir, exist_ok=True)
        full_sd = get_model_state_dict(model=model, options=FULL_SD_OPTS)
        if dist.get_rank() == 0:
            torch.save(full_sd, os.path.join(save_dir, MODEL_CHECKPOINT))
        dist.barrier()
        return save_dir

    def index_checkpoints(parent_dir: str) -> dict[int, list[tuple[str,int,float]]]:
        """
        Robustly index checkpoints for each round, supporting the nested layout:
            <parent>/<round>/<step_{N}_loss_{X}>
        Returns: { round: [(path, step, loss), ... sorted by step] }
        """
        index: dict[int, list[tuple[str,int,float]]] = {}
        if not os.path.isdir(parent_dir):
            return index
        for item in os.listdir(parent_dir):
            round_path = os.path.join(parent_dir, item)
            try:
                r = int(item)
            except ValueError:
                continue
            if not os.path.isdir(round_path):
                continue
            rows = []
            for sub in os.listdir(round_path):
                if not sub.startswith("step_"):
                    continue
                try:
                    # step_12345_loss_0.9876
                    parts = sub.split("_")
                    step = int(parts[1])
                    loss = float(parts[3])
                    rows.append((os.path.join(round_path, sub), step, loss))
                except Exception:
                    pass
            rows.sort(key=lambda t: t[1])
            if rows:
                index[r] = rows
        return index





































def get_latest_checkpoint_folder(path):
    """Return the largest numbered folder in the given path."""
    max_num = None
    if not os.path.exists(path):
        return max_num
    for name in os.listdir(path):
        folder_path = os.path.join(path, name)
        if os.path.isdir(folder_path):
            try:
                num = int(name)
                if max_num is None or num > max_num:
                    max_num = num
            except ValueError:
                pass  # Skip non-numeric folder names
    return max_num


# --------------- Checkpointer ---------------
class Checkpointer:
    def __init__(self, checkpoint_dir: str, max_checkpoints_per_round: int = 3):
        """Initialize checkpointer with standardized round-based directory structure."""
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints_per_round = max_checkpoints_per_round
        self.last_training_time = None
        self.last_checkpoint_path = None
        
        # Create base checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Find the latest checkpoint for resumption
        self._find_latest_checkpoint()

    def load_training_state(self):
        """Load training state (step counters, epoch, etc.) from the latest checkpoint."""
        if self.last_checkpoint_path is None:
            return None
        training_state_path = os.path.join(self.last_checkpoint_path, "training_state.pt")

        if not os.path.exists(training_state_path):
            return None
        
        try:
            return torch.load(training_state_path, map_location="cpu", weights_only=True)
        except Exception:
            return None

    def load_optim(self, model, opt):
        """Load the optimizer state from the latest checkpoint into the given optimizer."""
        if self.last_checkpoint_path is None:
            raise ValueError("No checkpoint found to load from")

        reader = FileSystemReader(self.last_checkpoint_path)
        optim_sd = get_optimizer_state_dict(model=model, optimizers=opt, options=DCP_SD_OPTS)
        dcp_load(state_dict={"optim": optim_sd}, storage_reader=reader)

        # NOTE: must call this before .backward() or after .step()
        set_optimizer_state_dict(model=model, optimizer=opt, optim_state_dict=optim_sd, options=DCP_SD_OPTS)


    


@dataclass(frozen=True)
class Checkpoint:
    """Record of a checkpoint's path, round/epoch/step, and loss metrics."""
    path: Path
    round: int
    epoch: int
    step: int
    current_loss: float
    min_loss: float

    def resolve_resume_checkpoint(run_dir: str) -> Path | None:
        """Resume from checkpoint"""
        latest, best = _manifest_pointers(run_dir)

        # Fallback to directory scan (you already have name parsing/indexing)
        try:
            index = index_checkpoints(run_dir)
            if not index:
                return None
            newest = None
            for r, items in index.items():
                for ck in items:
                    key = (ck.round, ck.epoch, ck.step)
                    if newest is None or key > newest[0]:
                        newest = (key, ck.path)
            return newest[1] if newest else None
        except Exception:
            return None

def save_rng_states():
    """Save random number generator states for reproducible resumption."""
    return {
        'python': random.getstate() if 'random' in globals() else None,
        'numpy': np.random.get_state() if 'np' in globals() else None,
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }


def restore_rng_states(rng_states):
    """Restore random number generator states from checkpoint."""
    if rng_states is None:
        return
    
    if 'python' in rng_states and rng_states['python'] is not None:
        import random
        random.setstate(rng_states['python'])
    
    if 'numpy' in rng_states and rng_states['numpy'] is not None:
        import numpy as np
        np.random.set_state(rng_states['numpy'])
    
    if 'torch' in rng_states:
        torch.set_rng_state(rng_states['torch'])
    
    if 'torch_cuda' in rng_states and rng_states['torch_cuda'] is not None:
        torch.cuda.set_rng_state_all(rng_states['torch_cuda'])