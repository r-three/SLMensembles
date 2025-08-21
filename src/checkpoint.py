import os
import re
import json
import math
import time
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from torch.distributed.fsdp import FSDPModule

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

    def __init__(self, output_base_dir: str, max_checkpoints_per_round: int = 3):
        self.output_dir = output_base_dir
        self.checkpoint_dir = os.path.join(output_base_dir, "checkpoints")
        self.max_checkpoints_per_round = max_checkpoints_per_round
        self.last_training_time = None
        self.last_checkpoint_path = None
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    @staticmethod
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

    def get_resumable_checkpoint(self):
        """Get the latest checkpoint to resume training from."""
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

    def load(self, model: FSDPModule, optim: torch.optim.Optimizer):
        """Load the latest checkpoint model, training args, and optim state"""
        self.get_resumable_checkpoint()

        if self.last_checkpoint_path is None:
            raise ValueError("No checkpoint found to load from")
        
        self.load_model(model)
        self.load_optim(model, optim)
        state = checkpointer.load_training_state()

        return state

    def load_model(self, model: FSDPModule):
        """Load (DCP) model weights from the latest checkpoint into the given FSDP model."""
        reader = FileSystemReader(self.last_checkpoint_path)

        model_sd = get_model_state_dict(model=model, options=DCP_SD_OPTS)
        dcp_load({"model": model_sd}, storage_reader=reader)
        set_model_state_dict(model=model, model_state_dict=model_sd, options=DCP_SD_OPTS)\

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

    def load_training_state(self):
        """Load training state (step counters, epoch, etc.) from the latest checkpoint."""
        if self.last_checkpoint_path is None:
            return None
        training_state_path = os.path.join(self.last_checkpoint_path, TRAIN_STATE)

        if not os.path.exists(training_state_path):
            return None
        state = torch.load(training_state_path, map_location="cpu", weights_only=True)
        if state and "rng" in state and state["rng"] is not None:
            _rng_restore(state["rng"])
        return state

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

        meta = {
            "round_num": int(round_num),
            "step": int(step),
            "loss": float(current_loss),
            "rng": _rng_capture(),
        }

        if training_state:
            meta.update(training_state)

        if dist.get_rank() == 0:
            torch.save(meta, os.path.join(ckpt_dir, TRAIN_STATE))

        self.last_checkpoint_path = ckpt_dir
        print(f"Saved DCP checkpoint: {ckpt_dir}")

        dist.barrier()
        self.rotate_checkpoints(os.path.join(self.checkpoint_dir, str(round_num)))

    
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


    def save_rng_states(self):
        """Save random number generator states for reproducible resumption."""
        return {
            'python': None,  # We'll keep this simple for now
            'numpy': None,   # Add if needed
            'torch': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }


    def restore_rng_states(self, rng_states):
        """Restore random number generator states from checkpoint."""
        if rng_states is None:
            return
        
        if 'torch' in rng_states and rng_states['torch'] is not None:
            torch.set_rng_state(rng_states['torch'])
        
        if 'torch_cuda' in rng_states and rng_states['torch_cuda'] is not None:
            torch.cuda.set_rng_state_all(rng_states['torch_cuda'])