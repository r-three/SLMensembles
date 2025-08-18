import os
import time
import re
import random

import torch
import numpy as np
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    _init_optim_state,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import distribute_tensor, DTensor
from typing import Dict, List, Tuple
from pathlib import Path
from dataclasses import dataclass

MODEL_CHECKPOINT = "model_state_dict.pt"
OPTIM_CHECKPOINT = "optim_state_dict.pt"
PARAMS = "params"

def _manifest_pointers(run_dir: str):
    """Return latest and per-round best checkpoint paths saved in the manifest file."""
    run_dir_path = Path(run_dir)
    latest = None
    best_per_round = {}

    mf = os.path.join(run_dir_path, "manifest.txt")
    if os.path.exists(mf):
        try:
            txt = open(mf, "r").read()
            m_latest = re.search(r"^latest:\s*(.+)$", txt, flags=re.MULTILINE)
            if m_latest:
                latest = Path(m_latest.group(1).strip())
            for m in re.finditer(r"^best\[(\d+)\]:\s*(.+)$", txt, flags=re.MULTILINE):
                best_per_round[int(m.group(1))] = Path(m.group(2).strip())
        except Exception:
            pass

    return latest, best_per_round

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


class Checkpointer:
    """Handles checkpoint saving and loading for distributed FSDP training."""

    def __init__(self, folder: str, dcp_api: bool = False, max_checkpoints_per_round: int = 3):
        """Initialize checkpointer with standardized round-based directory structure."""
        self.folder = folder  # Base checkpoint directory (outputdir/checkpoints)
        self.dcp_api = dcp_api
        self.max_checkpoints_per_round = max_checkpoints_per_round
        self.last_training_time = None
        self.last_checkpoint_path = None
        
        # Create base checkpoint directory
        os.makedirs(self.folder, exist_ok=True)
        
        # Find the latest checkpoint for resumption
        self._find_latest_checkpoint()

    def _find_latest_checkpoint(self):
        """Find the latest checkpoint in round-based directory structure."""
        if not os.path.exists(self.folder):
            self.last_checkpoint_path = None
            return
        
        try:
            # Look for round directories (0, 1, 2, ...)
            latest_round = -1
            latest_checkpoint = None
            latest_step = -1
            
            for item in os.listdir(self.folder):
                round_path = os.path.join(self.folder, item)
                if not os.path.isdir(round_path):
                    continue
                
                try:
                    round_num = int(item)
                    if round_num > latest_round:
                        # Look for checkpoint directories in this round
                        for ckpt_dir in os.listdir(round_path):
                            ckpt_path = os.path.join(round_path, ckpt_dir)
                            if os.path.isdir(ckpt_path) and 'step_' in ckpt_dir:
                                try:
                                    # Parse step number from directory name
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
    
    def is_empty(self):
        """Return True if no previous checkpoint exists in the target folder."""
        return self.last_checkpoint_path is None

    def load_model(self, model: FSDPModule):
        """Load the model weights from the latest checkpoint into the given FSDP model."""
        if self.use_flat_structure:
            if self.last_checkpoint_path is None:
                raise ValueError("No checkpoint found to load from")
            last_model_checkpoint = os.path.join(self.last_checkpoint_path, MODEL_CHECKPOINT)
        else:
            last_model_checkpoint = (
                f"{self.folder}/{'dcp_api' if self.dcp_api else 'dtensor_api'}"
                f"/{self.last_training_time}/{MODEL_CHECKPOINT}"
            )
        
        full_sd = torch.load(
            last_model_checkpoint, mmap=True, weights_only=True, map_location="cpu"
        )
        if self.dcp_api:
            set_model_state_dict(
                model=model,
                model_state_dict=full_sd,
                options=StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=True,
                ),
            )
            return
        meta_sharded_sd = model.state_dict()
        sharded_sd = {}
        for param_name, full_tensor in full_sd.items():
            sharded_meta_param = meta_sharded_sd.get(param_name)
            sharded_tensor = distribute_tensor(
                full_tensor,
                sharded_meta_param.device_mesh,
                sharded_meta_param.placements,
            )
            sharded_sd[param_name] = nn.Parameter(sharded_tensor)
        # choose `assign=True` since we cannot call `copy_` on meta tensor
        model.load_state_dict(sharded_sd, strict=False, assign=True)
    
    def load_org_model(self, model: FSDPModule, org_sd):
        """Load the provided full (CPU) state_dict into the sharded FSDP model."""
        # full_sd = torch.load(
        #     last_model_checkpoint, mmap=True, weights_only=True, map_location="cpu"
        # )
        full_sd = org_sd
        meta_sharded_sd = model.state_dict()
        sharded_sd = {}
        for param_name, full_tensor in full_sd.items():
            sharded_meta_param = meta_sharded_sd.get(param_name)
            sharded_tensor = distribute_tensor(
                full_tensor,
                sharded_meta_param.device_mesh,
                sharded_meta_param.placements,
            )
            sharded_sd[param_name] = nn.Parameter(sharded_tensor)
        # choose `assign=True` since we cannot call `copy_` on meta tensor
        model.load_state_dict(sharded_sd, strict=False, assign=True)

    def load_optim(self, model: FSDPModule, opt: torch.optim.Optimizer):
        """Load the optimizer state from the latest checkpoint into the given optimizer."""
        if self.use_flat_structure:
            if self.last_checkpoint_path is None:
                raise ValueError("No checkpoint found to load from")
            last_optim_checkpoint = os.path.join(self.last_checkpoint_path, OPTIM_CHECKPOINT)
        else:
            last_optim_checkpoint = (
                f"{self.folder}/{'dcp_api' if self.dcp_api else 'dtensor_api'}"
                f"/{self.last_training_time}/{OPTIM_CHECKPOINT}"
            )
        
        full_sd = torch.load(
            last_optim_checkpoint, mmap=True, weights_only=True, map_location="cpu"
        )
        if self.dcp_api:
            set_optimizer_state_dict(
                model=model,
                optimizers=opt,
                optim_state_dict=full_sd,
                options=StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=True,
                ),
            )
            return
        _init_optim_state(opt)
        param_groups = opt.state_dict()["param_groups"]
        state = opt.state_dict()["state"]

        full_param_groups = full_sd["param_groups"]
        full_state = full_sd["state"]

        for param_group, full_param_group in zip(param_groups, full_param_groups):
            for key, value in full_param_group.items():
                if key == PARAMS:
                    continue
                param_group[key] = value
            for pid, full_pid in zip(param_group[PARAMS], full_param_group[PARAMS]):
                if pid not in state:
                    continue
                param_state = state[pid]
                full_param_state = full_state[full_pid]
                for attr, full_tensor in full_param_state.items():
                    sharded_tensor = param_state[attr]
                    if isinstance(sharded_tensor, DTensor):
                        # exp_avg is DTensor
                        param_state[attr] = distribute_tensor(
                            full_tensor,
                            sharded_tensor.device_mesh,
                            sharded_tensor.placements,
                        )
                    else:
                        # step is plain tensor
                        param_state[attr] = full_tensor
        opt.load_state_dict(
            {
                "param_groups": param_groups,
                "state": state,
            }
        )

    def _get_full_model_state_dict(self, model: FSDPModule):
        """Assemble and return a full (CPU) state_dict for the given FSDP model."""
        if self.dcp_api:
            return get_model_state_dict(
                model=model,
                options=StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                ),
            )

        sharded_sd = model.state_dict()
        cpu_state_dict = {}
        for param_name, sharded_param in sharded_sd.items():
            full_param = sharded_param.full_tensor()
            if torch.distributed.get_rank() == 0:
                cpu_state_dict[param_name] = full_param.cpu()
            else:
                del full_param
        return cpu_state_dict

    def _get_full_optimizer_state_dict(
        self,
        model: FSDPModule,
        opt: torch.optim.Optimizer,
    ):
        """Assemble and return a full (CPU) optimizer state_dict for the given optimizer."""
        if self.dcp_api:
            return get_optimizer_state_dict(
                model=model,
                optimizers=opt,
                options=StateDictOptions(
                    full_state_dict=True,
                    cpu_offload=True,
                ),
            )
        is_rank_zero = torch.distributed.get_rank() == 0
        sharded_sd = opt.state_dict()
        sharded_state = sharded_sd["state"]
        full_state = {}
        for group_id, sharded_group in sharded_state.items():
            group_state = {}
            for attr, sharded_tensor in sharded_group.items():
                if isinstance(sharded_tensor, DTensor):
                    # "exp_avg" in AdamW is `DTensor`
                    full_tensor = sharded_tensor.full_tensor()
                else:
                    # "step" in AdamW is plain tensor
                    full_tensor = sharded_tensor
                if is_rank_zero:
                    group_state[attr] = full_tensor.cpu()
                else:
                    del full_tensor
            if is_rank_zero:
                full_state[group_id] = group_state
            else:
                del group_state
        if is_rank_zero:
            return {
                "param_groups": sharded_sd["param_groups"],
                "state": full_state,
            }
        else:
            return {}

    def save(self, model: FSDPModule, optim: torch.optim.Optimizer, round_num: int, step: int, current_loss: float, training_state=None):
        """Save checkpoint with standardized round-based directory structure and rotation."""
        model_state_dict = self._get_full_model_state_dict(model)
        optim_state_dict = self._get_full_optimizer_state_dict(model, optim)
        
        if torch.distributed.get_rank() == 0:
            # Create round directory
            round_dir = os.path.join(self.folder, str(round_num))
            os.makedirs(round_dir, exist_ok=True)
            
            # Create checkpoint directory with step and loss in name
            checkpoint_name = f"step_{step}_loss_{current_loss:.4f}"
            checkpoint_dir = os.path.join(round_dir, checkpoint_name)
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save model and optimizer state
            model_path = os.path.join(checkpoint_dir, MODEL_CHECKPOINT)
            optim_path = os.path.join(checkpoint_dir, OPTIM_CHECKPOINT)
            torch.save(model_state_dict, model_path)
            torch.save(optim_state_dict, optim_path)
            
            # Save training state if provided
            if training_state is not None:
                training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
                torch.save(training_state, training_state_path)
            
            # Implement checkpoint rotation (keep only max_checkpoints_per_round)
            self._rotate_checkpoints(round_dir)
            
            # Update latest checkpoint path
            self.last_checkpoint_path = checkpoint_dir
            
            print(f"Saved checkpoint: {checkpoint_dir}")
    
    def _rotate_checkpoints(self, round_dir: str):
        """Keep only the latest max_checkpoints_per_round checkpoints in the round directory."""
        if not os.path.exists(round_dir):
            return
        
        # Get all checkpoint directories in this round
        checkpoints = []
        for item in os.listdir(round_dir):
            item_path = os.path.join(round_dir, item)
            if os.path.isdir(item_path) and item.startswith('step_'):
                try:
                    # Extract step number for sorting
                    step_part = item.split('_')[1]
                    step = int(step_part)
                    checkpoints.append((step, item_path))
                except (ValueError, IndexError):
                    continue
        
        # Sort by step number (newest first)
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        
        # Remove oldest checkpoints if we exceed the limit
        if len(checkpoints) > self.max_checkpoints_per_round:
            for _, old_checkpoint_path in checkpoints[self.max_checkpoints_per_round:]:
                try:
                    import shutil
                    shutil.rmtree(old_checkpoint_path)
                    print(f"Removed old checkpoint: {old_checkpoint_path}")
                except Exception as e:
                    print(f"Warning: Failed to remove old checkpoint {old_checkpoint_path}: {e}")
    
    def load_training_state(self):
        """Load training state (step counters, epoch, etc.) from the latest checkpoint."""
        if self.use_flat_structure:
            if self.last_checkpoint_path is None:
                return None
            training_state_path = os.path.join(self.last_checkpoint_path, "training_state.pt")
        else:
            if self.last_training_time is None:
                return None
            training_state_path = (
                f"{self.folder}/{'dcp_api' if self.dcp_api else 'dtensor_api'}"
                f"/{self.last_training_time}/training_state.pt"
            )
        
        if not os.path.exists(training_state_path):
            return None
        
        try:
            return torch.load(training_state_path, map_location="cpu", weights_only=True)
        except Exception:
            return None
    
    def get_checkpoint_info(self):
        """Get information about the latest checkpoint for resumption."""
        if self.use_flat_structure and self.last_checkpoint_path:
            # Parse checkpoint name: round_epoch_step_current_loss_min_loss
            try:
                name = os.path.basename(self.last_checkpoint_path)
                parts = name.split('_')
                if len(parts) >= 3:
                    return {
                        'round': int(parts[0]),
                        'epoch': int(parts[1]), 
                        'step': int(parts[2]),
                        'current_loss': float(parts[3]) if len(parts) > 3 else None,
                        'min_loss': float(parts[4]) if len(parts) > 4 else None,
                        'path': self.last_checkpoint_path
                    }
            except (ValueError, IndexError):
                pass
        
        return None


@dataclass(frozen=True)
class Checkpoint:
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

def _parse_dirname(name: str) -> Tuple[int, int, float, float]:
    """Parse a checkpoint folder name into (round, epoch, step, current_loss, min_loss)."""
    r, e, s, cur, minl = name.split("_", 4)
    return int(r), int(e), int(s), float(cur), float(minl)

def index_checkpoints(parent: str) -> Dict[int, List[Checkpoint]]:
    """Build an index {round -> [Checkpoint, …]} under parent, skipping non-matching folders."""
    parent = Path(parent)
    index: Dict[int, List[Checkpoint]] = {}

    for entry in parent.iterdir():
        if not entry.is_dir():
            continue
        try:
            r, e, s, cur, minl = _parse_dirname(entry.name)
        except (ValueError, IndexError):
            # skip folders that don’t match the expected pattern
            continue

        ckpt = Checkpoint(entry, r, e, s, cur, minl)
        index.setdefault(r, []).append(ckpt)

    return index

def best_checkpoint(index: dict, round_num: int) -> Path:
    """Return path of lowest-current_eval_loss checkpoint for round_num, else raise KeyError."""
    if round_num not in index:
        raise KeyError(f"No checkpoints found for round {round_num}")

    best = min(index[round_num], key=lambda c: c.current_loss)
    return best.path


def create_training_state(round_num: int, epoch_num: int, step: int, 
                         current_loss: float, min_loss: float,
                         lr_scheduler_state=None, rng_states=None):
    """Create a training state dictionary for checkpointing."""
    state = {
        'round': round_num,
        'epoch': epoch_num,
        'step': step,
        'current_loss': current_loss,
        'min_loss': min_loss,
        'timestamp': time.time()
    }
    
    if lr_scheduler_state is not None:
        state['lr_scheduler_state'] = lr_scheduler_state
    
    if rng_states is not None:
        state['rng_states'] = rng_states
    
    return state


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