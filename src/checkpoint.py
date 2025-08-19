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
    """Handles checkpoint saving and loading for distributed FSDP training.

    The checkpoint directory structure is as follows:
    output_path/checkpoints/
    ├── 0/
    │   ├── step_5000_loss_1.2345/
    │   │   ├── model_state_dict.pt
    │   │   ├── optim_state_dict.pt
    │   │   └── training_state.pt
    │   └── step_3000_loss_1.5678/
    │       ├── model_state_dict.pt
    │       ├── optim_state_dict.pt
    │       └── training_state.pt
    └── 1/
        └── step_2000_loss_0.9876/
            ├── model_state_dict.pt
            ├── optim_state_dict.pt
            └── training_state.pt
    """

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

    def _manifest_pointers(self, run_dir: str):
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

    def _find_latest_checkpoint(self):
        """Find the latest checkpoint in round-based directory structure."""
        if not os.path.exists(self.checkpoint_dir):
            self.last_checkpoint_path = None
            return
        
        try:
            # Look for round directories (0, 1, 2, ...)
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


    def load_model(self, model):
        """Load the model weights from the latest checkpoint into the given FSDP model."""
        if self.last_checkpoint_path is None:
            raise ValueError("No checkpoint found to load from")

        reader = FileSystemReader(self.last_checkpoint_path)

        model_sd = get_model_state_dict(model=model, options=DCP_SD_OPTS)
        dcp_load(state_dict={"model": model_sd}, storage_reader=reader)
        set_model_state_dict(model=model, model_state_dict=model_sd, options=DCP_SD_OPTS)
        

    def save(self, model: FSDPModule, optim: torch.optim.Optimizer, round_num: int, step: int, current_loss: float, training_state=None):
        """Save checkpoint with standardized round-based directory structure and rotation."""
        # ------------------------
        # build target folder name 
        # ------------------------
        round_dir = os.path.join(self.checkpoint_dir, str(round_num))
        if dist.get_rank() == 0:
            os.makedirs(round_dir, exist_ok=True)
        dist.barrier()

        checkpoint_name = f"step_{step}_loss_{current_loss:.4f}"
        checkpoint_dir = os.path.join(round_dir, checkpoint_name)

        # ------------------------------------------
        # construct DCP-aware state dicts (sharded) 
        # ------------------------------------------
        model_sd = get_model_state_dict(model=model, options=DCP_SD_OPTS)
        optim_sd = get_optimizer_state_dict(model=model, optimizers=optim, options=DCP_SD_OPTS)
        state = {"model": model_sd, "optim": optim_sd}

        # ------------------------------------------
        # collective save across all ranks 
        # ------------------------------------------
        writer = FileSystemWriter(checkpoint_dir)
        dcp_save(state_dict=state, storage_writer=writer)

        dist.barrier() 

        # ------------------------------------------
        # store non-tensor training metadata 
        # ------------------------------------------
        if dist.get_rank() == 0:
            # Optionally store non-tensor training metadata alongside the DCP folder
            if training_state is not None:
                torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"))
            
            # ------------------------
            # rotation-by-step logic
            # ------------------------
            self._rotate_checkpoints(round_dir)
            self.last_checkpoint_path = checkpoint_dir
            print(f"Saved DCP checkpoint: {checkpoint_dir}")

    def is_empty(self):
        return self.last_checkpoint_path is None




    
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
    
    
    def get_checkpoint_info(self):
        """Get information about the latest checkpoint for resumption."""
        if self.last_checkpoint_path:
            # Parse checkpoint name: step_{step}_loss_{loss:.4f}
            try:
                name = os.path.basename(self.last_checkpoint_path)
                parts = name.split('_')
                if len(parts) >= 4:
                    return {
                        'step': int(parts[1]),
                        'loss': float(parts[3]),
                        'path': self.last_checkpoint_path
                    }
            except (ValueError, IndexError):
                pass
        
        return None


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