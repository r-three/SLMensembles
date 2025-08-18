import os, csv, time, glob, sys, atexit
from datetime import datetime
import pdb
from tqdm import tqdm
import shutil
import torch
import datasets
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import config
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk, DatasetDict, concatenate_datasets, Dataset
from trl import DataCollatorForCompletionOnlyLM
import random
import numpy as np
import subprocess, json, re
from hashlib import blake2s

# ---------------------- Training and FSDP2-specific functions ----------------------
try:
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.tensor import Shard, DTensor
    FSDP2_AVAILABLE = True
except ImportError:
    FSDP2_AVAILABLE = False
    FSDPModule = None
    DTensor = None

def inspect_model(model):
    """Inspect FSDP2 model structure and parameters."""
    if not FSDP2_AVAILABLE or not isinstance(model, FSDPModule):
        return
        
    if torch.distributed.get_rank() == 0:
        print(model)

    for param in model.parameters():
        assert param.placements == (Shard(0),)
        assert isinstance(param, DTensor) 


def inspect_mixed_precision(model):
    """Inspect mixed precision settings for FSDP2 model."""
    if not FSDP2_AVAILABLE or not isinstance(model, FSDPModule):
        return
        
    model.unshard()
    for param in model.parameters(recurse=False):
        assert param.dtype == torch.bfloat16
    model.reshard()

def check_batch_shape(train_dataloader):
    """Debug utility to check batch shapes."""
    batch = next(iter(train_dataloader))
    print("Batch shapes:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
            print(f"  {key}: {[v.shape for v in value[:3]]}...")  # Show first 3 shapes
        else:
            print(f"  {key}: {type(value)}")


def set_modules_to_forward_prefetch(model, num_to_forward_prefetch):
    """Set forward prefetching for model layers."""
    for i, layer in enumerate(model.layers):
        if i >= len(model.layers) - num_to_forward_prefetch:
            break
        layers_to_prefetch = [
            model.layers[i + j] for j in range(1, num_to_forward_prefetch + 1)
        ]
        layer.set_modules_to_forward_prefetch(layers_to_prefetch)


def set_modules_to_backward_prefetch(model, num_to_backward_prefetch):
    """Set backward prefetching for model layers."""
    for i, layer in enumerate(model.layers):
        if i < num_to_backward_prefetch:
            continue
        layers_to_prefetch = [
            model.layers[i - j] for j in range(1, num_to_backward_prefetch + 1)
        ]
        layer.set_modules_to_backward_prefetch(layers_to_prefetch)


# ---------------------- Utility functions ----------------------

def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main_print(*args, **kwargs):
    """Print only from main process in distributed training."""
    if is_main_process():
        print(*args, **kwargs)


def _get_rank():
    """Get current process rank in distributed training."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def is_main_process() -> bool:
    """Check if current process is the main process."""
    return _get_rank() == 0 if config.ddp else True


def format_time_elapsed(seconds):
    """Format elapsed time in minutes and seconds."""
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {int(seconds)}s"


def get_round_path(output_path, round_num):
    """Get the path for a specific training round."""
    return os.path.join(output_path, f"round_{round_num}")

# ---------------------- Run ID and Directory Config Functions ----------------------

def _git_short():
    """Retrieve the short Git commit hash of the current HEAD."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "nogit"

def _hp_fingerprint() -> str:
    """Generate a short hash fingerprint from config parameters."""
    keys = [
        "student_model_name", "teacher_model_name", "learning_rate", "alpha",
        "kl_temperature", "per_device_train_batch_size", "gradient_accumulation_steps",
        "num_train_epochs", "total_rounds", "dataset_name", "dataset_type", "seed"
    ]
    blob = json.dumps({k: getattr(config, k) for k in keys}, sort_keys=True).encode()
    return blake2s(blob, digest_size=4).hexdigest()

def _abbr_model(name: str) -> str:
    """Abbreviate model name for concise identification."""
    base = name.split("/")[-1].lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*b", base)
    size = (m.group(1) + "b") if m else ""
    family = "qwen" if "qwen" in base else base.split("-")[0]
    return f"{family}{size.replace('.','p')}"

def build_run_identity():
    """Construct run identity components for logging and tracking."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    git = _git_short()
    fp = _hp_fingerprint()

    run_id = f"{ts}-{git}-{fp}"
    slug = "-".join([
        _abbr_model(config.student_model_name),
        f"a{config.alpha}",
        f"t{config.kl_temperature}",
        f"lr{config.learning_rate}"
    ])
    alias = os.environ.get("RUN_ALIAS")
    if alias:
        slug = f"{alias}-{slug}"

    wandb_name = f"{slug} | {run_id}"
    wandb_id = f"{fp}-{ts}"

    return run_id, slug, wandb_name, wandb_id

def get_directory(run_id):
    output_dir = config.base_output_dir
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir)
    return run_dir

# ---------------------- CSV Logger ----------------------

class CSVLogger:
    def __init__(
        self,
        log_dir,
        fieldnames: list,
        overall_start_time,
        filename: str = "CSV_metrics.csv",
        flush_every: int = 10,
    ):
        self.fieldnames = fieldnames
        self.overall_start_time = overall_start_time
        self.buffer = []
        self.flush_every = flush_every
        self.counter = 0

        if config.checkpoint_log_path:
            self.filepath = config.checkpoint_log_path
            if not os.path.exists(self.filepath):
                main_print(f"[WARNING] Checkpoint CSV file does not exist: {self.filepath}")
                sys.exit(1)
        else:
            if os.path.exists(os.path.join(log_dir, filename)) and not config.overwrite_csv:
                idx = 2
                while os.path.exists(os.path.join(log_dir, f"{filename_base}_{idx}.csv")):
                    idx += 1
                filename = f"{filename_base}_{idx}.csv"

            self.filepath = os.path.join(log_dir, filename)

            if not os.path.exists(self.filepath) or config.overwrite_csv:
                with open(self.filepath, mode="w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                    writer.writeheader()

    def log(self, **kwargs):
        """Log a row of data with automatic timestamping."""
        row = {key: kwargs.get(key, None) for key in self.fieldnames}
        row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row["overall_elapsed"] = time.time() - self.overall_start_time
        self.buffer.append(row)
        self.counter += 1
        if self.counter >= self.flush_every:
            self.flush()

    def flush(self):
        """Write the buffered log entries to file."""
        if not self.buffer:
            return
        with open(self.filepath, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerows(self.buffer)
        self.buffer.clear()
        self.counter = 0

# ---------------------- Manifest File Handling ----------------------


def _git_short():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "nogit"

def _write_txt(path, manifest_dict):
    lines = []
    for section, content in manifest_dict.items():
        if isinstance(content, dict):
            lines.append(f"[{section}]")
            for k, v in content.items():
                lines.append(f"{k}: {v}")
            lines.append("")  # blank line between sections
        else:
            lines.append(f"{section}: {content}")
    text = "\n".join(lines)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)

def _touch(path):
    open(path, "a").close()


def create_manifest(output_path, start_time_str=None, wandb_run=None, wandb_id=None):
        run_id = os.path.basename(output_path)
        manifest_path = os.path.join(output_path, "manifest.txt")

        # Status sentinels
        status_running = os.path.join(output_path, "STATUS.RUNNING")
        status_done    = os.path.join(output_path, "STATUS.DONE")
        status_failed  = os.path.join(output_path, "STATUS.FAILED")

        # Resolve hardware info
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = int(os.environ.get("WORLD_SIZE", 1))
        devices = [f"cuda:{i}" for i in range(world_size)]

        # Resolve checkpoint and metrics paths
        run_dir = output_path
        ckpt_dir_default = os.path.join(output_path, "checkpoints")
        checkpoint_dir = ckpt_dir_default if os.path.isdir(ckpt_dir_default) else getattr(config, "checkpoint_dir", None)
        metrics_csv = os.path.join(output_path, "CSV_metrics.csv")

        # Timestamps
        start_time_str = start_time_str or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        created_at_epoch = time.time()

        # Build a concise, structured manifest (will be written as text)
        manifest = {
            "run_id": run_id,
            "start_time": start_time_str,
            "paths": {
                "run_dir": run_dir,
                "checkpoint_dir": checkpoint_dir,
                "metrics_csv": metrics_csv,
            },
            "wandb": {
                "id": (wandb_id if wandb_id is not None else (getattr(wandb_run, "id", None) if wandb_run is not None else None)),
                "name": (getattr(wandb_run, "name", None) if wandb_run is not None else None),
                "project": "slm-ensembles",
            },
            "code": {
                "git_commit": _git_short(),
                "entrypoint": "main.py",
            },
            "hardware": {
                "ddp": getattr(config, "ddp", False),
                "world_size": world_size,
                "devices": devices,
            },
            "data": {
                "dataset_name": getattr(config, "dataset_name", None),
                "dataset_type": getattr(config, "dataset_type", None),
                "dataset_path": getattr(config, "dataset_path", None),
                "synthetic_data": getattr(config, "synthetic_data", False),
            },
            "models": {
                "teacher": getattr(config, "teacher_model_name", None),
                "student": getattr(config, "student_model_name", None),
                "tokenizer": getattr(config, "tokenizer_name", None),
                "student_vocab_size": getattr(config, "student_vocab_size", None),
            },
            "train": {
                "alpha": getattr(config, "alpha", None),
                "kl_temperature": getattr(config, "kl_temperature", None),
                "learning_rate": getattr(config, "learning_rate", None),
                "weight_decay": getattr(config, "weight_decay", None),
                "warmup_ratio": getattr(config, "warmup_ratio", None),
                "warmup_steps": getattr(config, "warmup_steps", None),
                "num_train_epochs": getattr(config, "num_train_epochs", None),
                "total_rounds": getattr(config, "total_rounds", None),
                "steps_per_round": getattr(config, "steps_per_round", None),
                "per_device_train_batch_size": getattr(config, "per_device_train_batch_size", None),
                "eval_batch_size": getattr(config, "eval_batch_size", None),
                "gradient_accumulation_steps": getattr(config, "gradient_accumulation_steps", None),
                "max_grad_norm": getattr(config, "max_grad_norm", None),
                "eval_steps": getattr(config, "eval_steps", None),
                "logging_steps": getattr(config, "logging_steps", None),
                "save_steps": getattr(config, "save_steps", None),
                "save_total_limit": getattr(config, "save_total_limit", None),
                "seed": getattr(config, "seed", None),
                "description": getattr(config, "description", None),
                "id_string": getattr(config, "id_string", None),
            },
            # Filled in at the end:
            "outcomes": {
                "end_time": None,
                "wall_time_sec": None,
                "best_checkpoints": None,
                "min_eval_loss": None,
            },
            "status": "RUNNING",
        }

        _write_txt(manifest_path, manifest)
        _touch(status_running)

        # Ensure we flip RUNNING → DONE/FAILED at exit
        def _finalize_manifest(status_ok: bool, end_time: str, wall_time_sec: float, best_ckpts=None, min_eval=None):
            try:
                manifest["outcomes"]["end_time"] = end_time
                manifest["outcomes"]["wall_time_sec"] = wall_time_sec
                manifest["outcomes"]["best_checkpoints"] = best_ckpts
                manifest["outcomes"]["min_eval_loss"] = float(min_eval) if min_eval is not None else None
                manifest["status"] = "DONE" if status_ok else "FAILED"
                _write_txt(manifest_path, manifest)
                # flip sentinel
                if status_ok:
                    if os.path.exists(status_failed): os.remove(status_failed)
                    if os.path.exists(status_running): os.remove(status_running)
                    _touch(status_done)
                else:
                    if os.path.exists(status_done): os.remove(status_done)
                    if os.path.exists(status_running): os.remove(status_running)
                    _touch(status_failed)
            except Exception:
                # Last resort: at least flip a file
                if status_ok:
                    _touch(status_done)
                else:
                    _touch(status_failed)

        # Register atexit success path; we’ll override to FAILED on exceptions
        def _on_exit_success():
            _finalize_manifest(
                status_ok=True,
                end_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                wall_time_sec=time.time() - base_start_time,
                best_ckpts=None,   # filled later if available
                min_eval=None,     # filled later if available
            )
        atexit.register(_on_exit_success)













# ---------------------- Dataset handling ----------------------

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
  
def prepare_dataset(train_ds, eval_ds, config, max_length, seed):
    """Prepare datasets with distributed samplers for FSDP2 training."""

    dc = CustomPadCollator(max_length, 
                           pad_token_id=151699, 
                           pad_label_id=-100, 
                           pad_attention_id=0)

    train_sampler = DistributedSampler(
        train_ds,
        dist.get_world_size(),
        dist.get_rank(),
        shuffle=True,
        seed=seed,
    )
    test_sampler = DistributedSampler(
        eval_ds,
        dist.get_world_size(),
        dist.get_rank(),
        shuffle=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.student_model_name)

    train_dataloader = DataLoader(
        train_ds,
        batch_size=config.per_device_train_batch_size,
        sampler=train_sampler,
        shuffle=False,
        collate_fn=dc,
        num_workers=8,
        persistent_workers=False
    )
    eval_dataloader = DataLoader(
        eval_ds,
        batch_size=config.eval_batch_size,
        sampler=test_sampler,
        shuffle=False,
        collate_fn=dc,
        num_workers=8,
        persistent_workers=False
    )
    
    return train_dataloader, eval_dataloader
    
class DistillDataset:
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = self.get_dataset()

    def get_dataset(self):
        """Load the base dataset."""
        if config.synthetic_data:
            dataset = datasets.load_from_disk(config.synthetic_dataset_path)
        else:
            dataset = datasets.load_from_disk(config.dataset_path)

        if config.dataset_type == "single":
            return {
                "train": dataset["train"].select([0]),
                "test": dataset["test"].select([0]),
            }
        elif config.dataset_type == "batch":
            return {
                "train": dataset["train"].select(range(10)),
                "test": dataset["test"].select(range(10)),
            }
        return dataset

    def get_teacher_logprobs(self):
        """Load or generate teacher logits dataset."""
        if not os.path.exists(os.path.join(config.logprob_cache_path, "teacher_logprobs")):
            self.cache_teacher_logprobs()

        main_print("--> Loading Teacher Logits")
        return datasets.load_from_disk(os.path.join(config.logprob_cache_path, "teacher_logprobs"))

    def concatenate_logprobs_chunks(self, split_dirs: list[str]):
        """Concatenate logit chunks into a single dataset."""
        datasets_list = []
        for split_dir in split_dirs:
            chunk_paths = sorted(
                [p for p in glob.glob(os.path.join(split_dir, "chunk_*")) if os.path.isdir(p)],
                key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]),
            )
            main_print(f"--> Loading {len(chunk_paths)} chunks for '{os.path.basename(split_dir)}' split")
            datasets_list.extend(load_from_disk(p) for p in chunk_paths)
        return concatenate_datasets(datasets_list) if datasets_list else None

    def build_teacher_logprobs_dataset(self):
        """Build the final teacher logprobs dataset from chunks."""
        dict = {}

        for split in ["train", "test"]:
            split_dirs = sorted(
                [d for d in glob.glob(os.path.join(config.logprob_cache_path, f"teacher_logprobs_{split}_*")) if os.path.isdir(d)]
            )
            split_ds = self.concatenate_logprobs_chunks(split_dirs)
            dict[split] = split_ds

        dataset = DatasetDict(dict)
        combined_path = os.path.join(config.logprob_cache_path, "teacher_logprobs")
        dataset.save_to_disk(combined_path)

        return combined_path

    def cache_teacher_logprobs(self):
        if config.ddp and not dist.is_initialized():
            dist.init_process_group("nccl")
            main_print(f"Using {torch.distributed.get_backend()} backend")

        rank = dist.get_rank() if config.ddp else 0
        world_size = dist.get_world_size() if config.ddp else 1
        print(f"Rank: {rank}")
        print(f"World size: {world_size}")

        torch.cuda.set_device(rank)

        teacher_model = AutoModelForCausalLM.from_pretrained(
            config.teacher_model_name,
            torch_dtype=torch.bfloat16,
        ).to(f"cuda:{rank}")
        teacher_model.resize_token_embeddings(new_num_tokens=config.student_vocab_size)
        teacher_model.requires_grad_(False)

        main_print("\n--> Generating Teacher Logits")
        for split in ["train", "test"]:

            shard = self.dataset[split].shard(num_shards=world_size, index=rank)
            save_dir = os.path.join(config.logprob_cache_path, f"teacher_logprobs_{split}_rank{rank}")
            os.makedirs(save_dir, exist_ok=True)

            save_ds = {"input_ids": [], "attention_mask": [], "labels": [], "logprob_values": [], "logprob_indices": [], "start_idx": [], "end_idx": []}
            chunk_id = 0

            batch_size = 32  # tune this to your GPU
            batch_data = {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
            }

            with torch.no_grad():
                tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
                for idx, sample in tqdm(enumerate(shard), total=len(shard)):
                    batch_data["input_ids"].append(sample["input_ids"])
                    batch_data["attention_mask"].append(sample["attention_mask"])
                    batch_data["labels"].append(sample["labels"])
                    
                    if len(batch_data["input_ids"]) == batch_size or idx == len(shard) - 1:
                        input_ids = torch.stack(batch_data["input_ids"]).to(f"cuda:{rank}")
                        attention_mask = torch.stack(batch_data["attention_mask"]).to(f"cuda:{rank}")

                        outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = outputs.logits.squeeze(0)  # [sample, 1024, 151000]
                        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

                        values, indices = torch.topk(logprobs, k=100, dim=-1)
                        values = values.to('cpu')                           # BF16
                        indices = indices.to(torch.int32).to('cpu')         # INT32

                        if len(values.shape) == 2:
                            start = torch.where(batch_data['labels'][0] != -100)[0]
                            start_idx = start[0].item() if len(start) != 0 else 0

                            end = torch.where(batch_data['input_ids'][0] == tokenizer.pad_token_id)[0]
                            end_idx = end[0].item() if len(end) != 0 else len(batch_data['input_ids'][0]) - 1

                            save_ds["input_ids"].append(batch_data["input_ids"][0][:end_idx])
                            save_ds["attention_mask"].append(batch_data["attention_mask"][0][:end_idx])
                            save_ds["labels"].append(batch_data["labels"][0][:end_idx])
                            save_ds["logprob_values"].append(values[start_idx:end_idx])
                            save_ds["logprob_indices"].append(indices[start_idx:end_idx])
                            save_ds["start_idx"].append(start_idx)
                            save_ds["end_idx"].append(end_idx)
                        else:
                            for b in range(values.size(0)):
                                start = torch.where(batch_data['labels'][b] != -100)[0] 
                                if len(start) == 0:
                                    continue
                                start_idx = start[0].item()
                                
                                end = torch.where(batch_data['input_ids'][b] == tokenizer.pad_token_id)[0]
                                end_idx = end[0].item() if len(end) != 0 else len(batch_data['input_ids'][b]) - 1
                                
                                save_ds["input_ids"].append(batch_data["input_ids"][b][:end_idx])
                                save_ds["attention_mask"].append(batch_data["attention_mask"][b][:end_idx])
                                save_ds["labels"].append(batch_data["labels"][b][:end_idx])
                                save_ds["logprob_values"].append(values[b][start_idx:end_idx])
                                save_ds["logprob_indices"].append(indices[b][start_idx:end_idx])
                                save_ds["start_idx"].append(start_idx)
                                save_ds["end_idx"].append(end_idx)

                        batch_data = {"input_ids": [], "attention_mask": [], "labels": []}

                        if (idx + 1) % 3200 < batch_size or idx == len(shard) - 1:
                            main_print(f"--> [{split}] Saving chunk {chunk_id} with {len(save_ds['input_ids'])} samples")

                            save_path = os.path.join(save_dir, f"chunk_{chunk_id}.arrow")
                            if os.path.exists(save_path):
                                shutil.rmtree(save_path)
                            Dataset.from_dict(save_ds).save_to_disk(save_path)

                            save_ds = {
                                "input_ids": [],
                                "attention_mask": [],
                                "labels": [],
                                "logprob_values": [],
                                "logprob_indices": [],
                                "start_idx": [],
                                "end_idx": [],
                            }
                            chunk_id += 1

                    # if (idx + 1) % 3200 == 0:
                    #     break

            if save_ds["input_ids"]:
                print(f"--> [{split}] Saving final chunk {chunk_id} with {len(save_ds['input_ids'])} samples")
                save_path = os.path.join(save_dir, f"chunk_{chunk_id}.arrow")
                if os.path.exists(save_path):
                    shutil.rmtree(save_path)
                Dataset.from_dict(save_ds).save_to_disk(save_path)
        
        dist.barrier() if config.ddp else None

        if _get_rank() == 0:
            self.build_teacher_logprobs_dataset()
        
        main_print("\n--> Generation Done")
        dist.barrier() if config.ddp else None

# ---------------------- Main execution for testing ----------------------
if __name__ == "__main__":
    import torch
    import time
    import pdb
    import datasets
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import DataCollatorForCompletionOnlyLM
    import config

    main_print("--> Evaluate model")

    device = torch.cuda.current_device()
    main_print(device)

    dataset = datasets.load_from_disk("/scratch/klambert/dataset/tulu-3-sft-mixture-pretokenized")
    logit_dataset = datasets.load_from_disk("/scratch/klambert/slm_ensembles/teacher_logits/teacher_logits")

    tokenizer = AutoTokenizer.from_pretrained(config.student_model_name)
    response_template_ids = tokenizer("<|im_start|>assistant\n")["input_ids"]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    student_model = AutoModelForCausalLM.from_pretrained(
        config.student_model_name,
        torch_dtype=torch.bfloat16,
    ).to(device)

