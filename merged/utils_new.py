import os, csv, time, glob, sys
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

try:
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.tensor import Shard, DTensor
    FSDP2_AVAILABLE = True
except ImportError:
    FSDP2_AVAILABLE = False
    FSDPModule = None
    DTensor = None


# =============================================================================
# FSDP2-specific utility functions
# =============================================================================

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


def prepare_dataset(train_ds, eval_ds, config, response_template_ids, seed):
    """Prepare datasets with distributed samplers for FSDP2 training."""
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
    collate_fn = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    train_dataloader = DataLoader(
        train_ds,
        batch_size=config.per_device_train_batch_size,
        sampler=train_sampler,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=8,
        persistent_workers=False
    )
    eval_dataloader = DataLoader(
        eval_ds,
        batch_size=config.eval_batch_size,
        sampler=test_sampler,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=8,
        persistent_workers=False
    )
    
    return train_dataloader, eval_dataloader


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


# =============================================================================
# Core utility functions (shared between both approaches)
# =============================================================================

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


def evaluate_model(model, eval_dataset, collator):
    """Evaluate model on a dataset and return loss and perplexity."""
    model.eval()
    dataloader = DataLoader(eval_dataset, batch_size=config.eval_batch_size, collate_fn=collator)
    total_loss, total_tokens = 0, 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            valid_tokens = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * valid_tokens
            total_tokens += valid_tokens

    avg_loss = total_loss / total_tokens if total_tokens else float("inf")
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return {"eval_loss": avg_loss, "perplexity": perplexity}


# =============================================================================
# CSV Logger (enhanced version from main codebase)
# =============================================================================

class CSVLogger:
    def __init__(
        self,
        log_dir,
        fieldnames: list,
        overall_start_time,
        filename: str = "metrics.csv",
        flush_every: int = 10,
    ):
        os.makedirs(log_dir, exist_ok=True)

        self.fieldnames = fieldnames
        self.overall_start_time = overall_start_time
        self.buffer = []
        self.flush_every = flush_every
        self.counter = 0

        run_dirs = glob.glob(os.path.join(log_dir, "run_*"))
        next_run = max([int(os.path.basename(d).split("_")[1]) for d in run_dirs if "_" in d] + [0]) + 1

        if config.custom_run_name is None:
            filename = f"run_{next_run}_{filename}"
        else:
            filename = f"{config.custom_run_name}_metrics.csv"

        self.filepath = os.path.join(log_dir, filename)

        if config.checkpoint_log_path:
            # change the specified log dir to the one corresponding to the checkpointed model
            self.filepath = config.checkpoint_log_path
            if not os.path.exists(self.filepath):
                main_print(f"[WARNING] Checkpoint CSV file does not exist: {self.filepath}")
                sys.exit(1)
        elif os.path.exists(self.filepath) and not config.overwrite_csv:
            main_print(f"[ERROR] Log file {self.filepath} already exists. Aborting to prevent overwrite.")
            sys.exit(1)
        else:
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


# =============================================================================
# Dataset handling (enhanced version from main codebase)
# =============================================================================

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
        datasets_to_concat = []
        for chunk_dir in split_dirs:
            if os.path.exists(chunk_dir):
                datasets_to_concat.append(datasets.load_from_disk(chunk_dir))
        
        if datasets_to_concat:
            return concatenate_datasets(datasets_to_concat)
        return None

    def build_teacher_logprobs_dataset(self):
        """Build the final teacher logprobs dataset from chunks."""
        for split in ["train", "test"]:
            split_dir = os.path.join(config.logprob_cache_path, "teacher_logprobs_chunks", split)
            chunk_dirs = glob.glob(os.path.join(split_dir, "chunk_*"))
            chunk_dirs.sort(key=lambda x: int(x.split("_")[-1]))
            
            concatenated_dataset = self.concatenate_logprobs_chunks(chunk_dirs)
            if concatenated_dataset:
                save_path = os.path.join(config.logprob_cache_path, "teacher_logprobs", split)
                if os.path.exists(save_path):
                    shutil.rmtree(save_path)
                concatenated_dataset.save_to_disk(save_path)
                main_print(f"--> [{split}] Saved {len(concatenated_dataset)} teacher logit samples")

    def cache_teacher_logits(self):
        """Generate and cache teacher logits for the dataset."""
        main_print("--> Generating Teacher Logits")
        
        # Load teacher models
        teacher_models = []
        for model_name in config.teacher_model_names:
            main_print(f"Loading teacher model: {model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto" if not config.ddp else None,
            )
            if config.ddp:
                model = model.to(self.device)
            model.eval()
            teacher_models.append(model)

        tokenizer = AutoTokenizer.from_pretrained(config.student_model_name)
        
        # Process each split
        for split in ["train", "test"]:
            main_print(f"--> Processing {split} split")
            
            save_dir = os.path.join(config.logit_cache_path, "teacher_logits_chunks", split)
            os.makedirs(save_dir, exist_ok=True)
            
            shard = self.dataset[split]
            if config.ddp:
                # Distribute work across processes
                rank = _get_rank()
                world_size = dist.get_world_size()
                shard_size = len(shard) // world_size
                start_idx = rank * shard_size
                end_idx = start_idx + shard_size if rank < world_size - 1 else len(shard)
                shard = shard.select(range(start_idx, end_idx))
            
            batch_size = config.per_device_train_batch_size
            chunk_id = 0
            
            save_ds = {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
                "logit_values": [],
                "logit_indices": [],
            }
            
            batch_data = {"input_ids": [], "attention_mask": [], "labels": []}
            
            for idx, example in enumerate(tqdm(shard)):
                batch_data["input_ids"].append(example["input_ids"])
                batch_data["attention_mask"].append(example["attention_mask"])
                batch_data["labels"].append(example["labels"])
                
                if len(batch_data["input_ids"]) == batch_size or idx == len(shard) - 1:
                    # Process batch
                    input_ids = torch.tensor(batch_data["input_ids"]).to(self.device)
                    attention_mask = torch.tensor(batch_data["attention_mask"]).to(self.device)
                    labels = torch.tensor(batch_data["labels"]).to(self.device)
                    
                    # Get teacher logits
                    with torch.no_grad():
                        teacher_logits_list = []
                        for teacher_model in teacher_models:
                            outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                            teacher_logits_list.append(outputs.logits)
                        
                        # Average teacher logits
                        avg_teacher_logits = torch.stack(teacher_logits_list).mean(dim=0)
                        
                        # Get top-k logits for memory efficiency
                        top_k = getattr(config, 'teacher_logit_top_k', 100)
                        values, indices = torch.topk(avg_teacher_logits, k=top_k, dim=-1)
                    
                    # Save batch results
                    for b in range(input_ids.shape[0]):
                        save_ds["input_ids"].append(input_ids[b].cpu())
                        save_ds["attention_mask"].append(attention_mask[b].cpu())
                        save_ds["labels"].append(labels[b].cpu())
                        save_ds["logit_values"].append(values[b].cpu())
                        save_ds["logit_indices"].append(indices[b].cpu())
                    
                    batch_data = {"input_ids": [], "attention_mask": [], "labels": []}
                    
                    if (idx + 1) % 800 < batch_size:
                        main_print(f"--> [{split}] Generated {idx + 1} Teacher Logits")
                    
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
                            "logit_values": [],
                            "logit_indices": [],
                        }
                        chunk_id += 1
            
            # Save any remaining data
            if save_ds["input_ids"]:
                main_print(f"--> [{split}] Saving final chunk {chunk_id} with {len(save_ds['input_ids'])} samples")
                save_path = os.path.join(save_dir, f"chunk_{chunk_id}.arrow")
                if os.path.exists(save_path):
                    shutil.rmtree(save_path)
                Dataset.from_dict(save_ds).save_to_disk(save_path)
        
        # Synchronize processes
        dist.barrier() if config.ddp else None
        
        # Build final dataset on main process
        if _get_rank() == 0:
            self.build_teacher_logits_dataset()
        
        main_print("\n--> Teacher logit generation complete")
        dist.barrier() if config.ddp else None


# =============================================================================
# Main execution for testing
# =============================================================================

if __name__ == "__main__":
    import torch
    import time
    import pdb
    import datasets
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import DataCollatorForCompletionOnlyLM
    import config

    main_print("--> Testing merged utils")

    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
    main_print(f"Using device: {device}")

    # Test dataset loading
    try:
        dataset = datasets.load_from_disk("/scratch/klambert/dataset/tulu-3-sft-mixture-pretokenized")
        main_print(f"Loaded dataset with {len(dataset['train'])} train samples")
    except:
        main_print("Could not load test dataset")

    # Test model evaluation if dataset is available
    if 'dataset' in locals():
        tokenizer = AutoTokenizer.from_pretrained(config.student_model_name)
        response_template_ids = tokenizer("1assistant\n")["input_ids"]
        collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

        try:
            student_model = AutoModelForCausalLM.from_pretrained(
                config.student_model_name,
                torch_dtype=torch.bfloat16,
            ).to(device)

            main_print("Testing model evaluation on small sample")
            small_test = dataset["test"].select(range(10))
            results = evaluate_model(student_model, small_test, collator)
            main_print(f"Evaluation results: {results}")
        except Exception as e:
            main_print(f"Could not test model evaluation: {e}")
