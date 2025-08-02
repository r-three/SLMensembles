import torch
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import Shard

import os, csv, time, glob, sys
from datetime import datetime
import pdb
from tqdm import tqdm
import shutil
import datasets
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import config
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk, DatasetDict, concatenate_datasets, Dataset
from torch.distributed.tensor import DTensor
from trl import DataCollatorForCompletionOnlyLM

# datasets.config.IN_MEMORY_MAX_SIZE

def inspect_model(model: FSDPModule):
    # assert isinstance(model, Transformer)
    assert isinstance(model, FSDPModule)

    if torch.distributed.get_rank() == 0:
        print(model)

    for param in model.parameters():
        assert param.placements == (Shard(0),)
        # assert param.dtype == torch.float32
        assert isinstance(param, DTensor)
        # print(param.get_local_tensor())


def inspect_mixed_precision(model: FSDPModule):
    model.unshard()
    for param in model.parameters(recurse=False):
        assert param.dtype == torch.bfloat16
    model.reshard()

def main_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)

def _get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def is_main_process() -> bool:
    return _get_rank() == 0 if config.ddp else True


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


class DistillDataset:
    def __init__(self):
        self.dataset = self.get_dataset()

    def get_dataset(self):
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
        if not os.path.exists(os.path.join(config.logprob_cache_path, "teacher_logprobs")):
            self.cache_teacher_logprobs()

        main_print("--> Loading Teacher Logits")
        logprob_values = load_from_disk(os.path.join(config.logprob_cache_path, "teacher_logprobs"))

        main_print("--> Loading Done")
        return logprob_values

    def concatenate_logprobs_chunks(self, split_dirs: list[str]):
        datasets_list = []
        for split_dir in split_dirs:
            chunk_paths = sorted(
                [p for p in glob.glob(os.path.join(split_dir, "chunk_*")) if os.path.isdir(p)],
                key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]),
            )
            main_print(f"--> Loading {len(chunk_paths)} chunks for '{os.path.basename(split_dir)}' split")
            datasets_list.extend(load_from_disk(p) for p in chunk_paths)
        combined = concatenate_datasets(datasets_list)
        return combined

    def build_teacher_logprobs_dataset(self):
        main_print(f"--> Assembling full teacher-logprobs dataset")
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

        main_print(f"--> Full dataset saved to {combined_path}")
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
        for split in ["test"]:

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
                for idx, sample in tqdm(enumerate(shard), total=len(shard)):
                    batch_data["input_ids"].append(sample["input_ids"])
                    batch_data["attention_mask"].append(sample["attention_mask"])
                    batch_data["labels"].append(sample["labels"])
                    
                    if len(batch_data["input_ids"]) == batch_size or idx == len(shard) - 1:
                        input_ids = torch.stack(batch_data["input_ids"]).to(f"cuda:{rank}")
                        attention_mask = torch.stack(batch_data["attention_mask"]).to(f"cuda:{rank}")

                        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

                        outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = outputs.logits.squeeze(0)  # [sample, 1024, 151000]
                        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

                        values, indices = torch.topk(logprobs, k=100, dim=-1)
                        values = values.to('cpu')                           # BF16
                        indices = indices.to(torch.int32).to('cpu')         # INT32

                        if len(values.shape) == 2:
                            start = torch.where(batch_data['labels'][0] != -100)[0]
                            if len(start) == 0:
                                start_idx = 0
                            else:
                                start_idx = start[0].item()
                            end = torch.where(batch_data['input_ids'][0] == tokenizer.pad_token_id)[0]
                            if len(end) == 0:
                                end_idx = len(batch_data['input_ids'][0]) - 1
                            else:
                                end_idx = end[0].item()
                            save_ds["input_ids"].append(batch_data["input_ids"][0][:end_idx])
                            save_ds["attention_mask"].append(batch_data["attention_mask"][0][:end_idx])
                            save_ds["labels"].append(batch_data["labels"][0][:end_idx])
                            save_ds["logprob_values"].append(values[start_idx:end_idx])
                            save_ds["logprob_indices"].append(indices[start_idx:end_idx])
                            save_ds["start_idx"].append(start_idx)
                            save_ds["end_idx"].append(end_idx)
                        else:
                            for b in range(values.size(0)):
                                """
                                Truncate the labels, logprob_values, logprob_indices.
                                Exclude the logits for the chat template.
                                If the input_ids/labels have a valid sequence length of s, which includes <|im_end|>/n
                                We will pad again with the new collate function during training. 
                                """
                                start = torch.where(batch_data['labels'][b] != -100)[0]
                                if len(start) == 0:
                                    start_idx = 0
                                else:
                                    start_idx = start[0].item()
                                end = torch.where(batch_data['input_ids'][b] == tokenizer.pad_token_id)[0]
                                if len(end) == 0:
                                    end_idx = len(batch_data['input_ids'][b]) - 1
                                else:
                                    end_idx = end[0].item()        # This value should equal to where batch_data['attention_mask'][b] != 1
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

            # look into contents and size of the stored dataset
            # # of flops + utilization - measure the time and gpu usage
            # cumulative logit mass plot - pick adaptive k?

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
    
class custom_pad_collator:
    def __init__(self, max_length):
        self.max_length = max_length

    def __call__(self, batch):
        input_ids = []
        attention_mask = []
        labels = []

        for item in batch:
            length = len(item["input_ids"])
            pad_len = self.max_length - length
            # Pad input_ids and labels with -1
            input_ids.append(torch.cat([torch.tensor(item["input_ids"]), torch.full((pad_len,), -100)]))
            labels.append(torch.cat([torch.tensor(item["labels"]), torch.full((pad_len,), -100)]))

            # Pad attention_mask with 0
            attention_mask.append(torch.cat([torch.tensor(item["attention_mask"]), torch.zeros(pad_len)]))

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels)
        }


def prepare_dataset(train_ds, eval_ds, config, max_length, seed):
    # Writing seperately cuz the dataset may vary. Could replace with subclasses but too lazy. 
    
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

    train_dataloader = DataLoader(
        train_ds,
        batch_size=config.per_device_train_batch_size,
        sampler=train_sampler,
        shuffle=False,
        collate_fn=dc,
        num_workers=1,
        persistent_workers=False
    )
    eval_dataloader = DataLoader(
        eval_ds,
        batch_size=config.eval_batch_size,
        sampler=test_sampler,
        shuffle=False,
        collate_fn=dc,
        num_workers=1,
        persistent_workers=False
    )
    
    return train_dataloader, eval_dataloader


def format_time_elapsed(seconds):
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {int(seconds)}s"


def get_round_path(output_path, round_num):
    return os.path.join(output_path, f"round_{round_num}")


def evaluate_model(model, eval_dataset, collator):
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

def check_batch_shape(train_dataloader):
    temp_input_ids = next(iter(train_dataloader))['input_ids']
    temp_attention_mask = next(iter(train_dataloader))['attention_mask']
    temp_labels = next(iter(train_dataloader))['labels']
    # temp_logprob_values = next(iter(train_dataloader))['logprob_values']
    # temp_logprob_indices = next(iter(train_dataloader))['logprob_indices']
    print("shape input_ids: ", torch.tensor(temp_input_ids).shape)
    print("shape attention_mask: ", torch.tensor(temp_attention_mask).shape)
    # print("shape labels: ", torch.tensor(temp_labels).shape)
    # print("shape logprob_values: ", torch.tensor(temp_logprob_values).shape)
    # print("shape logprob_indices: ", torch.tensor(temp_logprob_indices).shape)