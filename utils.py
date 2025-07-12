import os, csv, time, glob, sys
from datetime import datetime
import pdb
import shutil
import torch
import datasets
import torch.distributed as dist
from torch.utils.data import DataLoader
import config
from transformers import AutoModelForCausalLM
from datasets import load_from_disk, DatasetDict, concatenate_datasets, Dataset


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
                print(f"[WARNING] Checkpoint CSV file does not exist: {self.filepath}")
                sys.exit(1)
        elif os.path.exists(self.filepath) and not config.overwrite_csv:
            print(f"[ERROR] Log file {self.filepath} already exists. Aborting to prevent overwrite.")
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
    def __init__(self, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = self.get_dataset()
        # if not config.synthetic_data:
        #     self.teacher_model = AutoModelForCausalLM.from_pretrained(
        #         config.teacher_model_name,
        #         torch_dtype=torch.bfloat16,
        #     ).to(self.device)
        #     self.teacher_model.resize_token_embeddings(new_num_tokens=config.student_vocab_size)
        #     self.teacher_model.requires_grad_(False)
        # else:
        #     self.teacher_model = None

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

    def get_teacher_logits(self):
        if not os.path.exists(os.path.join(config.logit_cache_path, "teacher_logits")):
            self.cache_teacher_logits()

        print("--> Loading Teacher Logits")
        logit_values = load_from_disk(os.path.join(config.logit_cache_path, "teacher_logits"))

        print("--> Loading Done")
        return logit_values

    def concatenate_logit_chunks(self, split_dirs: list[str]):
        datasets_list = []
        for split_dir in split_dirs:
            chunk_paths = sorted(
                [p for p in glob.glob(os.path.join(split_dir, "chunk_*")) if os.path.isdir(p)],
                key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]),
            )
            print(f"--> Loading {len(chunk_paths)} chunks for '{os.path.basename(split_dir)}' split")
            datasets_list.extend(load_from_disk(p) for p in chunk_paths)
        combined = concatenate_datasets(datasets_list)
        return combined

    def build_teacher_logits_dataset(self):
        print(f"--> Assembling full teacher-logits dataset")
        dict = {}

        for split in ["train", "test"]:
            split_dirs = sorted(
                [d for d in glob.glob(os.path.join(config.logit_cache_path, f"teacher_logits_{split}_*")) if os.path.isdir(d)]
            )
            split_ds = self.concatenate_logit_chunks(split_dirs)

            dict[split] = split_ds

        dataset = DatasetDict(dict)
        combined_path = os.path.join(config.logit_cache_path, "teacher_logits")
        dataset.save_to_disk(combined_path)

        print(f"--> Full dataset saved to {combined_path}")
        return combined_path

    def cache_teacher_logits(self):
        if not dist.is_initialized():
            dist.init_process_group("nccl")

        print(f"Using {torch.distributed.get_backend()} backend")

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"Rank: {rank}")
        print(f"World size: {world_size}")

        torch.cuda.set_device(rank)
        self.teacher_model.to(f"cuda:{rank}")

        print("\n--> Generating Teacher Logits")
        for split in ["test"]:

            shard = self.dataset[split].shard(num_shards=world_size, index=rank)
            save_dir = os.path.join(config.logit_cache_path, f"teacher_logits_{split}_rank{rank}")
            os.makedirs(save_dir, exist_ok=True)

            save_ds = {"input_ids": [], "attention_mask": [], "labels": [], "logit_values": [], "logit_indices": []}
            chunk_id = 0

            with torch.no_grad():
                for idx, sample in enumerate(shard):
                    input_ids = sample["input_ids"].unsqueeze(0).to(self.device)
                    attention_mask = sample["attention_mask"].unsqueeze(0).to(self.device)
                    labels = sample["labels"].unsqueeze(0).to(self.device)

                    outputs = self.teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits.squeeze(0).cpu()  # [sample, 1024, 151000]

                    values, indices = torch.topk(logits, k=100, dim=-1)

                    save_ds["input_ids"].append(input_ids.squeeze(0).cpu())
                    save_ds["attention_mask"].append(attention_mask.squeeze(0).cpu())
                    save_ds["labels"].append(labels.squeeze(0).cpu())
                    save_ds["logit_values"].append(values.cpu())
                    save_ds["logit_indices"].append(indices.cpu())

                    if (idx + 1) % 1000 == 0:
                        print(f"--> [{split}] Generated {idx} Teacher Logits")

                    if (idx + 1) % 3000 == 0 or (idx == len(shard) - 1):
                        print(f"--> [{split}] Saving chunk {chunk_id} with {len(save_ds['input_ids'])} samples")

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

                    # if (idx + 1) % 3000 == 0:
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

        dist.barrier()

        if dist.get_rank() == 0:
            self.build_teacher_logits_dataset()

        print("\n--> Generation Done")
        dist.barrier()


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
        for batch in dataloader:
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


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


if __name__ == "__main__":
    print("--> Loading Dataset and Caching Logits")

    dataClass = DistillDataset()
    dataClass.build_teacher_logits_dataset()
