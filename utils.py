import os, csv, time, glob, sys
from datetime import datetime
import pdb
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
        if not config.synthetic_data:
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                config.teacher_model_name,
                torch_dtype=torch.bfloat16,
            ).to(self.device)
            self.teacher_model.resize_token_embeddings(new_num_tokens=config.student_vocab_size)
            self.teacher_model.requires_grad_(False)
        else:
            self.teacher_model = None

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

        print("\n--> Loading Teacher Logits")
        logit_values = load_from_disk(os.path.join(config.logit_cache_path, "teacher_logits"))

        print(f"Teacher Logits:")
        print(logit_values)
        print(logit_values["train"])
        print(logit_values["test"])

        print("\n--> Loading Done")
        return logit_values

    def concatenate_logit_chunks(self, split_dir: str):
        chunk_paths = sorted(
            [p for p in glob.glob(os.path.join(split_dir, "chunk_*")) if os.path.isdir(p)],
            key=lambda p: int(os.path.basename(p).split("_")[-1].replace(".arrow", "")),
        )
        print(f"--> Loading {len(chunk_paths)} chunks for '{os.path.basename(split_dir)}' split")

        datasets_list = [load_from_disk(p) for p in chunk_paths]
        combined = concatenate_datasets(datasets_list)
        return combined

    def build_teacher_logits_dataset(self):
        print(f"--> Saving Dataset")

        train_dir = os.path.join(config.logit_cache_path, "teacher_logits_train")
        test_dir = os.path.join(config.logit_cache_path, "teacher_logits_test")

        train_ds = self.concatenate_logit_chunks(train_dir)
        test_ds = self.concatenate_logit_chunks(test_dir)

        dataset = DatasetDict({"train": train_ds, "test": test_ds})

        combined_path = os.path.join(config.logit_cache_path, "teacher_logits")
        dataset.save_to_disk(combined_path)
        print(f"--> Full dataset saved to {combined_path}")

        return combined_path

    def cache_teacher_logits(self):
        with torch.no_grad():
            print("\n--> Generating Teacher Logits")
            for split in ["train", "test"]:
                save_ds = {"input_ids": [], "attention_mask": [], "labels": [], "logit_values": [], "logit_indices": []}
                save_dir = os.path.join(config.logit_cache_path, f"teacher_logits_{split}")
                os.makedirs(save_dir, exist_ok=True)

                for idx, sample in enumerate(self.dataset[split]):
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

                    if (idx + 1) % 200 == 0:
                        print(f"\n--> [{split}] Generated {idx} Teacher Logits")

                    if (idx + 1) % 3000 == 0 or idx == len(self.dataset[split]) - 1:
                        save_id = idx // 3000
                        file_path = os.path.join(save_dir, f"chunk_{save_id}.arrow")
                        print(f"--> [{split}] Saving chunk {save_id} with {len(save_ds['input_ids'])} samples")

                        save_dataset = Dataset.from_dict(save_ds)
                        save_dataset.save_to_disk(file_path)

                        # Reset
                        save_ds = {
                            "input_ids": [],
                            "attention_mask": [],
                            "labels": [],
                            "logit_values": [],
                            "logit_indices": [],
                        }

            self.build_teacher_logits_dataset()
            print("\n--> Generation Done")


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
    teacher_logits = dataClass.cache_teacher_logits()
