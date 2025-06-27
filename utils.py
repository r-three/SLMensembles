# utils.py
import os, csv, time, glob, sys, tqdm
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import config
from datetime import datetime
import time
import config


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
        elif not os.path.exists(self.filepath):
            with open(self.filepath, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
        else:
            print(f"[ERROR] Log file {self.filepath} already exists. Aborting to prevent overwrite.")
            sys.exit(1)

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


class TeacherLogits():
    def __init__(self):
        
    @staticmethod
    def cache_teacher_logits(self, dataset, n: int = 1000):
        logit_values = []

        with torch.no_grad():
            for idx, sample in enumerate(tqdm(dataset["train"], desc="Caching Teacher Logits")):
                input_ids = sample["input_ids"].unsqueeze(0).to(config.teacher_device)
                attention_mask = sample["attention_mask"].unsqueeze(0).to(config.teacher_device)
                outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)

                logits = outputs.logits.squeeze(0).cpu()
                logit_values.append(logits)

                if idx >= n:
                    break

        torch.save(logit_values, os.path.join(config.logit_cache_path, "teacher_logits.pt"))

    @staticmethod
    def load_teacher_logits(self):
        print("\n--> Loading Teacher Logits")
        teacher_logits = np.load(os.path.join(config.logit_cache_path, "teacher_logits.npy"))
        print("\n--> Loading Done")


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
