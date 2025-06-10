# utils.py
import os, csv, time, glob
from datetime import datetime
import torch
from torch.utils.data import DataLoader

class CSVLogger:
    def __init__(self, log_dir, fieldnames, filename="metrics.csv", custom_path=None):
        os.makedirs(log_dir, exist_ok=True)
        if custom_path is None:
            run_dirs = glob.glob(os.path.join(log_dir, "run_*"))
            next_run = max([int(os.path.basename(d).split("_")[1]) for d in run_dirs if "_" in d] + [0]) + 1
            filename = f"run_{next_run}_{filename}"
        else:
            filename = f"{custom_path}_metrics.csv"
        self.filepath = os.path.join(log_dir, filename)
        self.fieldnames = fieldnames

        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

    def log(self, **kwargs):
        row = {key: kwargs.get(key, None) for key in self.fieldnames}
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

def format_time_elapsed(seconds):
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {int(seconds)}s"

def get_round_path(output_path, round_num):
    return os.path.join(output_path, f"round_{round_num}")

def evaluate_model(model, eval_dataset, collator, device, batch_size, round_num=None):
    model.eval()
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collator)
    total_loss, total_tokens = 0, 0

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
