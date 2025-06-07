import gc
import os
import csv
import time
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets
from torch.utils.data import DataLoader
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from ensemble import ModelEnsemble
import config

n = 0
overall_start_time = None
teacher_model = None
ensemble_model = None
device = config.device
eval_batch_size = config.eval_batch_size


def format_time_elapsed(seconds):
    """Convert seconds to a readable format with minutes and seconds."""
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {int(seconds)}s"


def get_round_path(output_path, round_num):
    """Return path for a specific training round."""
    return os.path.join(output_path, f"round_{round_num}")


def evaluate_model(
    model, eval_dataset, collator, round_num, max_eval_samples=None, end=False
):
    if end == True:
        model.eval()
    if max_eval_samples:
        eval_dataset = torch.utils.data.Subset(eval_dataset, range(max_eval_samples))

    eval_dataloader = DataLoader(
        eval_dataset, config.eval_batch_size, collate_fn=collator
    )

    total_loss = 0
    total_tokens = 0
    total_examples = 0

    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            # Count non-masked tokens for proper averaging
            valid_tokens = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * valid_tokens
            total_tokens += valid_tokens
            total_examples += input_ids.size(0)

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "eval_loss": avg_loss,
        "perplexity": perplexity,
    }


class CSVLogger:
    def __init__(self, log_dir, fieldnames: list, filename: str = "metrics.csv"):
        os.makedirs(log_dir, exist_ok=True)

        existing_runs = []
        run_dirs = glob.glob(os.path.join(log_dir, "run_*"))
        next_run = 1
        for dir_path in run_dirs:
            try:
                run_num = int(os.path.basename(dir_path).split("_")[1])
                existing_runs.append(run_num)
            except (ValueError, IndexError):
                continue
        if existing_runs:
            next_run = max(existing_runs) + 1

        if config.custom_path is None:
            filename = f"run_{next_run}_{filename}"
        else:
            filename = f"{config.custom_path}_metrics.csv"
        self.filepath = os.path.join(log_dir, filename)

        self.fieldnames = fieldnames

        if os.path.exists(self.filepath):
            self.headers_written = True
        else:
            with open(self.filepath, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
            self.headers_written = True

    def log(
        self,
        function,
        phase,
        role,
        round_num,
        *,
        round_duration=None,
        step=None,
        train_loss=None,
        train_kl_loss=None,
        train_next_token_loss=None,
        eval_loss=None,
        eval_kl_loss=None,
        perplexity=None,
        grad_norm=None,
        learning_rate=None,
        alpha=None,
        tags=None,
        metadata=None,
    ):
        if tags is not None and not isinstance(tags, str):
            tags = "|".join(tags)  # list as "tag1|tag2|tag3"

        data = {
            "function": function,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "overall_elapsed": time.time() - overall_start_time,
            "round_duration": round_duration,
            "round": round_num,
            "ensemble_num": len(ensemble_model.models) if ensemble_model else 0,
            "phase": phase,
            "role": role,
            "step": step,
            "train_loss": train_loss,
            "train_kl_loss": train_kl_loss,
            "train_next_token_loss": train_next_token_loss,
            "eval_loss": eval_loss,
            "eval_kl_loss": eval_kl_loss,
            "grad_norm": grad_norm,
            "perplexity": perplexity,
            "learning_rate": learning_rate,
            "alpha": alpha,
            "tags": tags,
            "metadata": metadata,
        }

        row = {key: data.get(key, None) for key in self.fieldnames}

        with open(self.filepath, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


class LoggingCallback(TrainerCallback):
    def __init__(self, logger, round_num, overall_start_time):
        self.logger = logger
        self.round_num = round_num,
        self.overall_start_time = overall_start_time

    def on_prediction_step_end(self, args, state, control, **kwargs):
        loss = kwargs.get("loss", None)
        if loss is not None:
            self.logger.log(
                function="on_prediction_step_end",
                round_num=self.round_num,
                phase="eval",
                role="student",
                step=state.global_step,
                eval_loss=loss.mean().item(),
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        control = super().on_log(args, state, control, logs=logs, **kwargs)

        self.logger.log(
            function="on_log",
            round_num=self.round_num,
            phase="train",
            role="student",
            step=state.global_step,
            train_loss=logs.get("loss"),
            learning_rate=logs.get("learning_rate"),
            grad_norm=logs.get('grad_norm'),
        )

        return control


class DistillationTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.round_num = kwargs.pop("round_num")
        self.steps_per_round = kwargs.pop("steps_per_round")
        self.overall_start_time = kwargs.pop("overall_start_time")
        self.logger = kwargs.pop("logger")
        self.extra_logging_info = {}
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = inputs["labels"].to(device)
        
        # Get the teacher and ensemble predictions
        with torch.no_grad():
            teacher_logits = teacher_model(input_ids=input_ids, attention_mask=attention_mask).logits.to(device)

            ensemble_logits = None
            if ensemble_model is not None:
                ensemble_logits = ensemble_model(input_ids=input_ids, attention_mask=attention_mask).logits.to(device)

        current_model_output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        current_model_logits = current_model_output.logits
        next_token_loss = current_model_output.loss
        
        kl_loss = self.compute_kl_loss(current_model_logits, ensemble_logits, teacher_logits, labels != -100)
        hybrid_loss = (1 - config.alpha) * kl_loss + config.alpha * next_token_loss

        global n
        if n % self.args.logging_steps == 0:

            self.logger.log(
                function="compute_loss",
                round_num=self.round_num,
                phase="train",
                role="student",
                step=self.state.global_step,
                train_loss=hybrid_loss.item(),
                train_next_token_loss=next_token_loss.item(),
                train_kl_loss=kl_loss.item(),
                alpha=config.alpha,
                learning_rate=self._get_learning_rate(),
            )
        
        n += 1

        return (hybrid_loss, current_model_logits) if return_outputs else hybrid_loss

    def compute_kl_loss(self, student_logits, ensemble_logits, teacher_logits, mask, temperature=1.0):
        """Computes KL divergence loss between teacher and student model logits."""

        # Combines the model predictions with the ensemble
        if ensemble_logits is not None:
            num_models = len(ensemble_model.models)
            student_logits = student_logits / (num_models + 1) + ensemble_logits * (num_models / (num_models + 1))

        # Compute KL Loss
        student_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
        kl_loss = F.kl_div(student_probs, teacher_probs, log_target=True, reduction="none").sum(-1)
        return kl_loss[mask].mean()

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Model eval"""
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = inputs["labels"].to(device)

        # Compute_loss add this to calculate loss
        with torch.no_grad():
            student_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            teacher_logits = teacher_model(input_ids=input_ids, attention_mask=attention_mask).logits.to(device)
            
            if ensemble_model is not None:
                num_models = len(ensemble_model.models)
                ensemble_logits = ensemble_model(input_ids=input_ids, attention_mask=attention_mask).logits.detach()
                total_ensemble_logits = student_logits / (num_models + 1) + ensemble_logits * (num_models / (num_models + 1))
            else:
                ensemble_logits = None
                total_ensemble_logits = student_logits

        # Handle potential model wrapping (DataParallel/DistributedDataParallel)
        if hasattr(model, "module"):
            model = model.module

        # next token prediction loss 
        loss = model.loss_function(
            logits=total_ensemble_logits,
            labels=labels,
            vocab_size=model.config.vocab_size,
        )
        
        global n
        if n % self.args.logging_steps == 0:

            kl_loss = self.compute_kl_loss(student_logits, ensemble_logits, teacher_logits, labels != -100)
            if "kl_losses" not in self.extra_logging_info:
                self.extra_logging_info["kl_losses"] = []
            self.extra_logging_info["kl_losses"].append(kl_loss.item())

            self.logger.log(
                function="prediction_step",
                round_num=self.round_num,
                phase="eval",
                role="student",
                step=self.state.global_step,
                eval_loss=loss.item(),
                eval_kl_loss=kl_loss.item(),
            )
        
        n += 1

        return (loss, None, None) if prediction_loss_only else (loss, student_logits, labels)
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
        return output
   

def main():
    global teacher_model, ensemble_model, overall_start_time, n

    overall_start_time = time.time()
    overall_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nStarting training at: {overall_start_datetime}")
    
    log_dir=config.get_directory(config.log_dir)
    logger = CSVLogger(log_dir, fieldnames=config.CSV_COLUMNS)

    output_path = config.get_directory(config.base_output_dir)
    run_name = f"{os.path.basename(output_path)}"
    
    print(f"Run: {run_name}")
    print(f"Created logging directory: {log_dir}")
    print(f"Models stored in: {output_path}\n")

    # Load tokenizer and models
    tokenizer = AutoTokenizer.from_pretrained(config.student_model_name)
    teacher_model = AutoModelForCausalLM.from_pretrained(config.teacher_model_name, torch_dtype=torch.bfloat16, device_map=device)
    teacher_model.requires_grad_(False)

    # Load dataset and setup data collator
    dataset = datasets.load_from_disk(config.dataset_path)
    response_template_ids = tokenizer("<|im_start|>assistant\n")["input_ids"]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    teacher_eval_results = evaluate_model(teacher_model, dataset["test"], collator, round_num=0, end=True)
    logger.log(
        function="main",
        round_num=0,
        phase="custom_eval",
        role="teacher",
        eval_loss=teacher_eval_results["eval_loss"],
        perplexity=teacher_eval_results["perplexity"],
    )
    
    # TODO: how is my eval statement different from the trainer and how do I align them

    existing_models = []
    for run_dir in config.past_run_dirs:
        for i in range(config.total_rounds):
            round_dir = os.path.join(run_dir, f"round_{i}")
            model_file = os.path.join(round_dir, "config.json")
            if os.path.exists(model_file):
                existing_models.append((i, round_dir))

    # Sort by round index
    existing_models.sort(key=lambda x: x[0])

    # Load ensemble model
    start_round = max((r for r, _ in existing_models), default=-1) + 1
    ensemble_model_names = [path for _, path in existing_models]
    ensemble_model = None

    if ensemble_model_names:
        print(f"Resuming from ensemble with {len(ensemble_model_names)} models")
        temp_model = AutoModelForCausalLM.from_pretrained(
            config.student_model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        ensemble_model = ModelEnsemble(
            model_names=ensemble_model_names,
            torch_dtype=torch.bfloat16,
            device_map=device,
            vocab_size=temp_model.vocab_size,
        )
        ensemble_model.requires_grad_(False)
        del temp_model

    for round_num in range(start_round, config.total_rounds):
        round_start_time = time.time()
        round_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n{'='*50}")
        print(f"Starting Round {round_num} at: {round_start_datetime}")
        print(f"{'='*50}")

        dataset["train"] = dataset["train"].shuffle(seed=config.seed + round_num)
        round_output_dir = get_round_path(output_path, round_num)
        print(f"Round '{round_num}' model stored in: {round_output_dir}")
        
        student_model = AutoModelForCausalLM.from_pretrained(config.student_model_name, torch_dtype=torch.bfloat16, device_map=device)
        student_eval_results = evaluate_model(student_model, dataset["test"], collator, round_num, eval_batch_size, end=True)
        logger.log(
            function="main",
            round_num=round_num,
            phase="custom_eval",
            role="student",
            eval_loss=student_eval_results["eval_loss"],
            perplexity=student_eval_results["perplexity"],
            tags=['initial eval'],
        )
        
        n = 0
        training_args = config.get_training_args(round_output_dir)
        trainer = DistillationTrainer(
            round_num=round_num,
            steps_per_round=config.steps_per_round,
            model=student_model,
            logger=logger,
            overall_start_time=overall_start_time,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=collator,
            args=training_args,
            callbacks=[LoggingCallback(logger, round_num, overall_start_time)],
        )
        
        # TODO: Why not use the trainer.predict method instead of the whole evaluation function
        # TODO: refactor code

        trainer.train()
        trainer.model.save_pretrained(round_output_dir)
        
        # Add model to the ensemble
        if ensemble_model is None:
            ensemble_model = ModelEnsemble(
                [round_output_dir],
                torch_dtype=torch.bfloat16,
                device_map=device,
                vocab_size=student_model.vocab_size,
            )
            ensemble_model.requires_grad_(False)
        else:
            ensemble_model.add_model(round_output_dir)

        # Evaluate
        student_eval_results = evaluate_model(trainer.model, dataset["test"], collator, round_num, eval_batch_size, end=True)
        ensemble_eval_results = evaluate_model(ensemble_model, dataset["test"], collator, round_num, eval_batch_size, end=True)

        print(f"\n{'-'*25}")
        print(f"Student evaluation for {round_num}: {student_eval_results['eval_loss']}")
        print(f"Ensemble evaluation for {round_num}: {ensemble_eval_results['eval_loss']}")
        print(f"Teacher evaluation for {round_num}: {teacher_eval_results['eval_loss']}")
        print(f"{'-'*25}")

        # After training, record round end time
        round_end_time = time.time()
        round_duration = round_end_time - round_start_time
        overall_elapsed = round_end_time - overall_start_time
        round_duration_str = format_time_elapsed(round_duration)
        overall_elapsed_str = format_time_elapsed(overall_elapsed)
        round_end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{'='*50}")
        print(f"Ending Round {round_num} at: {round_end_datetime}")
        print(f"Completed in: {round_duration_str}")
        print(f"Total training time: {overall_elapsed_str}")
        print(f"{'='*50}\n")

        # End of round logging
        logger.log(
            function="main",
            round_num=round_num,
            phase="custom_eval",
            role="ensemble",
            eval_loss=ensemble_eval_results["eval_loss"],
            perplexity=ensemble_eval_results["perplexity"],
            round_duration=round_duration,
        )
        logger.log(
            function="main",
            round_num=round_num,
            phase="custom_eval",
            role="student",
            eval_loss=student_eval_results["eval_loss"],
            perplexity=student_eval_results["perplexity"],
            round_duration=round_duration,
        )
        logger.log(
            function="main",
            round_num=round_num,
            phase="custom_eval",
            role="teacher",
            eval_loss=teacher_eval_results["eval_loss"],
            perplexity=teacher_eval_results["perplexity"],
            round_duration=round_duration,
        )
        
        del student_model
        gc.collect()
        torch.cuda.empty_cache()

    # Record overall end time
    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    overall_duration_str = format_time_elapsed(overall_duration)
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*50}")
    print(f"Training completed at: {end_datetime}")
    print(f"Total training time: {overall_duration_str}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
