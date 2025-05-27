import gc
import os
import time
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets
from torch.utils.data import DataLoader
from datetime import datetime, timedelta
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_utils import speed_metrics
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from ensemble import ModelEnsemble
import config

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


def evaluate_model(model, eval_dataset, collator, round_num, max_eval_samples=None, end=False):
    if end == True:
        model.eval()
    if max_eval_samples:
        eval_dataset = torch.utils.data.Subset(eval_dataset, range(max_eval_samples))
    
    eval_dataloader = DataLoader(eval_dataset, eval_batch_size, collate_fn=collator)

    total_loss = 0
    total_tokens = 0
    total_examples = 0

    start_time = time.time()

    # TODO: add KL div to teacher and to ensemble evaluation metric

    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            # Count non-masked tokens for proper averaging
            valid_tokens = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * valid_tokens
            total_tokens += valid_tokens
            total_examples += input_ids.size(0)

    runtime = time.time() - start_time
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    num_steps = len(eval_dataloader)
    steps_per_sec = num_steps / runtime
    tokens_per_sec = total_tokens / runtime
    samples_per_sec = total_examples / runtime

    return {
        "eval_loss": avg_loss,
        "perplexity": perplexity,
        "num_tokens": total_tokens,
        "eval_runtime": runtime,
        "eval_tokens_per_second": tokens_per_sec,
        "eval_samples_per_second": samples_per_sec,
        "eval_steps_per_second": steps_per_sec,
    }


class DistillationTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.round_num = kwargs.pop("round_num")
        self.steps_per_round = kwargs.pop("steps_per_round")
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

        current_model_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        # Distill teacher into current model
        loss = self.compute_kl_loss(current_model_logits, ensemble_logits, teacher_logits, labels != -100)
        return (loss, current_model_logits) if return_outputs else loss

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
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = inputs["labels"].to(device)

        with torch.no_grad():
            student_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

            if ensemble_model is not None:
                num_models = len(ensemble_model.models)
                ensemble_logits = ensemble_model(input_ids=input_ids, attention_mask=attention_mask).logits.detach()
                student_logits = student_logits / (num_models + 1) + ensemble_logits * (num_models / (num_models + 1))

        # Handle potential model wrapping (DataParallel/DistributedDataParallel)
        if hasattr(model, "module"):
            model = model.module

        loss = model.loss_function(
            logits=student_logits,
            labels=labels,
            vocab_size=model.config.vocab_size,
        )
        return (loss, None, None) if prediction_loss_only else (loss, student_logits, labels)

    def log(self, logs, start_time=None):
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            if start_time is not None:
                speed_metrics("train", start_time, num_tokens=self.state.num_input_tokens_seen)

        output = {
            **logs,
            **{"step": self.state.global_step + (self.round_num * self.steps_per_round)},
        }
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)


class WandbEvalsCallback(TrainerCallback):
    """Custom WandbCallback to log model predictions during training."""
    def __init__(self, round_num, steps_per_round, teacher_eval_results, ensemble_eval_results, eval_dataset, collator):
        self.round_num = round_num
        self.steps_per_round = steps_per_round
        self.teacher_eval_results = teacher_eval_results
        self.ensemble_eval_results = ensemble_eval_results
        self.eval_dataset = eval_dataset
        self.collator = collator

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation."""

        # print(f"\n\nOriginal Eval Metrics Keys (Round {self.round_num}, Step {state.global_step}): {list(metrics.keys())}\n\n")
        adjusted_step = state.global_step + (self.round_num * args.max_steps)
        current_model = kwargs["model"]
        student_eval_results = evaluate_model(current_model, self.eval_dataset, self.collator, self.round_num, max_eval_samples=200)

        # custom_logs = {f"on_evaluate_round_{self.round_num}/eval/{k}": v for k, v in metrics.items()}

        combined_eval_logs = {}
        # combined_eval_logs["round"] = self.round_num
        
        for k, v in student_eval_results.items():
            combined_eval_logs[f"on_eval_round_{self.round_num}/eval_student/{k}"] = v
        for k, v in self.teacher_eval_results.items():
            combined_eval_logs[f"on_eval_round_{self.round_num}/eval_teacher/{k}"] = v
        if self.ensemble_eval_results is not None:
            for k, v in self.ensemble_eval_results.items():
                combined_eval_logs[f"on_eval_round_{self.round_num}/eval_ensemble/{k}"] = v

        wandb.log(combined_eval_logs, step=adjusted_step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log training loss, learning rate, and gradient norm during training."""
        if logs is None:
            return

        adjusted_step = state.global_step + (self.round_num * self.steps_per_round)

        custom_logs = {}
        if "loss" in logs:
            custom_logs[f"on_log_round_{self.round_num}/train/loss"] = logs["loss"]
        if "learning_rate" in logs:
            custom_logs[f"on_log_round_{self.round_num}/train/learning_rate"] = logs["learning_rate"]
        if "grad_norm" in logs:
            custom_logs[f"on_log_round_{self.round_num}/train/grad_norm"] = logs["grad_norm"]

        # Optional: include raw logs for debugging
        # for k, v in logs.items():
        #     custom_logs[f"on_log_round_{self.round_num}/train/{k}"] = v

        wandb.log(custom_logs, step=adjusted_step)

        
    def on_step_end(self, args, state, control, **kwargs):
        """Log training metrics like loss, lr, and gradient norm during training."""

        logs = {}
        trainer = kwargs.get("trainer", None)

        # Step-aligned logging
        adjusted_step = state.global_step + (self.round_num * self.steps_per_round)

        # 1. Learning Rate
        if trainer is not None and hasattr(trainer, "optimizer"):
            lr = trainer.optimizer.param_groups[0]["lr"]
            logs[f"on_step_round_{self.round_num}/train/learning_rate"] = lr

        # 2. Loss (might need to extract manually if not in state)
        if hasattr(state, "log_history") and state.log_history:
            recent = state.log_history[-1]
            if "loss" in recent:
                logs[f"on_step_round_{self.round_num}/train/loss"] = recent["loss"]

        # 3. Gradient Norm
        if trainer is not None and hasattr(trainer.model, "parameters"):
            total_norm = 0.0
            for p in trainer.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            logs[f"on_step_round_{self.round_num}/train/grad_norm"] = total_norm

        wandb.log(logs, step=adjusted_step)



def main():
    global teacher_model, ensemble_model

    overall_start_time = time.time()
    overall_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nStarting training at: {overall_start_datetime}")

    output_path = config.get_run_directory()
    run_name = f"{os.path.basename(output_path)}"
    print(f"Run: {run_name}")

    print(f"Models stored in: {output_path}\n")

    # Setup wandb
    wandb.init(project="<slm_ensembles>", name=run_name)

    # Load tokenizer and models
    tokenizer = AutoTokenizer.from_pretrained(config.student_model_name)
    teacher_model = AutoModelForCausalLM.from_pretrained(config.teacher_model_name, torch_dtype=torch.bfloat16, device_map=device)
    teacher_model.requires_grad_(False)

    # Load dataset and setup data collator
    dataset = datasets.load_from_disk(config.dataset_path)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    teacher_eval_results = evaluate_model(teacher_model, dataset["test"], collator, 0, True)
    ensemble_eval_results = None

    for round_num in range(0, config.total_rounds):
        round_start_time = time.time()
        round_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n{'='*50}")
        print(f"Starting Round {round_num} at: {round_start_datetime}")
        print(f"{'='*50}")

        dataset["train"] = dataset["train"].shuffle(seed=config.seed + round_num)
        round_output_dir = get_round_path(output_path, round_num)
        print(f"Round '{round_num}' model stored in: {round_output_dir}")

        student_model = AutoModelForCausalLM.from_pretrained(config.student_model_name, torch_dtype=torch.bfloat16, device_map=device)

        # Disable sliding window
        if hasattr(student_model.config, "use_sliding_window"):
            student_model.config.use_sliding_window = False

        training_args = config.get_training_args(round_output_dir)

        trainer = DistillationTrainer(
            round_num=round_num,
            steps_per_round=config.steps_per_round,
            model=student_model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=collator,
            args=training_args,
            callbacks=[WandbEvalsCallback(round_num, config.steps_per_round, teacher_eval_results, ensemble_eval_results, dataset["train"], collator)],
        )

        trainer.train()
        trainer.model.save_pretrained(round_output_dir)
        log_dict = {"round": round_num}
        wandb.log(log_dict, step=(round_num + 1) * config.steps_per_round)

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
        round_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{'='*50}")
        print(f"Ending Round {round_num} at: {round_start_datetime}")
        print(f"Completed in: {round_duration_str}")
        print(f"Total training time: {overall_elapsed_str}")
        print(f"{'='*50}\n")

        # Reset the student model for the next round and load a fresh copy
        del student_model
        gc.collect()
        torch.cuda.empty_cache()
        # student_model = AutoModelForCausalLM.from_pretrained(
        #     config.student_model_name, torch_dtype=torch.bfloat16, device_map=device
        # )

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
