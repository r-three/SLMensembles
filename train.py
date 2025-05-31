import gc
import os
import time
import wandb

import numpy as np
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
        self.run = kwargs.pop("run")
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

        current_model_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        # Distill teacher into current model
        loss = self.compute_kl_loss(current_model_logits, ensemble_logits, teacher_logits, labels != -100)
        self.run.log({f"round_{self.round_num}/train/kl_loss": loss})
        
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
        """Model eval"""
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        labels = inputs["labels"].to(device)

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

        loss = model.loss_function(
            logits=student_logits,
            labels=labels,
            vocab_size=model.config.vocab_size,
        )

        kl_loss = self.compute_kl_loss(student_logits, ensemble_logits, teacher_logits, labels != -100)
        if "kl_losses" not in self.extra_logging_info:
            self.extra_logging_info["kl_losses"] = []
        self.extra_logging_info["kl_losses"].append(kl_loss.item())

        return (loss, None, None) if prediction_loss_only else (loss, student_logits, labels)
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
        self.run.log({
            f"round_{self.round_num}/eval/lm_loss": output.metrics["eval_loss"],
            f"round_{self.round_num}/eval/kl_loss": np.mean(self.extra_logging_info["kl_losses"])
        })
        return output

    def log(self, logs, start_time=None):
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            if start_time is not None:
                speed_metrics("train", start_time, num_tokens=self.state.num_input_tokens_seen)

        output = {**logs}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
        

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
    run = wandb.init(project="<slm_ensembles>", name=f"test_{run_name}")
    global_step = 0

    # Load tokenizer and models
    tokenizer = AutoTokenizer.from_pretrained(config.student_model_name)
    teacher_model = AutoModelForCausalLM.from_pretrained(config.teacher_model_name, torch_dtype=torch.bfloat16, device_map=device)
    teacher_model.requires_grad_(False)

    # Load dataset and setup data collator
    dataset = datasets.load_from_disk(config.dataset_path)
    response_template_ids = tokenizer("<|im_start|>assistant\n")["input_ids"]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    # collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    teacher_eval_results = evaluate_model(teacher_model, dataset["test"], collator, 0, True)
    wandb.log({"eval/teacher_loss": teacher_eval_results["eval_loss"]}, step=global_step)
    
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

        run.watch(student_model)

        training_args = config.get_training_args(round_output_dir)
        trainer = DistillationTrainer(
            round_num=round_num,
            steps_per_round=config.steps_per_round,
            run=run,
            model=student_model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=collator,
            args=training_args,
        )

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
        round_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{'='*50}")
        print(f"Ending Round {round_num} at: {round_start_datetime}")
        print(f"Completed in: {round_duration_str}")
        print(f"Total training time: {overall_elapsed_str}")
        print(f"{'='*50}\n")

        run.log({"eval/custom_student_loss": student_eval_results["eval_loss"]}, step=global_step)
        global_step += trainer.state.global_step
        # global_step += config.steps_per_round

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
