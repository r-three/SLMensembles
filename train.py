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

# TODO: add langauge modeling loss logging during the training loop

def format_time_elapsed(seconds):
    """Convert seconds to a readable format with minutes and seconds."""
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {int(seconds)}s"


def get_round_path(output_path, round_num):
    """Return path for a specific training round."""
    return os.path.join(output_path, f"round_{round_num}")


def evaluate_model(model, eval_dataset, batch_size, collator):
    """Evaluates any model on a dataset using existing collator."""
    model.eval()
    
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collator)
    
    total_loss = 0
    total_tokens = 0
    
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
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return {
        "eval_loss": avg_loss,
        "perplexity": perplexity,
        "num_tokens": total_tokens
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


# class WandbCallback(TrainerCallback):
#     def on_evaluate(self, args, state, control, metrics=None, **kwargs):
#         if metrics is not None:
#             wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=state.global_step)


def main():
    global teacher_model, ensemble_model

    overall_start_time = time.time()
    overall_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Starting training at: {overall_start_datetime}")

    output_path = config.get_run_directory()
    print(f"Models stored in: {output_path}")

    # Setup wandb
    # run_name = f"{os.path.basename(output_path)}"
    # wandb.init(project="slm_ensembles", name=run_name)
    # wandb.config.update(
    #     {
    #         "student_model": config.student_model_name,
    #         "teacher_model": config.teacher_model_name,
    #         "total_rounds": config.total_rounds,
    #         "steps_per_round": config.steps_per_round,
    #         "training_start_time": overall_start_datetime,
    #     }
    # )
    
    # TODO: an initial evaluation of the teacher and the student model
        # 2) similarity
        # 1) performance on the dataset

    # Load tokenizer and models
    tokenizer = AutoTokenizer.from_pretrained(config.student_model_name)
    student_model = AutoModelForCausalLM.from_pretrained(
        config.student_model_name, torch_dtype=torch.bfloat16, device_map=device)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        config.teacher_model_name, torch_dtype=torch.bfloat16, device_map=device)
    teacher_model.requires_grad_(False)
    
    # Load dataset and setup data collator
    dataset = datasets.load_from_disk(config.dataset_path)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    all_student_results = {}

    for round_num in range(0, config.total_rounds):
        round_start_time = time.time()
        round_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n{'='*50}")
        print(f"Starting Round {round_num} at: {round_start_datetime}")
        print(f"{'='*50}")

        dataset["train"] = dataset["train"].shuffle(seed=config.seed + round_num)
        round_output_dir = get_round_path(output_path, round_num)
        print(f"Round '{round_num}' model stored in: {round_output_dir}")

        # # Callback to log under the round-specific namespace
        # class RoundSpecificCallback(TrainerCallback):
        #     def on_log(self, args, state, control, logs=None, **kwargs):
        #         if logs:
        #             if "loss" in logs and "eval_loss" not in logs:  # save only the training loss
        #                 round_logs = {f"round_{round_num}/train/{k}": v for k, v in logs.items()}
        #                 # Include the round number so we can plot by round
        #                 round_logs["round"] = round_num
        #                 wandb.log(round_logs, step=state.global_step)

        training_args = config.get_training_args(round_output_dir)

        trainer = DistillationTrainer(
            round_num=round_num,
            steps_per_round=config.steps_per_round,
            model=student_model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=collator,
            args=training_args,
            # callbacks=[RoundSpecificCallback],
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
        eval_batch_size = config.eval_batch_size
        student_eval_results = evaluate_model(trainer.model, dataset["test"], eval_batch_size, collator)
        ensemble_eval_results = evaluate_model(ensemble_model, dataset["test"], eval_batch_size, collator)
        teacher_eval_results = evaluate_model(teacher_model, dataset["test"], eval_batch_size, collator)
    
        print(f"\n{'-'*25}")
        print(f"Student evaluation for {round_num}: {student_eval_results["eval_loss"]}, perplexity: {student_eval_results["perplexity"]}, tokens: {student_eval_results["num_tokens"]}")
        print(f"Ensemble evaluation for {round_num}: {ensemble_eval_results["eval_loss"]}, perplexity: {ensemble_eval_results["perplexity"]}, tokens: {ensemble_eval_results["num_tokens"]}")
        print(f"Teacher evaluation for {round_num}: {teacher_eval_results["eval_loss"]}, perplexity: {teacher_eval_results["perplexity"]}, tokens: {teacher_eval_results["num_tokens"]}")
        print(f"{'-'*25}")

        # Log all metrics in a consistent structure
        # metrics = {
        #     "round": round_num,  # Round number for X-axis
        #     "student/eval_loss": student_eval_results["eval_loss"],  # Student eval metrics (current round)
        #     "ensemble/eval_loss": ensemble_eval_results["eval_loss"],  # Ensemble metrics
        #     "ensemble/size": len(ensemble_model.models),
        #     # New metrics:
        #     "performance/teacher_vs_ensemble_gap": teacher_eval_results["eval_loss"]
        #     - ensemble_eval_results["eval_loss"],
        #     "performance/ensemble_improvement": (
        #         previous_ensemble_loss - ensemble_eval_results["eval_loss"] if round_num > 0 else 0
        #     ),
        #     "performance/teacher_achievement_pct": (
        #         teacher_eval_results["eval_loss"] / ensemble_eval_results["eval_loss"]
        #     )
        #     * 100,
        # }
        # wandb.log(metrics)

        # Track previous ensemble loss for improvement calculation
        # previous_ensemble_loss = ensemble_eval_results["eval_loss"]

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

        # Log timing information to wandb
        # timing_metrics = {
        #     "time/round_duration_seconds": round_duration,
        #     "time/round_duration_minutes": round_duration / 60.0,
        #     "time/total_elapsed_seconds": overall_elapsed,
        #     "time/total_elapsed_minutes": overall_elapsed / 60.0,
        #     "time/round": round_num,
        # }
        # wandb.log(timing_metrics)

        # Reset the student model for the next round and load a fresh copy
        del student_model
        student_model = AutoModelForCausalLM.from_pretrained(
            config.student_model_name, torch_dtype=torch.bfloat16, device_map=device
        )

    # Log final metrics
    # student_table = wandb.Table(columns=["Round", "Eval Loss"])
    # for round_num, results in all_student_results.items():
    #     student_table.add_data(round_num, results["eval_loss"])
    # wandb.log({"student_performance_table": student_table})

    # Record overall end time
    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    overall_duration_str = format_time_elapsed(overall_duration)
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*50}")
    print(f"Training completed at: {end_datetime}")
    print(f"Total training time: {overall_duration_str}")
    print(f"{'='*50}")

    # Log final timing information
    # final_timing = {
    #     "time/total_training_seconds": overall_duration,
    #     "time/total_training_minutes": overall_duration / 60.0,
    #     "time/average_round_minutes": (overall_duration / 60.0) / (config.total_rounds - start_round),
    #     "time/training_end_time": end_datetime,
    # }
    # wandb.log(final_timing)

    # Create timing summary table
    # timing_table = wandb.Table(columns=["Round", "Duration (min)", "Cumulative (min)"])
    # total_mins = 0
    # for r in range(start_round, config.total_rounds):
        # Get the round duration from our logs if available
        # round_duration_min = wandb.run.summary.get(f"time/round_duration_minutes_{r}", 0)
        # total_mins += round_duration_min
        # timing_table.add_data(r, round_duration_min, total_mins)

    # wandb.log({"time/summary_table": timing_table})

    # Close wandb
    # wandb.finish()


if __name__ == "__main__":
    main()
