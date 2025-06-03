import os
import torch
from datetime import datetime
import glob

# Model and dataset setup
seed = 42
device = "cuda" if torch.cuda.is_available() else "cpu"
teacher_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
student_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer_name = "Qwen/Qwen2.5-0.5B-Instruct"
dataset_name = "allenai/tulu-3-sft-mixture"
ensemble_model_names = []

dataset_path = "/scratch/ssd004/scratch/klambert/slm_ensembles/tulu-3-sft-mixture-pretokenized"
base_output_dir = "/projects/distilling_llms/model_log"
log_dir = "/scratch/ssd004/scratch/klambert/slm_ensembles/csv_logs"

# Training parameters
total_rounds = 6  # number of ensemble models
steps_per_round = 1000
kl_temperature = 1.0
eval_batch_size = 4
alpha = 0.5

# Logging Arguments
CSV_COLUMNS = [
    "function",             # function name where the logging occured
    "timestamp",            # e.g. "2025-06-01 12:34:56"
    "overall_elapsed",      # float (seconds; from start-of-all-rounds)
    "round_duration",       # float
    "ensemble_size",        # int (how many models are in the ensemble so far)
    "round",                # distillation round (0, 1, 2, …)
    "phase",                # e.g. "train", "eval"
    "role",                 # e.g. "student", "teacher", "ensemble"
    "step",                 # training step within the round
    "train_loss",           # next token prediction loss during training
    "kl_loss",              # KL divergence loss
    "eval_loss",            # float (or None)
    "perplexity",           # float (only on eval rows)
    "learning_rate",
]


def get_directory(output_dir):
    # Get current date in YYYY-MM-DD format
    current_date = datetime.now().strftime("%d-%m-%Y")

    # Create a date-specific directory path
    date_dir = os.path.join(output_dir, current_date)
    
    # return before making a separate run directory
    if output_dir == log_dir:
        return date_dir

    # Find existing run directories for today
    existing_runs = []
    if os.path.exists(date_dir):
        # Get all subdirectories with pattern "run_X"
        run_dirs = glob.glob(os.path.join(date_dir, "run_*"))

        # Extract run numbers
        for dir_path in run_dirs:
            try:
                run_num = int(os.path.basename(dir_path).split("_")[1])
                existing_runs.append(run_num)
            except (ValueError, IndexError):
                continue

    next_run = 1
    if existing_runs:
        next_run = max(existing_runs) + 1

    run_dir = os.path.join(date_dir, f"run_{next_run}")
    os.makedirs(run_dir, exist_ok=True)

    return run_dir


def get_training_args(output_dir):
    from trl import SFTConfig

    return SFTConfig(
        output_dir=output_dir,
        overwrite_output_dir=False,
        report_to="wandb",
        hub_model_id=None,
        learning_rate=5e-5,
        warmup_steps=50,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=8,
        gradient_checkpointing=False,
        bf16=True,
        max_steps=steps_per_round,
        eval_strategy="steps",
        eval_steps=int(steps_per_round / 10),
        eval_on_start=True,
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="no",
    )
