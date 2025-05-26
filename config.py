import os
import torch
from datetime import datetime
import glob

# WandB Setup
os.environ["WANDB_PROJECT"] = "<slm_ensembles>"
os.environ["WANDB_LOG_MODEL"] = "end"

# Model and dataset setup
seed = 42
device = "cuda" if torch.cuda.is_available() else "cpu"
teacher_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
student_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer_name = "Qwen/Qwen2.5-0.5B-Instruct"
dataset_name = "allenai/tulu-3-sft-mixture"
ensemble_model_names = []

dataset_path = "/scratch/ssd004/scratch/klambert/slm_ensembles/tulu-3-sft-mixture-pretokenized"
base_output_dir = "/scratch/ssd004/scratch/klambert/slm_ensembles/boosted_distillation_1.5B_teacher_average_fixed_logging"

# Training parameters
total_rounds = 10  # number of ensemble models
steps_per_round = 1000
kl_temperature = 1.0
eval_batch_size = 8


def get_run_directory():
    # Get current date in YYYY-MM-DD format
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Create a date-specific directory path
    date_dir = os.path.join(base_output_dir, current_date)

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
        report_to="none",
        hub_model_id=None,
        learning_rate=1e-6,
        # lr_scheduler_type="constant",
        warmup_steps=50,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=8,
        gradient_checkpointing=False,
        bf16=True,
        max_steps=steps_per_round,
        eval_strategy="steps",
        eval_steps=int(steps_per_round / 10),
        logging_strategy="steps",
        logging_steps=10,
        # save_strategy="true",
        # save_steps=500,
        # save_total_limit=3,
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        # greater_is_better=False,
    )
