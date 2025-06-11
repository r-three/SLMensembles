import os
import datasets
from datetime import datetime
import glob

# TODO before each run:
# change alpha value
# change custom_path
# clean up and safely store csv files
# remove empty directories in past_run_dirs
# uncomment and comment out past_run_dirs
# launch the sbatch script with custom names

# Model and dataset setup
seed = 42
teacher_model_name = "Qwen/Qwen2.5-7B-Instruct"
student_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer_name = "Qwen/Qwen2.5-0.5B-Instruct"
dataset_name = "allenai/tulu-3-sft-mixture"
dataset_type = "batch"  # "single" or "batch" or "full"
teacher_device = "cuda:0"
student_device = "cuda:1"

dataset_path = (
    "/scratch/ssd004/scratch/klambert/slm_ensembles/tulu-3-sft-mixture-pretokenized"
)
base_output_dir = "/projects/distilling_llms/model_log"
log_dir = "/scratch/ssd004/scratch/klambert/slm_ensembles/csv_logs"
custom_path = "batch"

ensemble_model_names = []
past_run_dirs = []

# Training parameters
alpha = 1  # 1 = next_token loss to 0 = kl_loss
total_rounds = 16  # number of ensemble models
steps_per_round = 1000
kl_temperature = 1.0
eval_batch_size = 4

# Logging Arguments
CSV_COLUMNS = [
    "function",
    "timestamp",
    "overall_elapsed",
    "round_duration",
    "round_num",
    "ensemble_num",
    "phase",
    "role",
    "step",
    "train_loss",
    "train_kl_loss",
    "train_next_token_loss",
    "eval_loss",
    "eval_kl_loss",
    "grad_norm",
    "perplexity",
    "learning_rate",
    "alpha",
    "tags",
    "metadata",
]


def get_dataset():
    dataset = datasets.load_from_disk(dataset_path)
    if dataset_type == "single":
        return {
            "train": dataset["train"].select([0]),
            "test": dataset["test"].select([0]),
        }
    elif dataset_type == "batch":
        return {
            "train": dataset["train"].select(range(10)),
            "test": dataset["test"].select(range(10)),
        }
    return dataset


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

    if custom_path is not None:
        run_dir = os.path.join(date_dir, f"run_{next_run}_{custom_path}")
    else:
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
        learning_rate=5e-5,
        warmup_steps=50,
        per_device_train_batch_size=1,  # 4
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=8,
        gradient_checkpointing=False,
        bf16=True,
        max_steps=steps_per_round,
        eval_strategy="steps",
        eval_steps=int(steps_per_round / 10),
        eval_on_start=True,
        logging_strategy="steps",
        logging_steps=40,
        save_strategy="no",
    )


# TODO:
# learning rate
# batch size
# steps
# check that the student can reach 0 loss (Eval) (single batch / example); add second student & keep the first student static (untrained) and check how the loss goes down

# From Qwen2.5 technical report: Ultimately, we construct a dataset of over 1 million SFT examples. The model is fine-tuned for two epochs
# with a sequence length of 32,768 tokens. To optimize learning, the learning rate is gradually decreased
# from 7 × 10−6 to 7 × 10−7. To address overfitting, we apply a weight decay of 0.1, and gradient norms are clipped at a maximum value of 1.0.

# It is for SFT but might be helpful in the distillation setting as well!
