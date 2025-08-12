import os
import torch
from datetime import datetime
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from trl import DataCollatorForCompletionOnlyLM
from utils import (CSVLogger, prepare_dataset, format_time_elapsed, 
                  is_main_process, main_print, check_batch_shape, fix_seed,
                  inspect_mixed_precision, inspect_model,
                  set_modules_to_forward_prefetch, set_modules_to_backward_prefetch)
from ensemble import ModelEnsemble
from checkpoint import Checkpointer, index_checkpoints, best_checkpoint
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from tqdm.auto import tqdm
from shard_weight import *
from utils import fix_seed
import atexit
from pathlib import Path
from datasets import Dataset, DatasetDict
from utils import DistillDataset, get_round_path
from checkpoint import Checkpoint
from transformers import TrainingArguments, Trainer
import datasets
import wandb
import config

def main():
    print("Loading ...")
    dataset = datasets.load_from_disk(config.dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_name)
    response_template_ids = tokenizer("<|im_start|>assistant\n")["input_ids"]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    teacher_model = AutoModelForCausalLM.from_pretrained(
        config.teacher_model_name,
        torch_dtype=torch.bfloat16,
    ).to('cuda')

    wandb_run = wandb.init(
        project="slm-ensembles",
        name=config.run_name,
        config={
            "model_name": config.teacher_model_name,
            "learning_rate": config.learning_rate,
            "batch_size": config.per_device_train_batch_size,
            "seed": config.seed,
            "description": config.description,
            "dataset_name": config.dataset_name,
            "dataset_type": config.dataset_type,
            "total_rounds": config.total_rounds,
            "num_train_epochs": config.num_train_epochs,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "max_grad_norm": getattr(config, 'max_grad_norm', 1.0),
        },
        tags=["fine-tuning"],
        resume="allow",
    )

    main_print(f"--> Initialized wandb run: {wandb_run.name}")

    print("Initializing trainer...")
    training_args = TrainingArguments(
        output_dir=config.get_directory(config.base_output_dir),
        overwrite_output_dir=False,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        report_to="wandb",
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler,
        weight_decay=config.weight_decay,
        hub_model_id=None,
        warmup_steps=config.warmup_steps,
        gradient_checkpointing=False,
        bf16=True,
        remove_unused_columns=False,
        max_steps=config.steps_per_round,
        num_train_epochs=config.num_train_epochs,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        eval_on_start=False,
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        save_strategy="no",
    )

    trainer = Trainer(
        model=teacher_model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=tokenizer,
        data_collator=collator,
    )
    print("Training...")
    trainer.train()
    print("Done training")

    teacher_model.save_pretrained(os.path.join(config.get_directory(config.base_output_dir), "fine-tuned-teacher"))

    if wandb_run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()
    