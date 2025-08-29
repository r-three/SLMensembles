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
import glob

def main():
    print("Loading ...")
    dataset = datasets.load_from_disk(config.dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_name)
    response_template_ids = tokenizer("<|im_start|>assistant\n")["input_ids"]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    # Setup output directories with checkpoint structure
    teacher_output_dir = os.path.join(config.base_output_dir, "Qwen-7B-fine-tuned")
    checkpoint_dir = os.path.join(teacher_output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize checkpointer
    checkpointer = Checkpointer(teacher_output_dir)
    
    # Load model (either from checkpoint or pretrained)
    start_step = 0
    start_epoch = 0
    
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            config.teacher_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        # Load checkpoint state using HuggingFace format for teacher model
        try:
            if os.path.isdir(args.resume_from_checkpoint):
                # HuggingFace checkpoint format
                print("Loading from HuggingFace checkpoint format")
                teacher_model = AutoModelForCausalLM.from_pretrained(
                    args.resume_from_checkpoint,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
            else:
                print("Invalid checkpoint path")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting from pretrained model instead")
    else:
        print("Starting from pretrained model")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            config.teacher_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

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
    resume_from_checkpoint = args.resume_from_checkpoint if args.resume_from_checkpoint else None

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        overwrite_output_dir=False,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        report_to="wandb",
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler,
        weight_decay=config.weight_decay,
        hub_model_id=None,
        warmup_steps=config.warmup_steps,
        gradient_checkpointing=True,
        bf16=True,
        remove_unused_columns=False,
        max_steps=config.steps_per_round,
        num_train_epochs=config.num_train_epochs,
        eval_on_start=False,
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        # Enable periodic checkpointing
        save_strategy="steps",
        save_steps=getattr(config, 'ckpt_save_steps', 500),
        save_total_limit=3,  # Keep only 3 most recent checkpoints
        resume_from_checkpoint=resume_from_checkpoint,
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
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        print("Done training")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Saving checkpoint before exit...")
        trainer.save_model(os.path.join(teacher_output_dir, "interrupted_checkpoint"))
        raise
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        print("Saving checkpoint before exit...")
        trainer.save_model(os.path.join(teacher_output_dir, "error_checkpoint"))
        raise

    # Save final model
    final_model_path = os.path.join(teacher_output_dir, "final_model")
    teacher_model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    print(f"Final teacher model saved to: {final_model_path}")
    
    # Also save in the old format for compatibility
    teacher_model.save_pretrained(teacher_output_dir)

    if wandb_run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()
    