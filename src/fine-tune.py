import os
import torch
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from trl import DataCollatorForCompletionOnlyLM
from utils import main_print
import datasets
import wandb
import config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_from_checkpoint', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, help='Path to checkpoint directory')
    args = parser.parse_args()

    dataset = datasets.load_from_disk(config.dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_name)
    response_template_ids = tokenizer("<|im_start|>assistant\n")["input_ids"]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    teacher_output_dir = os.path.join(config.base_output_dir, "Qwen-7B-fine-tuned")
    checkpoint_dir = os.path.join(teacher_output_dir, "checkpoints")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    start_step = 0
    start_epoch = 0
    
    if args.resume_from_checkpoint:        
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
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
    resume_from_checkpoint = args.checkpoint_dir if args.resume_from_checkpoint and args.checkpoint_dir else False

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
        save_strategy="steps",
        save_steps=config.ckpt_save_steps,
        save_total_limit=3,
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
        trainer.save_model(os.path.join(checkpoint_dir, "interrupted_checkpoint"))
        raise
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        print("Saving checkpoint before exit...")
        trainer.save_model(os.path.join(checkpoint_dir, "error_checkpoint"))
        raise

    teacher_model.save_pretrained(os.path.join(teacher_output_dir, "final_model"))
    teacher_model.save_pretrained(teacher_output_dir)
    tokenizer.save_pretrained(teacher_output_dir)
    
    print(f"Final teacher model saved to: {teacher_output_dir}")

    if wandb_run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()
    