import argparse
import os
import time
import torch
import torch.distributed as dist
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from tqdm.auto import tqdm
import sys

from simple_config import config
from simple_trainer import Trainer
from simple_utils import prepare_dataset, get_dataset, is_main_process, main_print, fix_seed
from simple_checkpoint import SimpleCheckpointer

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


def main(args):
    """
    Simplified single teacher-student distillation pipeline.
    """
    # ----------------------------------
    # DDP Setup
    # ----------------------------------
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    fix_seed(config.seed)

    overall_start_time = time.time()
    main_print(f"--> Starting training at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ----------------------------------
    # Output Directory
    # ----------------------------------
    output_path = config.output_dir
    os.makedirs(output_path, exist_ok=True)
    
    # ----------------------------------
    # Wandb Initialization
    # ----------------------------------
    use_wandb = WANDB_AVAILABLE
    if is_main_process():
        # Generate run name if not provided
        run_name = config.wandb_run_name
        if run_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            student_short = config.student_model_name.split('/')[-1]
            teacher_short = config.teacher_model_name.split('/')[-1]
            run_name = f"{student_short}_from_{teacher_short}_{timestamp}"
        
        # Initialize wandb
        wandb.init(
            project=config.wandb_project,
            name=run_name,
            config={
                "teacher_model": config.teacher_model_name,
                "student_model": config.student_model_name,
                "num_epochs": config.num_epochs,
                "batch_size": config.batch_size,
                "eval_batch_size": config.eval_batch_size,
                "learning_rate": config.learning_rate,
                "max_grad_norm": config.max_grad_norm,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "alpha": config.alpha,
                "temperature": config.temperature,
                "dataset": config.dataset_name,
                "max_seq_length": config.max_seq_length,
                "seed": config.seed,
                "mixed_precision": args.mixed_precision,
            },
        )
        main_print(f"Wandb initialized: {wandb.run.url}")
    else:
        use_wandb = False
    
    # ----------------------------------
    # Dataset Loading
    # ----------------------------------
    main_print("Loading dataset...")
    dataset = get_dataset()
    train_dataloader, eval_dataloader = prepare_dataset(
        dataset['train'],
        dataset['test'],
    )
    
    # ----------------------------------
    # Load Teacher Model (frozen)
    # ----------------------------------
    main_print("Loading teacher model...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        config.teacher_model_name,
        torch_dtype=torch.bfloat16,
    ).to('cuda')
    teacher_model.eval()
    
    # Shard teacher model for memory efficiency
    for layer in teacher_model.model.layers:
        fully_shard(layer)
    fully_shard(teacher_model)
    
    # ----------------------------------
    # Load Student Model
    # ----------------------------------
    main_print("Loading student model...")
    student_model = AutoModelForCausalLM.from_pretrained(
        config.student_model_name,
        torch_dtype=torch.bfloat16,
    ).to('cuda')
    
    # ----------------------------------
    # FSDP Setup for Student
    # ----------------------------------
    fsdp_kwargs = {}
    if args.mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
    
    for layer in student_model.model.layers:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(student_model, **fsdp_kwargs)
    
    # ----------------------------------
    # Optimizer and Scheduler
    # ----------------------------------
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=config.learning_rate)
    
    num_training_steps = len(train_dataloader) * config.num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # ----------------------------------
    # Checkpointer
    # ----------------------------------
    checkpointer = SimpleCheckpointer(output_path)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    if config.resume_from_checkpoint:
        checkpoint_data = checkpointer.load(student_model, optimizer, lr_scheduler)
        if checkpoint_data:
            start_epoch = checkpoint_data.get('epoch', 0) + 1
            global_step = checkpoint_data.get('global_step', 0)
            main_print(f"Resumed from epoch {start_epoch-1}, step {global_step}")
    
    # ----------------------------------
    # Initialize Trainer
    # ----------------------------------
    trainer = Trainer(
        student_model=student_model,
        teacher_model=teacher_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpointer=checkpointer,
    )
    
    # Sync trainer state with loaded checkpoint
    trainer.global_step = global_step
    trainer.epoch = start_epoch
    
    # ----------------------------------
    # Training Loop
    # ----------------------------------
    main_print("\n" + "="*50)
    main_print("Starting Training")
    main_print("="*50)
    
    for epoch in range(start_epoch, config.num_epochs):
        trainer.epoch = epoch
        main_print(f"\nEpoch {epoch}/{config.num_epochs-1}")
        
        # Set epoch for distributed sampler
        if hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)
        
        # Training
        epoch_train_loss = 0.0
        num_train_steps = 0
        
        progress_bar = tqdm(train_dataloader, disable=rank != 0, file=sys.stdout, desc=f"Training Epoch {epoch}")
        for batch_idx, batch in enumerate(progress_bar):
            # Train step (handles gradient accumulation internally)
            loss = trainer.train_step(batch)
            epoch_train_loss += loss
            num_train_steps += 1
            
            # Update progress bar
            if rank == 0:
                progress_bar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'step': trainer.global_step
                })
            
            # Periodic evaluation
            if trainer.global_step % config.eval_steps == 0:
                eval_loss = trainer.eval_step(eval_dataloader)
                main_print(f"Step {trainer.global_step}: eval_loss = {eval_loss:.4f}")
                
                # Check early stopping
                if trainer.should_stop:
                    main_print("Early stopping triggered, ending training.")
                    break
                
            # Periodic checkpointing
            if trainer.global_step % config.save_steps == 0:
                dist.barrier()
                trainer.save_checkpoint(loss=None)
                dist.barrier()
        
        # Check if early stopping was triggered
        if trainer.should_stop:
            break
        
        # End of epoch evaluation
        avg_train_loss = epoch_train_loss / num_train_steps if num_train_steps > 0 else 0.0
        eval_loss = trainer.eval_step(eval_dataloader)
        
        main_print(f"Epoch {epoch} Summary:")
        main_print(f"  Average Train Loss: {avg_train_loss:.4f}")
        main_print(f"  Eval Loss: {eval_loss:.4f}")
        
        # Save checkpoint at end of epoch
        dist.barrier()
        trainer.save_checkpoint(loss=eval_loss)
        dist.barrier()
    
    # ----------------------------------
    # Final Model Save
    # ----------------------------------
    if is_main_process():
        final_model_path = os.path.join(output_path, "final_model")
        os.makedirs(final_model_path, exist_ok=True)
        
        # Save just the model state dict for inference
        torch.save(student_model.state_dict(), os.path.join(final_model_path, "model.pt"))
        main_print(f"\nSaved final model to {final_model_path}")
    
    # ----------------------------------
    # Cleanup
    # ----------------------------------
    total_time = time.time() - overall_start_time
    main_print(f"\nTraining completed in {total_time/3600:.2f} hours")
    
    # Finish wandb
    if is_main_process():
        wandb.finish()
    
    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Teacher-Student Distillation")
    parser.add_argument("--mixed-precision", action="store_true", default=True,
                        help="Use mixed precision training")
    args = parser.parse_args()
    main(args)
