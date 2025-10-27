import argparse
import os
import pdb
import time
import torch
import torch.distributed as dist
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from tqdm.auto import tqdm
from simple_ensemble import ModelEnsemble
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


# Watch GPU usage in real-time
# ssh kn149  # Your node
# watch -n 1 nvidia-smi

# ==================================================
# Main Training Function
# ==================================================
def main(args):
    """
    Simplified single teacher-student distillation pipeline.
    """
    # ----------------------------------
    # DDP Setup and Initialization
    # ----------------------------------
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    fix_seed(config.seed)

    # ----------------------------------
    # Timer and Logging Start
    # ----------------------------------
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
                "temperature": config.kl_temperature,
                "dataset": config.dataset_name,
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
    if rank == 0:
        teacher_model = AutoModelForCausalLM.from_pretrained(
            config.teacher_model_name,
            torch_dtype=torch.bfloat16,
        )
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        main_print(f"Teacher model loaded on {device} (rank 0 only)")
    else:
        teacher_model = None
        main_print(f"[Rank {rank}] Skipping teacher model (will receive logits from rank 0)")
    
    dist.barrier()

    # ----------------------------------
    # Load Ensemble Model
    # ----------------------------------
    if config.ensemble_dirs is not None:
        ensemble_model = ModelEnsemble()
    else:
        ensemble_model = None

    breakpoint()

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
    # Checkpointer Setup
    # ----------------------------------
    checkpointer = SimpleCheckpointer(output_path)
    
    # ----------------------------------
    # Resume from Checkpoint (if applicable)
    # ----------------------------------
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
        ensemble_model=ensemble_model,
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
        
        # ----------------------------------
        # Epoch Setup
        # ----------------------------------
        if hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)
        
        # Initialize tracking variables
        epoch_train_loss = 0.0
        num_train_steps = 0
        eval_count = 0
        
        # ----------------------------------
        # Training Iteration
        # ----------------------------------
        progress_bar = tqdm(train_dataloader, disable=rank != 0, file=sys.stdout, desc=f"Training Epoch {epoch}")
        for batch_idx, batch in enumerate(progress_bar):
            # Debug mode: stop after max_steps
            if config.debug_mode and trainer.global_step >= config.debug_max_steps:
                main_print(f"[DEBUG MODE] Reached max steps ({config.debug_max_steps}), stopping training")
                break
            
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
            
            # ------ Periodic Evaluation ------
            if trainer.global_step > 0 and trainer.global_step % config.eval_steps == 0:
                dist.barrier()  # Sync before eval
                eval_loss = trainer.eval_step(eval_dataloader)
                main_print(f"Step {trainer.global_step}: eval_loss = {eval_loss:.4f}")
                eval_count += 1
                dist.barrier()  # Sync after eval
            
            # ------ Periodic Checkpointing ------
            # Skip in debug mode to avoid NCCL timeout
            if not config.debug_mode and trainer.global_step > 0 and trainer.global_step % config.save_steps == 0:
                dist.barrier()
                trainer.save_checkpoint(loss=None)
                dist.barrier()

        # Skip end-of-epoch processing in debug mode (already stopped)
        if config.debug_mode and trainer.global_step >= config.debug_max_steps:
            main_print("[DEBUG MODE] Pipeline test complete!")
            break
        
        # ----------------------------------
        # End of Epoch Summary
        # ----------------------------------
        # Compute average training loss
        avg_train_loss = epoch_train_loss / num_train_steps if num_train_steps > 0 else 0.0
        
        # Run final evaluation for the epoch
        eval_loss = trainer.eval_step(eval_dataloader)
        
        main_print(f"Epoch {epoch} Summary:")
        main_print(f"  Average Train Loss: {avg_train_loss:.4f}")
        main_print(f"  Eval Loss: {eval_loss:.4f}")
        
        # ----------------------------------
        # Save Epoch Checkpoint
        # ----------------------------------
        dist.barrier()
        trainer.save_checkpoint(loss=eval_loss)
        dist.barrier()
    
    # ----------------------------------
    # Final Model Save
    # ----------------------------------
    if is_main_process():
        final_model_path = os.path.join(output_path, "final_model")
        os.makedirs(final_model_path, exist_ok=True)
        
        # Use proper FSDP2-compatible state dict extraction
        state_dict_opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        model_state_dict = get_model_state_dict(model=student_model, options=state_dict_opts)
        
        # Save just the model state dict for inference
        torch.save(model_state_dict, os.path.join(final_model_path, "model.pt"))
        main_print(f"\nSaved final model to {final_model_path}")
    
    # ----------------------------------
    # Cleanup and Finalization
    # ----------------------------------
    total_time = time.time() - overall_start_time
    main_print(f"\nTraining completed in {total_time/3600:.2f} hours")
    
    # Finish wandb logging
    if is_main_process():
        wandb.finish()
    
    # Clean up distributed processes
    dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


# ==================================================
# Script Entry Point
# ==================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Teacher-Student Distillation")
    parser.add_argument("--mixed-precision", action="store_true", default=True,
                        help="Use mixed precision training")
    args = parser.parse_args()
    main(args)
