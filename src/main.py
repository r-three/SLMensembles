import argparse
import os
import sys
import time
import torch
from datetime import datetime
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, Qwen2ForCausalLM, get_cosine_schedule_with_warmup
from trl import DataCollatorForCompletionOnlyLM

from datasets import load_dataset
import config
from trainer import Trainer, DistillTrainer
from utils import (CSVLogger, prepare_dataset, format_time_elapsed, 
                  is_main_process, main_print, check_batch_shape, fix_seed,
                  inspect_mixed_precision, inspect_model,
                  set_modules_to_forward_prefetch, set_modules_to_backward_prefetch,
                  create_manifest, build_run_identity, get_directory, init_wandb_run, slurm_term_handler)
from ensemble import ModelEnsemble
from checkpoint import index_checkpoints, best_checkpoint
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from tqdm.auto import tqdm
from shard_weight import *
from utils import fix_seed
import atexit
from pathlib import Path
from datasets import Dataset, DatasetDict
from utils import DistillDataset, get_round_path
from checkpoint import Checkpoint
import wandb
import signal, threading, functools

def train_single_round(round_num, args, config, dataset, output_path, logger, wandb_run, overall_start_time, rank, device):
    """ Train a single round of the ensemble distillation process. """

    main_print(f"\n{'='*50}")
    round_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    main_print(f"--> Starting Round {round_num} at: {round_start_datetime}")
    main_print(f"{'='*50}")

    # ----------------------------------
    # Load and Shard Student Model
    # ----------------------------------

    # with torch.device("meta"):
    # TODO: not exactly following the example
    if round_num == 0 or not config.ensemble_random_init:
        student_model = AutoModelForCausalLM.from_pretrained(
            config.student_model_name,
            torch_dtype=torch.bfloat16,
        ).to('cuda')
    else:
        # Initialize from scratch for round > 1
        cfg = AutoConfig.from_pretrained(config.student_model_name)
        student_model = Qwen2ForCausalLM(cfg).to('cuda')
    
    fsdp_kwargs = {}
    if args.mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
    
    # TODO: not sure if mp will be properly triggered. Didn't verify
    for layer in student_model.model.layers:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(student_model, **fsdp_kwargs)

    inspect_model(student_model)

    if args.explicit_prefetching:
        set_modules_to_forward_prefetch(student_model, num_to_forward_prefetch=2)
        set_modules_to_backward_prefetch(student_model, num_to_backward_prefetch=2)
    
    # ----------------------------------
    # Load Model Weights and Set up Optimizer
    # ----------------------------------

    if resume_info and not checkpointer.is_empty():
        main_print("Loading model from checkpoint...")
        try:
            checkpointer.load_model(student_model)
            main_print("Successfully loaded model from checkpoint")
        except Exception as e:
            main_print(f"Failed to load model from checkpoint: {e}")
            main_print("Loading original pretrained weights instead...")
            student_state_dict = AutoModelForCausalLM.from_pretrained(
                config.student_model_name, torch_dtype=torch.bfloat16
            ).state_dict()
            load_original_weights_fsdp2(student_model, student_state_dict, use_dcp_api=args.dcp_api)
    else:
        main_print("Loading original pretrained weights...")
        student_state_dict = AutoModelForCausalLM.from_pretrained(
            config.student_model_name, torch_dtype=torch.bfloat16
        ).state_dict()
        load_original_weights_fsdp2(student_model, student_state_dict, use_dcp_api=args.dcp_api)
    
    if args.mixed_precision:
        inspect_mixed_precision(student_model)

    # ----------------------------------
    # Load Checkpointed Models
    # ----------------------------------

    # TODO: to be checked if correctly distributed.
    # Ideally it should be properly distributed, for distributed inference. But here I think each proc will save it's own copies.
    # Maybe easy thing to do is to shard all ensemble models. 

    ckpt_index = index_checkpoints(config.checkpoint_dir)

    if len(ckpt_index) != 0:
        completed_rounds = ckpt_index.keys()
        completed_rounds = list(completed_rounds)
        completed_rounds = sorted(completed_rounds)
        is_continuous = completed_rounds == list(range(len(completed_rounds)))
        max_rounds = max(completed_rounds)
        if not is_continuous:
            raise Exception("The rounds obtained is not continuous.")
        best_ckpts = [best_checkpoint(ckpt_index, r) for r in range(max_rounds + 1)]
    else:
        best_ckpts = []
    
    print("Best ckpts: ", best_ckpts)

    if best_ckpts:
        ensemble_model = ModelEnsemble(
            model_paths=best_ckpts,
            model_base=config.student_model_name,
            torch_dtype=torch.bfloat16,
            vocab_size=student_model.config.vocab_size,
        ).to(device)
        ensemble_model.requires_grad_(False)
        start_round = max_rounds + 1
        start_epoch = 0
    else:
        start_round = 0
        start_epoch = 0
        ensemble_model = None

    # Update wandb run name for this round
    if is_main_process() and wandb_run is not None:
        wandb_run.name = f"{config.run_name}_round_{round_num}"
        wandb_run.log({"round": round_num})

    # ----------------------------------
    # Initialize trainer
    # ----------------------------------

    # TODO: move all the init, prepare steps and DS and DL into the class
    train_dataloader, eval_dataloader = prepare_dataset(
        dataset['train'],
        dataset['test'],
        config,
        1024,
        config.seed,
    )  # Just to get the length, initialize again for each epoch.
    num_training_steps = len(train_dataloader) * config.num_train_epochs
    num_warmup_steps = config.warmup_steps  # e.g., 10% warmup
    lr_scheduler = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Restore learning rate scheduler state if resuming
    if resume_info and not checkpointer.is_empty():
        training_state = checkpointer.load_training_state()
        if training_state and 'lr_scheduler_state' in training_state:
            try:
                lr_scheduler.load_state_dict(training_state['lr_scheduler_state'])
                main_print("Successfully restored learning rate scheduler state")
            except Exception as e:
                main_print(f"Failed to restore LR scheduler state: {e}")
        
        # Restore RNG states for reproducibility
        if training_state and 'rng_states' in training_state:
            try:
                restore_rng_states(training_state['rng_states'])
                main_print("Successfully restored RNG states")
            except Exception as e:
                main_print(f"Failed to restore RNG states: {e}")
    # Initialize trainer with logger and round information
    trainer = DistillTrainer(
        student_model, 
        optim, 
        lr_scheduler, 
        config, 
        ensemble_model,
        logger=logger,
        round_num=round_num,
        overall_start_time=overall_start_time,  
        checkpointer=checkpointer,
        wandb_run=wandb_run if is_main_process() else None,
        report_to="wandb" if is_main_process() else "none",
    )


# TODO: train model to repdict loss? 

    # ----------------------------------
    # SLURM Signal Handling
    # ----------------------------------
    handler = functools.partial(slurm_term_handler, trainer=trainer)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    
    # Try to resume from latest checkpoint if it exists
    if resume_info and not checkpointer.is_empty():
        if trainer.load_checkpoint(checkpoint_dir):
            main_print(f"Successfully resumed from checkpoint in round {trainer.round_num}")
        else:
            main_print("Failed to resume from checkpoint, starting training from scratch")
    else:
        main_print("Starting training from scratch")
    
    trainer.prepare_train()

    # ----------------------------------
    # Epoch Loop
    # ----------------------------------

    for epoch_num in range(start_epoch, config.num_train_epochs):
        epoch_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        main_print(f"\n{'='*50}")
        main_print(f"--> Starting Epoch {epoch_num} at: {epoch_start_datetime}")
        main_print(f"{'='*50}")
        
        # ----------------------------------
        # Prepare dataset
        # ----------------------------------

        train_dataloader, eval_dataloader = prepare_dataset(
            dataset['train'],
            dataset['test'],
            config,
            1024,
            config.seed + round_num + epoch_num,
        )
        if is_main_process():
            check_batch_shape(train_dataloader)
        
        train_dl_iterator = iter(train_dataloader)

        # ----------------------------------
        # Inner Training Loop
        # ----------------------------------

        for step_idx in tqdm(
            range(len(train_dataloader)),
            disable=rank != 0,
            file=sys.__stdout__,
        ):
            if args.explicit_prefetching:
                trainer.model.unshard()
            batch = next(train_dl_iterator)
            trainer.step(batch, eval_dataloader, epoch_num, wandb_run)
            
            # Simple periodic checkpointing for SLURM interruption safety
            if (trainer.save_steps and trainer.save_steps > 0 and trainer.tr_step % trainer.save_steps == 0):
                torch.distributed.barrier()
                # Build your small metadata dict if you want (epoch/step/scheduler/RNG)
                training_state = create_training_state(
                    round_num=trainer.round_num,
                    epoch_num=current_epoch,
                    step=trainer.tr_step,
                    current_loss=trainer.current_loss,
                    min_loss=trainer.min_eval_loss,
                    lr_scheduler_state=lr_scheduler.state_dict() if lr_scheduler else None,
                    rng_states=capture_rng_states(),  # if you have this
                )
                checkpointer.save(student_model, optim, trainer.round_num, trainer.tr_step, trainer.current_loss, training_state)
                if rank == 0:
                    main_print(f"Saved periodic checkpoint at step {trainer.tr_step}")
        # ----------------------------------
        # Save checkpoint
        # ----------------------------------
        torch.distributed.barrier()  # Wait for all ranks to finish the epoch
        
        # Save end-of-epoch checkpoint using standardized structure
        checkpoint_dir = os.path.join(output_path, "checkpoints")
        trainer.save_checkpoint(checkpoint_dir)
        if rank == 0:
            main_print(f"Saved end-of-epoch checkpoint: round {round_num}, epoch {epoch_num}, step {trainer.tr_step}")

        # ----------------------------------
        # Update wandb run name and log metrics
        # ----------------------------------

    if is_main_process() and wandb_run is not None:
        wandb_run.name = f"round_{round_num}_epoch_{epoch_num}"
        wandb_run.log({"epoch": epoch_num, "round": round_num}) 

    # Return the checkpoint path and metrics for this round
    final_checkpoint_path = checkpointer.last_checkpoint_path if hasattr(checkpointer, 'last_checkpoint_path') else None
    round_metrics = {
        'round_num': round_num,
        'final_loss': trainer.current_loss if hasattr(trainer, 'current_loss') else None,
        'min_eval_loss': trainer.min_eval_loss if hasattr(trainer, 'min_eval_loss') else None,
        'total_steps': trainer.tr_step if hasattr(trainer, 'tr_step') else None,
    }
    
    return final_checkpoint_path, round_metrics


def main(args):

    # ----------------------------------
    # Pipeline
    # ----------------------------------
    # Raw Dataset → Chat Template → Tokenization → Label Creation → Teacher Inference → Logit Caching → Distributed Sampling → Custom Collation → FSDP2 Model → Training Loop
    #  ↓              ↓                  ↓               ↓                 ↓                   ↓                  ↓                  ↓               ↓             ↓
    # Messages    Single String       Token IDs     Loss Labels       Top-K Logits        Cached Data           Batches            Padded         Sharded       Training
    # (JSON)      (Text)             (Tensors)       (-100/IDs)         (Tensors)           (Disk)               (GPU)              (GPU)          (GPU)         (GPU)

    # ----------------------------------
    # Directory Structure
    # ----------------------------------   
    # Run_ID Format: {timestamp}-{git_hash}-{hyperparameter_fingerprint}
    # Example: "20241221-143025-a1b2c3d4-e5f6g7h8"
    #
    # {config.base_output_dir}/
    # └── {run_id}/                           # Unique run directory
    #     ├── CSV_metrics.csv                 # Training metrics log (CSVLogger)
    #     ├── manifest.txt                    # Run metadata and configuration
    #     ├── STATUS.RUNNING                  # Status sentinel (during training)
    #     ├── STATUS.DONE                     # Status sentinel (on completion)
    #     ├── STATUS.FAILED                   # Status sentinel (on failure)
    #     └── checkpoints/                    # Model checkpoints directory (Checkpointer)
    #         ├── 0/                          # Round 0 checkpoints
    #         │   ├── step_00001000_loss_2.3456/
    #         │   │   ├── model_state_dict.pt
    #         │   │   ├── optim_state_dict.pt
    #         │   │   └── training_state.pt
    #         │   └── step_00003000_loss_1.9876/ # Final checkpoint for round 0
    #         ├── 1/                          # Round 1 checkpoints
    #         │   ├── step_00001000_loss_1.8765/
    #         │   └── step_00002000_loss_1.7654/
    #         └── 2/                          # Round 2 checkpoints
    #             └── step_00001500_loss_1.6543/
    #     ├── round_0_model/                  # Final inference-ready model for round 0
    #     │   ├── model.safetensors
    #     │   ├── config.json
    #     │   ├── tokenizer.json
    #     │   └── generation_config.json
    #     └── round_1_model/                  # Final inference-ready model for round 1
    #         ├── model.safetensors
    #         └── ... (etc.)

    # ----------------------------------
    # DDP Setup
    # ----------------------------------

    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    fix_seed(config.seed)

    overall_start_time = time.time()
    overall_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    main_print(f"--> Starting training at: {overall_start_datetime}\n")

    # ----------------------------------
    # Exception Handling
    # ----------------------------------
    _exit_once = threading.Event()

    default_excepthook = sys.excepthook
    sys.excepthook = _on_exception

    # ----------------------------------
    # Run Configuration 
    # ----------------------------------
    fix_seed(config.seed)

    if config.resume_from_checkpoint:
        output_path = checkpointed_dir
    else:
        run_id, slug, wandb_name, wandb_id = build_run_identity()
        output_path = get_directory(run_id)

    # ----------------------------------
    # Dataset Loading
    # ----------------------------------
    dataClass = DistillDataset()
    dataset = dataClass.get_dataset() if config.synthetic_data else dataClass.get_teacher_logprobs()

    # ----------------------------------
    # Set Up Optimizer and LR Scheduler
    # ----------------------------------
    optim = torch.optim.Adam(student_model.parameters(), lr=config.learning_rate)
    num_training_steps = len(train_dataloader) * config.num_train_epochs
    num_warmup_steps = config.warm_up_steps
    lr_scheduler = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # ----------------------------------
    # Checkpoint Logic
    # ----------------------------------

    if config.resume_from_checkpoint:
        checkpointer = Checkpointer(output_path) # output path is the prev checkpoint dir

        student_model = AutoModelForCausalLM.from_pretrained(
            config.student_model_name,
            torch_dtype=torch.bfloat16,
        ).to('cuda')

        state = checkpointer.load(student_model, optimizer)

        global_step = int(state.get("global_step", state.get("step", 0)))
        epoch = int(state.get("epoch", 0))

        # TODO: load CSV file as well

        if state.get("lr_scheduler_state") and lr_scheduler: lr_scheduler.load_state_dict(state["lr_scheduler_state"])
        start_round = round + 1
        start_epoch = epoch + 1
        resume_info = True
    else:
        checkpointer = Checkpointer(os.path.join(output_path, "checkpoints"))
        start_round = 0
        start_epoch = 0
        resume_info = False

    # ----------------------------------
    # Logger config
    # ----------------------------------

    logger = None
    if is_main_process():
        logger = CSVLogger(output_path, fieldnames=config.CSV_COLUMNS, overall_start_time=overall_start_time)
        atexit.register(logger.flush)

    # ----------------------------------
    # Initialize wandb
    # ----------------------------------
    wandb_run = init_wandb_run() if is_main_process() else None

    # ----------------------------------
    # Manifest file
    # ----------------------------------

    if config.resume_from_checkpoint:
        # TODO: load round info and everything from the manifest file
        round = load_from_manifest()
    else: 
        if is_main_process(): create_manifest(output_path, start_time_str=overall_start_datetime, wandb_run=wandb_run, wandb_id=wandb_id)

    # ----------------------------------
    # Metrics
    # ----------------------------------
    if is_main_process():   
        metadata_dict = {
            "Run id": run_id,
            "Wandb run id": wandb_run.id if wandb_run else None,
            "Description": config.description,
            "Teacher Model": config.teacher_model_name,
            "Student Model": config.student_model_name,
            "Dataset Name": config.dataset_name,
            "Dataset Type": config.dataset_type,
            "Alpha": config.alpha,
            "Learning rate": config.learning_rate,
            "Total Rounds": config.total_rounds,
            "Steps per Round": config.steps_per_round,
            "Eval batch size": config.eval_batch_size,
            "Start Time": overall_start_datetime,
            "Model Save Dir": output_path,
        }
    main_print("\n==== RUN CONFIGURATION ====")
    main_print(f"Run: {run_id}")
    main_print(f"Created logging directory: {output_path}")
    main_print(f"Models stored in: {output_path}")
    main_print(f"{run_id}")
    main_print(f"{config.description}\n")
    if is_main_process():
        for k, v in metadata_dict.items():
            main_print(f"{k}: {v}")
    main_print("===========================")

    if is_main_process(): logger.log(function="main", phase="none", round_num=0, metadata=metadata_dict)


    # ----------------------------------
    # Determine Starting Round
    # ----------------------------------
    
    # Check existing checkpoints to determine where to start
    ckpt_index = index_checkpoints(os.path.join(output_path, "checkpoints"))
    if len(ckpt_index) != 0:
        completed_rounds = sorted(list(ckpt_index.keys()))
        is_continuous = completed_rounds == list(range(len(completed_rounds)))
        if not is_continuous:
            raise Exception("The rounds obtained is not continuous.")
        start_round = max(completed_rounds) + 1
    else:
        start_round = 0
    
    main_print(f"Starting training from round {start_round}")

    # ----------------------------------
    # Outer Training Loop
    # ----------------------------------

    for round_num in range(start_round, config.total_rounds):
        # Train a single round using the extracted function
        checkpoint_path, metrics = train_single_round(
            round_num=round_num,
            args=args,
            config=config,
            dataset=dataset,
            output_path=output_path,
            logger=logger,
            wandb_run=wandb_run,
            overall_start_time=overall_start_time,
            rank=rank,
            device=device
        )
        
        # Log round completion
        if is_main_process():
            main_print(f"Completed round {round_num}")
            main_print(f"Round metrics: {metrics}")
            if logger:
                logger.log(
                    function="main", 
                    phase="round_complete", 
                    round_num=round_num, 
                    metadata=metrics
                ) 

        # ----------------------------------
        # Cleanup
        # ----------------------------------
        torch.distributed.barrier()
        if is_main_process() and wandb_run is not None:
            wandb_run.finish()
            main_print("--> Finished wandb run")
            
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch FSDP2 example")
    parser.add_argument("--explicit-prefetching", action="store_true", default=False)
    parser.add_argument("--mixed-precision", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
