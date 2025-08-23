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
                  create_manifest, build_run_identity, get_directory, init_wandb_run, slurm_term_handler, 
                  _on_exception, ManifestManager, DistillDataset, get_round_path)
from ensemble import ModelEnsemble, EnsembleLoader
from checkpoint import index_checkpoints, best_checkpoint, Checkpointer, Checkpoint
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from tqdm.auto import tqdm
from shard_weight import *
import atexit
from pathlib import Path
from datasets import Dataset, DatasetDict
import wandb
import signal, threading, functools

def train_single_round(start_round, round_num, dataset, output_path, logger, wandb_run, overall_start_time, rank, device, ensembleloader, args, manifest, lr_scheduler=None):
    """ Train a single round of the ensemble distillation process. """
    main_print(f"\n{'='*50}")
    round_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    main_print(f"--> Starting Round {round_num} at: {round_start_datetime}")
    main_print(f"{'='*50}")

    # -------------------------------------
    # Update wandb run name for this round
    # -------------------------------------
    if is_main_process() and wandb_run is not None:
        wandb_run.name = f"{config.run_name}_round_{round_num}"
        wandb_run.log({"round": round_num})
    
    # Update manifest with current round
    if is_main_process():
        manifest.update('round', round_num, section='train')
        manifest.set_status('TRAINING')

    # ----------------------------------
    # Load and Shard Student Model
    # ----------------------------------
    if not (round_num == start_round and config.resume_from_checkpoint):
        if not config.ensemble_random_init:
            student_model = AutoModelForCausalLM.from_pretrained(
                config.student_model_name,
                torch_dtype=torch.bfloat16,
            ).to('cuda')
        else:
            # Initialize from scratch
            cfg = AutoConfig.from_pretrained(config.student_model_name)
            student_model = Qwen2ForCausalLM(cfg).to('cuda')
        
        # ----------------------------------
        # Mixed precision setup
        # ----------------------------------
        fsdp_kwargs = {}
        if args.mixed_precision:
            fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,  # 16-bit precision for model parameters
                reduce_dtype=torch.float32,  # 32-bit precision for reduction operations
            )

        # ----------------------------------
        # Shard Model
        # ----------------------------------
        for layer in student_model.model.layers:
            fully_shard(layer, **fsdp_kwargs)
        fully_shard(student_model, **fsdp_kwargs)

        if args.explicit_prefetching:
            set_modules_to_forward_prefetch(student_model, num_to_forward_prefetch=2)
            set_modules_to_backward_prefetch(student_model, num_to_backward_prefetch=2)

        inspect_model(student_model)
    
        if args.mixed_precision: inspect_mixed_precision(student_model)

    # ----------------------------------
    # Load or Update Ensemble Models
    # ----------------------------------
    if hasattr(ensembleloader, 'current_ensemble'):
        ensemble = ensembleloader.current_ensemble
    else:
        ensemble = ensembleloader.load_or_update_ensemble(None, device="cuda")

    # ----------------------------------
    # Set Up Optimizer and LR Scheduler
    # ----------------------------------
    optim = torch.optim.Adam(student_model.parameters(), lr=config.learning_rate)
    train_dataloader, _ = prepare_dataset(dataset['train'], dataset['test'])
    num_training_steps = len(train_dataloader) * config.num_train_epochs
    num_warmup_steps = config.warm_up_steps
    if not lr_scheduler:
        lr_scheduler = get_cosine_schedule_with_warmup(
            optim,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    # ----------------------------------
    # Initialize trainer
    # ----------------------------------
    trainer = DistillTrainer(
        student_model, 
        optim, 
        lr_scheduler,
        ensemble,
        logger=logger,
        round_num=round_num,
        checkpointer=checkpointer,
        overall_start_time=overall_start_time,  
        wandb_run=wandb_run if is_main_process() else None,
    )
    trainer.prepare_train()

    # ----------------------------------
    # SLURM Signal Handling
    # ----------------------------------
    handler = functools.partial(slurm_term_handler, trainer=trainer)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    # Needs to exit cleanly and save the very latest model checkpoint

    # ----------------------------------
    # Epoch Loop
    # ----------------------------------
    for epoch_num in range(0, config.num_train_epochs):
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
        )
        if hasattr(train_dataloader, "sampler") and hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch_num)
        if is_main_process():
            check_batch_shape(train_dataloader)

        train_dl_iterator = iter(train_dataloader)

        # ----------------------------------
        # Training Loop
        # ----------------------------------
        for step_idx in tqdm(range(len(train_dataloader)), disable=rank != 0, file=sys.__stdout__):
            if args.explicit_prefetching:
                trainer.model.unshard()
            batch = next(train_dl_iterator)
            trainer.step(batch, eval_dataloader, epoch_num)

        dist.barrier()

    # ----------------------------------
    # Save model
    # ----------------------------------
    ensemble_dir = ensembleloader.save_model_for_ensemble(student_model, round_num)
    main_print(f"Saved ensemble model at: {ensemble_dir}")

    del student_model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------------------------------------
    # Update wandb run name and log metrics
    # ---------------------------------------
    if is_main_process() and wandb_run is not None:
        wandb_run.name = f"round_{round_num}_epoch_{epoch_num}"
        wandb_run.log({"epoch": epoch_num, "round": round_num}) 

    # ---------------------------------------
    # Return round metrics
    # ---------------------------------------
    round_metrics = {
        'round_num': round_num,
        'final_loss': trainer.current_loss if hasattr(trainer, 'current_loss') else None,
        'min_eval_loss': trainer.min_eval_loss if hasattr(trainer, 'min_eval_loss') else None,
        'total_steps': trainer.tr_step if hasattr(trainer, 'tr_step') else None,
    }
    # TODO: ensure the manifest file has these columns
    # Update manifest with round metrics
    if is_main_process():
        manifest.update({
            f'round_{round_num}_final_loss': round_metrics['final_loss'],
            f'round_{round_num}_min_eval_loss': round_metrics['min_eval_loss'],
            f'round_{round_num}_total_steps': round_metrics['total_steps'],
        }, section='outcomes')
    
    return ensemble_dir, round_metrics


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
    # └── {output_path (run_id)}/                           # Unique run directory
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
    #     │   ├── model_state_dict.pt
    #     │   ├── config.json
    #     │   ├── tokenizer.json
    #     │   └── generation_config.json
    #     └── round_1_model/                  # Final inference-ready model for round 1
    #         ├── model_state_dict.pt
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
    default_excepthook = sys.excepthook
    sys.excepthook = _on_exception

    # ----------------------------------
    # Run Configuration 
    # ----------------------------------
    fix_seed(config.seed)

    if config.resume_from_checkpoint:
        output_path = config.checkpointed_dir
        run_id = os.path.basename(output_path)
        wandb_id = None  # Will be loaded from manifest
        wandb_name = None
        slug = None
    else:
        run_id, slug, wandb_name, wandb_id = build_run_identity()
        output_path = get_directory(run_id)

    # ----------------------------------
    # Dataset Loading
    # ----------------------------------
    dataClass = DistillDataset()
    dataset = dataClass.get_dataset()

    # ----------------------------------
    # Checkpoint Logic
    # ----------------------------------
    if config.resume_from_checkpoint:
        checkpointer = Checkpointer(output_path) # output path is the path from prev checkpoint

        student_model = AutoModelForCausalLM.from_pretrained(
            config.student_model_name,
            torch_dtype=torch.bfloat16,
        ).to('cuda')

        state = checkpointer.load(student_model, optimizer)

        if state.get("lr_scheduler_state") and lr_scheduler: lr_scheduler.load_state_dict(state["lr_scheduler_state"])
        global_step = int(state.get("global_step", state.get("step", 0)))
        epoch = int(state.get("epoch", 0))
        start_epoch = epoch + 1
        resume_info = True
    else:
        checkpointer = Checkpointer(os.path.join(output_path, "checkpoints"))
        start_epoch = 0
        resume_info = False

    # ----------------------------------
    # Logging and WandB config
    # ----------------------------------
    logger = None

    if is_main_process():
        logger = CSVLogger(output_path, fieldnames=config.CSV_COLUMNS, overall_start_time=overall_start_time)
        atexit.register(logger.flush)
    
    wandb_run = init_wandb_run() if is_main_process() else None

    # ----------------------------------
    # Manifest file
    # ----------------------------------
    if config.resume_from_checkpoint:
        # Load existing manifest
        manifest_path = os.path.join(output_path, "manifest.txt")
        manifest = ManifestManager(manifest_path)
        start_round = manifest.get('round', section='train', default=0)
        start_round = start_round + 1 if start_round > 0 else 0
    else:
        if is_main_process():
            manifest = create_manifest(output_path, start_time_str=overall_start_datetime, 
                                     wandb_run=wandb_run, wandb_id=wandb_id)
        else:
            manifest_path = os.path.join(output_path, "manifest.txt")
            manifest = ManifestManager(manifest_path)
        start_round = 0

    # ----------------------------------
    # Ensemble Loader
    # ----------------------------------
    ensembleloader = EnsembleLoader(output_path)

    # ----------------------------------
    # Metrics Output and Logging
    # ----------------------------------
    if is_main_process():   
        metadata_dict = {
            "Run id": run_id,
            "Slug": slug if slug else "N/A",
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

    if is_main_process() and not config.resume_from_checkpoint: logger.log(function="main", phase="none", round_num=0, metadata=metadata_dict)

    # ----------------------------------
    # Outer Training Loop
    # ----------------------------------
    ensemble_model = None
    if config.resume_from_checkpoint:
        ensemble_model = ensembleloader.load_or_update_ensemble(None, device="cuda")
        
    for round_num in range(start_round, config.total_rounds):
        ensembleloader.current_ensemble = ensemble_model
        
        ensemble_dir, metrics = train_single_round(
            start_round = start_round,
            round_num=round_num,
            dataset=dataset,
            output_path=output_path,
            logger=logger,
            wandb_run=wandb_run,
            overall_start_time=overall_start_time,
            rank=rank,
            device=device,
            ensembleloader = ensembleloader,
            args=args,
            manifest=manifest,
            lr_scheduler=lr_scheduler if (round_num == start_round and config.resume_from_checkpoint) else None,
        )
        
        if is_main_process():
            main_print(f"Completed round {round_num}")
            main_print(f"Round metrics: {metrics}")

        if ensemble_model is None:
            ensemble_model = ModelEnsemble(
                [ensemble_dir],
                torch_dtype=torch.bfloat16,
                vocab_size=config.student_vocab_size,
            )
            ensemble_model = ensemble_model.to("cuda")
        else:
            ensemble_model.add_model(ensemble_dir)

    # ----------------------------------
    # Cleanup and Final Summary
    # ----------------------------------
    dist.barrier()
    if is_main_process() and wandb_run is not None:
        wandb_run.finish()
        main_print("--> Finished wandb run")
        
        # Final summary
        total_time = time.time() - overall_start_time
        if is_main_process():
            main_print("\n" + "="*60)
            main_print("TRAINING COMPLETED SUCCESSFULLY")
            main_print("="*60)
            main_print(f"Total rounds completed: {config.total_rounds}")
            main_print(f"Output directory: {output_path}")
            main_print(f"Total training time: {format_time_elapsed(total_time)}")
            main_print(f"Final manifest status: {manifest.get('status', default='UNKNOWN')}")
            
            # Update manifest with final status
            manifest.finalize(success=True, wall_time_sec=total_time)
            main_print("="*60)
    
    # Destroy process group
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch FSDP2 example")
    parser.add_argument("--explicit-prefetching", action="store_true", default=False)
    parser.add_argument("--mixed-precision", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
