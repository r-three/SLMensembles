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
                  create_manifest, build_run_identity, get_directory)
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

def main(args):

    # ----------------------------------
    # Pipeline
    # ----------------------------------
    # Raw Dataset → Chat Template → Tokenization → Label Creation → Teacher Inference → Logit Caching → Distributed Sampling → Custom Collation → FSDP2 Model → Training Loop
    #  ↓              ↓                  ↓               ↓                 ↓                   ↓                  ↓                  ↓               ↓             ↓
    # Messages    Single String       Token IDs     Loss Labels       Top-K Logits        Cached Data           Batches            Padded         Sharded       Training
    # (JSON)      (Text)             (Tensors)       (-100/IDs)         (Tensors)           (Disk)               (GPU)              (GPU)          (GPU)         (GPU)

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
    # Logging and Run Configuration 
    # ----------------------------------

    run_id, slug, wandb_name, wandb_id = build_run_identity()
    output_path = get_directory(run_id)

    logger = None
    if is_main_process():
        logger = CSVLogger(output_path, fieldnames=config.CSV_COLUMNS, overall_start_time=overall_start_time)
        atexit.register(logger.flush)

    checkpoint_dir = os.path.join(output_path, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    main_print(f"Checkpoints will be saved to: {checkpoint_dir}")

    # ----------------------------------
    # Dataset Loading
    # ----------------------------------

    dataClass = DistillDataset()
    dataset = dataClass.get_dataset() if config.synthetic_data else dataClass.get_teacher_logprobs()

    # ----------------------------------
    # Initialize wandb (single run per experiment)
    # ----------------------------------
    if is_main_process():
        try:
            wandb_run = wandb.init(
                project="slm-ensembles",
                id=run_id,   
                name=run_id,
                config={
                    "model_name": config.student_model_name,
                    "teacher_model": config.teacher_model_name,
                    "learning_rate": config.learning_rate,
                    "batch_size": config.per_device_train_batch_size * torch.distributed.get_world_size(),
                    "max_length": 1024,
                    "alpha": config.alpha,
                    "seed": config.seed,
                    "description": config.description,
                    "dataset_name": config.dataset_name,
                    "dataset_type": config.dataset_type,
                    "total_rounds": config.total_rounds,
                    "num_train_epochs": config.num_train_epochs,
                    "gradient_accumulation_steps": config.gradient_accumulation_steps,
                    "max_grad_norm": getattr(config, 'max_grad_norm', 1.0),
                },
                tags=["knowledge-distillation", "fsdp2", "ensemble"],
                resume="allow",
            )
            main_print(f"--> Initialized wandb run: {wandb_run.name}")
        except Exception as e:
            main_print(f"--> Warning: Failed to initialize wandb: {e}")
            main_print("--> Continuing without wandb logging")
            wandb_run = None
    else:
        wandb_run = None

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

    if is_main_process():
        logger.log(
            function="main",
            phase="none",
            round_num=0,
            metadata=metadata_dict,
        )

    # ----------------------------------
    # Manifest File Creation
    # ----------------------------------
    if is_main_process():
        create_manifest(
            output_path,
            start_time_str=overall_start_datetime,
            wandb_run=wandb_run,
            wandb_id=wandb_id,
        )

    default_excepthook = sys.excepthook
    sys.excepthook = _on_exception

    # ----------------------------------
    # Checkpoint Logic
    # ----------------------------------

    checkpointer = Checkpointer(checkpoint_dir)

    if resume_info and not checkpointer.is_empty():
        main_print("Loading model from checkpoint via DCP.")
        checkpointer.load_model(student_model)
    else:
        main_print("Loading original pretrained weights for the initial run.")

    # ----------------------------------
    # Set Up Optimizer and LR Scheduler
    # ----------------------------------
    optim = torch.optim.Adam(student_model.parameters(), lr=config.learning_rate)

    if resume_info and not checkpointer.is_empty():
        main_print("Loading optimizer from checkpoint via DCP.")
        checkpointer.load_optim(student_model, optim)

    training_state = checkpointer.load_training_state()
    if training_state and 'lr_scheduler_state' in training_state:
        lr_scheduler.load_state_dict(training_state['lr_scheduler_state'])




    
    # Load checkpoint index for ensemble building and resumption
    ckpt_index = index_checkpoints(checkpoint_dir)
    if len(ckpt_index) != 0:
        completed_rounds = list(sorted(ckpt_index.keys()))
        is_continuous = completed_rounds == list(range(len(completed_rounds)))
        max_rounds = max(completed_rounds)
        if not is_continuous:
            raise Exception("The rounds obtained is not continuous.")
        best_ckpts = [best_checkpoint(ckpt_index, r) for r in range(max_rounds + 1)]
        main_print(f"Found {len(best_ckpts)} completed rounds: {best_ckpts}")
        start_round = max_rounds + 1
        start_epoch = 0
        resume_info = True
    else:
        best_ckpts = []
        start_round = 0
        start_epoch = 0
        ensemble_model = None
        resume_info = False

    # ----------------------------------
    # Outer Training Loop
    # ----------------------------------

    for round_num in range(start_round, config.total_rounds):
        fix_seed(config.seed)

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
        # checkpointer = Checkpointer(checkpoint_dir)
        # student_state_dict = AutoModelForCausalLM.from_pretrained(config.student_model_name, torch_dtype=torch.bfloat16).state_dict()
        
        # TODO: also checkpoint the dataloader sampler
        # TODO: fix to device issue (can't initialize). If use to_empty(device="cuda"), cannot reload the state_dict. If load state_dict, will get: NotImplementedError: Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() when moving module from meta to a different device.
        # if checkpointer.last_training_time is None:
        #     student_model.to(device="cuda")
            # checkpointer.load_org_model(student_model, student_state_dict)
            # load_original_weights_fsdp2(student_model, student_state_dict, use_dcp_api=False)
        # else:
            # checkpointer.load_model(student_model)
        # Load model weights - either from checkpoint or original pretrained
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
        
        # Configure simple checkpointing with standardized structure
        trainer.checkpoint_dir = checkpoint_dir
        trainer.save_steps = getattr(config, 'checkpoint_every_n_steps', 500)
        
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
            trainer.save_checkpoint(checkpoint_dir)
            if rank == 0:
                main_print(f"Saved end-of-epoch checkpoint: round {round_num}, epoch {epoch_num}, step {trainer.tr_step}")

            # ----------------------------------
            # Update wandb run name and log metrics
            # ----------------------------------

        if is_main_process() and wandb_run is not None:
            wandb_run.name = f"round_{round_num}_epoch_{epoch_num}"
            wandb_run.log({"epoch": epoch_num, "round": round_num}) 

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
