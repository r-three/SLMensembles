import argparse
import os, gc, time, sys, pdb
import torch
import atexit
from datetime import datetime
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from pathlib import Path

from datasets import load_dataset, Dataset, DatasetDict
import config
from trainer import Trainer, DistillTrainer
from utils import CSVLogger, DistillDataset, evaluate_model, prepare_dataset, format_time_elapsed, get_round_path, is_main_process, main_print, check_batch_shape
from ensemble import ModelEnsemble

from checkpoint import Checkpointer, Checkpoint, index_checkpoints, best_checkpoint
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from utils import inspect_mixed_precision, inspect_model
from tqdm.auto import tqdm

from shard_weight import *

def set_modules_to_forward_prefetch(model, num_to_forward_prefetch):
    for i, layer in enumerate(model.layers):
        if i >= len(model.layers) - num_to_forward_prefetch:
            break
        layers_to_prefetch = [
            model.layers[i + j] for j in range(1, num_to_forward_prefetch + 1)
        ]
        layer.set_modules_to_forward_prefetch(layers_to_prefetch)


def set_modules_to_backward_prefetch(model, num_to_backward_prefetch):
    for i, layer in enumerate(model.layers):
        if i < num_to_backward_prefetch:
            continue
        layers_to_prefetch = [
            model.layers[i - j] for j in range(1, num_to_backward_prefetch + 1)
        ]
        layer.set_modules_to_backward_prefetch(layers_to_prefetch)


def main(args):

    # ----------------------------------
    # Device Setup
    # ----------------------------------

    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    torch.manual_seed(0)

    overall_start_time = time.time()
    overall_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    main_print(f"--> Starting training at: {overall_start_datetime}\n")

    # ----------------------------------
    # Logging and Run Name
    # ----------------------------------

    main_print("--> Setting up logging and run name")

    log_dir = None
    logger = None

    if is_main_process():
        log_dir = config.get_directory(config.log_dir)
        logger = CSVLogger(log_dir, fieldnames=config.CSV_COLUMNS, overall_start_time=overall_start_time)
        atexit.register(logger.flush)

    output_path = config.get_directory(config.base_output_dir)
    run_name = f"{os.path.basename(output_path)}"
    os.makedirs(config.logprob_cache_path, exist_ok=True)

    # ----------------------------------
    # Loading the Teacher Dataset
    # ----------------------------------
    dataClass = DistillDataset()
    dataset = dataClass.get_dataset() if config.synthetic_data else dataClass.get_teacher_logprobs()

    # ----------------------------------
    # Metrics
    # ----------------------------------

    if is_main_process():   
        metadata_dict = {
            "Custom run name": config.custom_run_name,
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
            "ID string": config.id_string,
            "Description": config.description,
        }
    main_print("\n==== RUN CONFIGURATION ====")

    main_print(f"Run: {run_name}")
    main_print(f"Created logging directory: {log_dir}")
    main_print(f"Models stored in: {output_path}")

    main_print(f"{config.id_string}")
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

    ckpt_index = index_checkpoints('distill_ckpt/')
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
        start_round = max_rounds + 1
        start_epoch = 0
    else:
        start_round = 0
        start_epoch = 0
        ensemble_model = None


    for round_num in range(start_round, config.total_rounds):
        # ----------------------------------
        # Outer Training Loop
        # ----------------------------------

        round_start_time = time.time()
        round_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        main_print(f"\n{'='*50}")
        main_print(f"--> Starting Round {round_num} at: {round_start_datetime}")
        main_print(f"{'='*50}")

        # ----------------------------------
        # Load Student 
        # ----------------------------------
        # with torch.device("meta"):
        # TODO: not exactly following the example
        student_model = AutoModelForCausalLM.from_pretrained(
            config.student_model_name,
            torch_dtype=torch.bfloat16,
        ).to('cuda')
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
        # Load checkpoint
        # ----------------------------------
            
        checkpointer = Checkpointer("checkpoints", dcp_api=args.dcp_api)
        student_state_dict = AutoModelForCausalLM.from_pretrained(config.student_model_name, torch_dtype=torch.bfloat16).state_dict()
        # TODO: also checkpoint the dataloader sampler
        # TODO: fix to device issue (can't initialize). If use to_empty(device="cuda"), cannot reload the state_dict. If load state_dict, will get: NotImplementedError: Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() when moving module from meta to a different device.
        # if checkpointer.last_training_time is None:
        #     student_model.to(device="cuda")
            # checkpointer.load_org_model(student_model, student_state_dict)
            # load_original_weights_fsdp2(student_model, student_state_dict, use_dcp_api=False)
        # else:
            # checkpointer.load_model(student_model)
        
        if args.mixed_precision:
            inspect_mixed_precision(student_model)

        optim = torch.optim.Adam(student_model.parameters(), lr=config.learning_rate)
        if checkpointer.last_training_time is not None:
            checkpointer.load_optim(student_model, optim)

        # ----------------------------------
        # Load Existing Models
        # ----------------------------------
        # TODO: to be checked if correctly distributed.
        # Ideally it should be properly distributed, for distributed inference. But here I think each proc will save it's own copies.
        # Maybe easy thing to do is to shard all ensemble models. 

        ckpt_index = index_checkpoints('distill_ckpt/')
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

        # ----------------------------------
        # Initialize trainer
        # ----------------------------------
        # TODO: move all the init, prepare steps and DS and DL into the class
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optim, factor=1)
        trainer = DistillTrainer(student_model, optim, lr_scheduler, config, ensemble_model)
        trainer.prepare_train()

        for epoch_num in range(start_epoch, config.num_train_epochs):
            epoch_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            main_print(f"\n{'='*50}")
            main_print(f"--> Starting Epoch {epoch_num} at: {epoch_start_datetime}")
            main_print(f"{'='*50}")
            # ----------------------------------
            # Prepare dataset
            # ----------------------------------
            train_dataloader, eval_dataloader = prepare_dataset(dataset['train'], dataset['test'], config, 1024, config.seed + round_num + epoch_num)
            if is_main_process():
                check_batch_shape(train_dataloader)
            
            train_dl_iterator = iter(train_dataloader)

            # ----------------------------------
            # Inner Training Loop
            # ----------------------------------

            for i in tqdm(
                range(len(train_dataloader)),
                disable=rank != 0,
                file=sys.__stdout__,
            ):
                if args.explicit_prefetching:
                    trainer.model.unshard()
                batch = next(train_dl_iterator)
                trainer.step(batch, eval_dataloader, epoch_num)

            # ----------------------------------
            # Save checkpoint
            # ----------------------------------
            path = f"distill_ckpt/{round_num}_{epoch_num}_{trainer.tr_step}_{trainer.current_eval_loss}_{trainer.min_eval_loss}"
            checkpointer.save(trainer.model, optim, path)

    checkpointer.save(trainer.model, optim)
    torch.distributed.destroy_process_group()
    # Finish all process so no process exit early.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch FSDP2 example")
    parser.add_argument("--explicit-prefetching", action="store_true", default=False)
    parser.add_argument("--mixed-precision", action="store_true", default=False)
    parser.add_argument("--dcp-api", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
