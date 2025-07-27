import argparse
import os, gc, time, sys, pdb
import torch
import atexit
from datetime import datetime
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM

from datasets import load_dataset, Dataset, DatasetDict
import config
from trainer import Trainer, DistillTrainer
from utils import CSVLogger, DistillDataset, evaluate_model, prepare_dataset, format_time_elapsed, get_round_path, is_main_process, main_print, check_batch_shape
from ensemble import ModelEnsemble

from checkpoint import Checkpointer
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
    os.makedirs(config.logit_cache_path, exist_ok=True)

    # ----------------------------------
    # Loading the Teacher Dataset
    # ----------------------------------
    # TODO: to be changed to distributed
    dataClass = DistillDataset('cpu')
    dataset = dataClass.get_dataset() if config.synthetic_data else dataClass.get_teacher_logits()

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

    # ----------------------------------
    # Load Tokenizer (needed for collator & evaluation)
    # ----------------------------------
    main_print("--> Loading Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(config.student_model_name)
    response_template_ids = tokenizer("<|im_start|>assistant\n")["input_ids"]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

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

    existing_models = []
    for run_dir in config.ensemble_members:
        for i in range(config.total_rounds):
            round_dir = os.path.join(run_dir, f"round_{i}")
            model_file = os.path.join(round_dir, "config.json")
            if os.path.exists(model_file):
                existing_models.append((i, round_dir))

    if existing_models:
        raise NotImplementedError
        existing_models.sort(key=lambda x: x[0])
        start_round = max((r for r, _ in existing_models)) + 1
        ensemble_model_names = [path for _, path in existing_models]
        ensemble_model = ModelEnsemble(
            model_names=ensemble_model_names,
            torch_dtype=torch.bfloat16,
            vocab_size=student_model.config.vocab_size,
        ).to(device)
        # TODO: check placement
        ensemble_model.requires_grad_(False)
    else:
        start_round = 0
        ensemble_model = None

    # ----------------------------------
    # Initialize trainer
    # ----------------------------------
    # TODO: move all the init, prepare steps and DS and DL into the class
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optim, factor=1)
    trainer = DistillTrainer(student_model, tokenizer, optim, lr_scheduler, config, ensemble_model)


    # ----------------------------------
    # Outer Training Loop
    # ----------------------------------

    for round_num in range(start_round, config.total_rounds):
        # TODO: Is this epoch?
        round_start_time = time.time()
        round_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        main_print(f"\n{'='*50}")
        main_print(f"--> Starting Round {round_num} at: {round_start_datetime}")
        main_print(f"{'='*50}")
        
        # ----------------------------------
        # Prepare dataset
        # ----------------------------------
        train_dataloader, eval_dataloader = prepare_dataset(dataset['train'], dataset['test'], config, response_template_ids, config.seed + round_num)
        if is_main_process():
            check_batch_shape(train_dataloader)

        round_output_dir = get_round_path(output_path, round_num)
        main_print(f"Round '{round_num}' model stored in: {round_output_dir}")

        trainer.prepare_train()
        train_dl_iterator = iter(train_dataloader)

        # ----------------------------------
        # Inner Training Loop
        # ----------------------------------

        for _ in tqdm(
            range(len(train_dataloader)),
            disable=rank != 0,
            file=sys.__stdout__,
        ):
            if args.explicit_prefetching:
                trainer.model.unshard()
            batch = next(train_dl_iterator)
            trainer.step(batch, eval_dataloader, round_num)

    # ----------------------------------
    # Save checkpoint
    # ----------------------------------
        checkpointer.save(trainer.model, optim)
        
    checkpointer.save(trainer.model, optim)
    torch.distributed.destroy_process_group()
    # Finish all process so no process exit early.


    #     # training_args = config.get_training_args(round_output_dir)

    #     # if is_main_process():
    #     #     training_args.eval_strategy = "steps"
    #     #     training_args.eval_steps = config.eval_steps
    #     #     training_args.eval_on_start = False
    #     #     training_args.logging_strategy = "steps"
    #     #     training_args.logging_steps = config.logging_steps
    #     #     training_args.save_strategy = "steps"
    #     #     training_args.save_steps = config.save_steps
    #     #     training_args.save_total_limit = config.save_total_limit
    #     # else:
    #     #     training_args.eval_strategy = "no"
    #     #     training_args.logging_strategy = "no"
    #     #     training_args.save_strategy = "no"

    #     # trainer = DistillationTrainer(
    #     #     ensemble_model=ensemble_model,
    #     #     logger=logger,
    #     #     round_num=round_num,
    #     #     overall_start_time=overall_start_time,
    #     #     model=student_model,
    #     #     train_dataset=dataset["train"],
    #     #     eval_dataset=dataset["test"],
    #     #     data_collator=collator,
    #     #     args=training_args,
    #     #     callbacks=[LoggingCallback(logger, round_num, overall_start_time)] if is_main_process() else [],
    #     # )
    #     # trainer.train(resume_from_checkpoint=config.checkpoint_path)
    #     logger.flush() if is_main_process() else None
    #     trainer.model.save_pretrained(round_output_dir)

    #     # ----------------------------------
    #     # Add model to ensemble
    #     # ----------------------------------

    #     if ensemble_model is None:
    #         ensemble_model = ModelEnsemble(
    #             [round_output_dir],
    #             torch_dtype=torch.bfloat16,
    #             vocab_size=student_model.vocab_size,
    #         )
    #         ensemble_model.requires_grad_(False)
    #     else:
    #         ensemble_model.add_model(round_output_dir)

    #     # ----------------------------------
    #     # Evaluate and log
    #     # ----------------------------------

    #     if is_main_process():
    #         student_eval_results = evaluate_model(trainer.model, dataset["test"], collator)
    #         ensemble_eval_results = evaluate_model(ensemble_model, dataset["test"], collator)

    #         main_print(f"\n{'-'*25}")
    #         main_print(f"Student evaluation for {round_num}: {student_eval_results['eval_loss']}")
    #         main_print(f"Ensemble evaluation for {round_num}: {ensemble_eval_results['eval_loss']}")
    #         main_print(f"Teacher evaluation for {round_num}: {teacher_eval_results[0]}")
    #         main_print(f"{'-'*25}")

    #     round_end_time = time.time()
    #     round_duration = round_end_time - round_start_time
    #     overall_elapsed = round_end_time - overall_start_time
    #     round_duration_str = format_time_elapsed(round_duration)
    #     overall_elapsed_str = format_time_elapsed(overall_elapsed)
    #     round_end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     main_print(f"{'='*50}")
    #     main_print(f"Ending Round {round_num} at: {round_end_datetime}")
    #     main_print(f"Completed in: {round_duration_str}")
    #     main_print(f"Total training time: {overall_elapsed_str}")
    #     main_print(f"{'='*50}\n")

    #     if is_main_process():
    #         logger.log(
    #             function="main",
    #             round_num=round_num,
    #             phase="custom_eval",
    #             role="ensemble",
    #             eval_loss=ensemble_eval_results["eval_loss"],
    #             perplexity=ensemble_eval_results["perplexity"],
    #             round_duration=round_duration,
    #             overall_elapsed=overall_elapsed,
    #         )
    #         logger.log(
    #             function="main",
    #             round_num=round_num,
    #             phase="custom_eval",
    #             role="student",
    #             eval_loss=student_eval_results["eval_loss"],
    #             perplexity=student_eval_results["perplexity"],
    #             round_duration=round_duration,
    #             overall_elapsed=overall_elapsed,
    #         )
    #         logger.log(
    #             function="main",
    #             round_num=round_num,
    #             phase="custom_eval",
    #             role="teacher",
    #             eval_loss=teacher_eval_results[0],
    #             perplexity=teacher_eval_results[1],
    #             round_duration=round_duration,
    #             overall_elapsed=overall_elapsed,
    #         )

    #         logger.flush()

    #     # ----------------------------------
    #     # Reset memory
    #     # ----------------------------------

    #     del student_model, trainer
    #     gc.collect()
    #     torch.cuda.empty_cache()

    #     dist.barrier() if config.ddp else None

    #     # ----------------------------------
    #     # Load Student
    #     # ----------------------------------

    #     student_model = AutoModelForCausalLM.from_pretrained(
    #         config.student_model_name,
    #         torch_dtype=torch.bfloat16,
    #     ).to(device)

    #     if is_main_process():
    #         student_eval_results = evaluate_model(student_model, dataset["test"], collator)
    #         logger.log(
    #             function="main",
    #             round_num=round_num,
    #             phase="custom_eval",
    #             role="student",
    #             eval_loss=student_eval_results["eval_loss"],
    #             perplexity=student_eval_results["perplexity"],
    #             tags=["initial eval"],
    #         )

    # # ----------------------------------
    # # End round
    # # ----------------------------------

    # overall_end_time = time.time()
    # overall_duration = overall_end_time - overall_start_time
    # overall_duration_str = format_time_elapsed(overall_duration)
    # end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # main_print(f"\n{'='*50}")
    # main_print(f"Training completed at: {end_datetime}")
    # main_print(f"Total training time: {overall_duration_str}")
    # main_print(f"{'='*50}")

    # dist.barrier() if config.ddp else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch FSDP2 example")
    parser.add_argument("--explicit-prefetching", action="store_true", default=False)
    parser.add_argument("--mixed-precision", action="store_true", default=False)
    parser.add_argument("--dcp-api", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
