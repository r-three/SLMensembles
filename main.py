import os, gc, time, sys, pdb
import torch
import atexit
from datetime import datetime
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM

from datasets import load_dataset, Dataset, DatasetDict
import config
from train import DistillationTrainer, LoggingCallback
from utils import CSVLogger, DistillDataset, evaluate_model, format_time_elapsed, get_round_path, is_main_process, main_print
from ensemble import ModelEnsemble


def main():
    overall_start_time = time.time()
    overall_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    main_print(f"--> Starting training at: {overall_start_datetime}\n")

    # ----------------------------------
    # Device Setup
    # ----------------------------------

    default_local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if config.ddp and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if default_local_rank >= num_gpus:
            raise RuntimeError(
                f"LOCAL_RANK={default_local_rank} but only {num_gpus} CUDA devices are available."
            )
        torch.cuda.set_device(default_local_rank)
        device = torch.device(f"cuda:{default_local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_print(f"Using device: {device}")

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
    dataClass = DistillDataset(device)
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

    student_model = AutoModelForCausalLM.from_pretrained(
        config.student_model_name,
        torch_dtype=torch.bfloat16,
    ).to(device)

    # ----------------------------------
    # Load Existing Models
    # ----------------------------------

    existing_models = []
    for run_dir in config.ensemble_members:
        for i in range(config.total_rounds):
            round_dir = os.path.join(run_dir, f"round_{i}")
            model_file = os.path.join(round_dir, "config.json")
            if os.path.exists(model_file):
                existing_models.append((i, round_dir))

    if existing_models:
        existing_models.sort(key=lambda x: x[0])
        start_round = max((r for r, _ in existing_models)) + 1
        ensemble_model_names = [path for _, path in existing_models]
        ensemble_model = ModelEnsemble(
            model_names=ensemble_model_names,
            torch_dtype=torch.bfloat16,
            vocab_size=student_model.config.vocab_size,
        ).to(device)
        ensemble_model.requires_grad_(False)
    else:
        start_round = 0
        ensemble_model = None

    # ----------------------------------
    # Evaluate
    # ----------------------------------

    # if is_main_process():
        # student_eval_results = evaluate_model(student_model, dataset["test"], collator)
        # logger.log(
        #     function="main",
        #     round_num=0,
        #     phase="custom_eval",
        #     role="student",
        #     eval_loss=student_eval_results["eval_loss"],
        #     perplexity=student_eval_results["perplexity"],
        #     tags=["initial eval"],
        # )
        # teacher_eval_results = config.teacher_eval
        # logger.log(
        #     function="main",
        #     round_num=0,
        #     phase="custom_eval",
        #     role="teacher",
        #     eval_loss=teacher_eval_results[0],
        #     perplexity=teacher_eval_results[1],
        # )

    # ----------------------------------
    # Load checkpoint
    # ----------------------------------

    if config.checkpoint_path:
        if not os.path.exists(config.checkpoint_path):
            main_print(f"[ERROR] Checkpointed model does not exist at: {config.checkpoint_path}")
            sys.exit(1)
        main_print(f"Resuming training from checkpoint: {config.checkpoint_path}")

    # ----------------------------------
    # Outer Training Loop
    # ----------------------------------

    for round_num in range(start_round, config.total_rounds):
        round_start_time = time.time()
        round_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        main_print(f"\n{'='*50}")
        main_print(f"--> Starting Round {round_num} at: {round_start_datetime}")
        main_print(f"{'='*50}")

        dataset["train"] = dataset["train"].shuffle(seed=config.seed + round_num)
        round_output_dir = get_round_path(output_path, round_num)
        main_print(f"Round '{round_num}' model stored in: {round_output_dir}")

        # ----------------------------------
        # Inner Training Loop
        # ----------------------------------

        training_args = config.get_training_args(round_output_dir)

        if is_main_process():
            training_args.eval_strategy = "steps"
            training_args.eval_steps = config.eval_steps
            training_args.eval_on_start = False
            training_args.logging_strategy = "steps"
            training_args.logging_steps = config.logging_steps
            training_args.save_strategy = "steps"
            training_args.save_steps = config.save_steps
            training_args.save_total_limit = config.save_total_limit
        else:
            training_args.eval_strategy = "no"
            training_args.logging_strategy = "no"
            training_args.save_strategy = "no"

        trainer = DistillationTrainer(
            ensemble_model=ensemble_model,
            logger=logger,
            round_num=round_num,
            overall_start_time=overall_start_time,
            model=student_model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=collator,
            args=training_args,
            callbacks=[LoggingCallback(logger, round_num, overall_start_time)] if is_main_process() else [],
        )

        trainer.train(resume_from_checkpoint=config.checkpoint_path)
        logger.flush() if is_main_process() else None
        trainer.model.save_pretrained(round_output_dir)

        # ----------------------------------
        # Add model to ensemble
        # ----------------------------------

        if ensemble_model is None:
            ensemble_model = ModelEnsemble(
                [round_output_dir],
                torch_dtype=torch.bfloat16,
                vocab_size=student_model.vocab_size,
            )
            ensemble_model.requires_grad_(False)
        else:
            ensemble_model.add_model(round_output_dir)

        # ----------------------------------
        # Evaluate and log
        # ----------------------------------

        if is_main_process():
            student_eval_results = evaluate_model(trainer.model, dataset["test"], collator)
            ensemble_eval_results = evaluate_model(ensemble_model, dataset["test"], collator)

            main_print(f"\n{'-'*25}")
            main_print(f"Student evaluation for {round_num}: {student_eval_results['eval_loss']}")
            main_print(f"Ensemble evaluation for {round_num}: {ensemble_eval_results['eval_loss']}")
            main_print(f"Teacher evaluation for {round_num}: {teacher_eval_results[0]}")
            main_print(f"{'-'*25}")

        round_end_time = time.time()
        round_duration = round_end_time - round_start_time
        overall_elapsed = round_end_time - overall_start_time
        round_duration_str = format_time_elapsed(round_duration)
        overall_elapsed_str = format_time_elapsed(overall_elapsed)
        round_end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        main_print(f"{'='*50}")
        main_print(f"Ending Round {round_num} at: {round_end_datetime}")
        main_print(f"Completed in: {round_duration_str}")
        main_print(f"Total training time: {overall_elapsed_str}")
        main_print(f"{'='*50}\n")

        if is_main_process():
            logger.log(
                function="main",
                round_num=round_num,
                phase="custom_eval",
                role="ensemble",
                eval_loss=ensemble_eval_results["eval_loss"],
                perplexity=ensemble_eval_results["perplexity"],
                round_duration=round_duration,
                overall_elapsed=overall_elapsed,
            )
            logger.log(
                function="main",
                round_num=round_num,
                phase="custom_eval",
                role="student",
                eval_loss=student_eval_results["eval_loss"],
                perplexity=student_eval_results["perplexity"],
                round_duration=round_duration,
                overall_elapsed=overall_elapsed,
            )
            logger.log(
                function="main",
                round_num=round_num,
                phase="custom_eval",
                role="teacher",
                eval_loss=teacher_eval_results[0],
                perplexity=teacher_eval_results[1],
                round_duration=round_duration,
                overall_elapsed=overall_elapsed,
            )

            logger.flush()

        # ----------------------------------
        # Reset memory
        # ----------------------------------

        del student_model, trainer
        gc.collect()
        torch.cuda.empty_cache()

        dist.barrier() if config.ddp else None

        # ----------------------------------
        # Load Student
        # ----------------------------------

        student_model = AutoModelForCausalLM.from_pretrained(
            config.student_model_name,
            torch_dtype=torch.bfloat16,
        ).to(device)

        if is_main_process():
            student_eval_results = evaluate_model(student_model, dataset["test"], collator)
            logger.log(
                function="main",
                round_num=round_num,
                phase="custom_eval",
                role="student",
                eval_loss=student_eval_results["eval_loss"],
                perplexity=student_eval_results["perplexity"],
                tags=["initial eval"],
            )

    # ----------------------------------
    # End round
    # ----------------------------------

    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    overall_duration_str = format_time_elapsed(overall_duration)
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    main_print(f"\n{'='*50}")
    main_print(f"Training completed at: {end_datetime}")
    main_print(f"Total training time: {overall_duration_str}")
    main_print(f"{'='*50}")

    dist.barrier() if config.ddp else None


if __name__ == "__main__":
    main()
