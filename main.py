import os, gc, time
import torch
import datasets
import atexit
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM

import config
from train import DistillationTrainer, LoggingCallback
from utils import CSVLogger, evaluate_model, format_time_elapsed, get_round_path
from ensemble import ModelEnsemble


def main():
    overall_start_time = time.time()
    overall_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nStarting training at: {overall_start_datetime}")

    log_dir = config.get_directory(config.log_dir)
    logger = CSVLogger(
        log_dir, fieldnames=config.CSV_COLUMNS, overall_start_time=overall_start_time
    )

    atexit.register(logger.flush)

    output_path = config.get_directory(config.base_output_dir)
    run_name = f"{os.path.basename(output_path)}"

    # ----------------------------------
    # Metrics
    # ----------------------------------

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
    print("\n==== RUN CONFIGURATION ====")

    print(f"Run: {run_name}")
    print(f"Created logging directory: {log_dir}")
    print(f"Models stored in: {output_path}\n")

    print(f"\n{config.id_string}")
    print(f"{config.description}\n")

    for k, v in metadata_dict.items():
        print(f"{k}: {v}")

    print("===========================\n")

    logger.log(
        function="main",
        phase="none",
        round_num=0,
        metadata=metadata_dict,
    )

    # ----------------------------------
    # Load Tokenizer and Models
    # ----------------------------------

    tokenizer = AutoTokenizer.from_pretrained(config.student_model_name)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        config.teacher_model_name,
        torch_dtype=torch.bfloat16,
        device_map=config.teacher_device,
    )
    student_model = AutoModelForCausalLM.from_pretrained(
        config.student_model_name, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    teacher_model.resize_token_embeddings(new_num_tokens=student_model.vocab_size)
    del student_model
    teacher_model.requires_grad_(False)

    # ----------------------------------
    # Load dataset and evaluate
    # ----------------------------------

    dataset = config.get_dataset()
    response_template_ids = tokenizer("<|im_start|>assistant\n")["input_ids"]
    collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )

    teacher_eval_results = evaluate_model(teacher_model, dataset["test"], collator)
    logger.log(
        function="main",
        round_num=0,
        phase="custom_eval",
        role="teacher",
        eval_loss=teacher_eval_results["eval_loss"],
        perplexity=teacher_eval_results["perplexity"],
    )

    # ----------------------------------
    # Load Existing Ensemble Models
    # ----------------------------------

    existing_models = []
    for run_dir in config.past_run_dirs:
        for i in range(config.total_rounds):
            round_dir = os.path.join(run_dir, f"round_{i}")
            model_file = os.path.join(round_dir, "config.json")
            if os.path.exists(model_file):
                existing_models.append((i, round_dir))

    existing_models.sort(key=lambda x: x[0])

    start_round = max((r for r, _ in existing_models), default=-1) + 1
    ensemble_model_names = [path for _, path in existing_models]
    ensemble_model = None

    if ensemble_model_names:
        print(f"Resuming from ensemble with {len(ensemble_model_names)} models")
        temp_model = AutoModelForCausalLM.from_pretrained(
            config.student_model_name,
            torch_dtype=torch.bfloat16,
            device_map=config.student_device,
        )
        ensemble_model = ModelEnsemble(
            model_names=ensemble_model_names,
            torch_dtype=torch.bfloat16,
            device_map=config.student_device,
            vocab_size=temp_model.vocab_size,
        )
        ensemble_model.requires_grad_(False)
        del temp_model

    # ----------------------------------
    # Outer Training Loop
    # ----------------------------------

    for round_num in range(start_round, config.total_rounds):
        round_start_time = time.time()
        round_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n{'='*50}")
        print(f"Starting Round {round_num} at: {round_start_datetime}")
        print(f"{'='*50}")

        dataset["train"] = dataset["train"].shuffle(seed=config.seed + round_num)
        round_output_dir = get_round_path(output_path, round_num)
        print(f"Round '{round_num}' model stored in: {round_output_dir}")

        # ----------------------------------
        # Load Student
        # ----------------------------------

        student_model = AutoModelForCausalLM.from_pretrained(
            config.student_model_name,
            torch_dtype=torch.bfloat16,
            device_map=config.student_device,
        )
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
        # Inner Training Loop
        # ----------------------------------

        training_args = config.get_training_args(round_output_dir)
        trainer = DistillationTrainer(
            teacher_model=teacher_model,
            ensemble_model=ensemble_model,
            student_model=student_model,
            logger=logger,
            round_num=round_num,
            overall_start_time=overall_start_time,
            model=student_model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=collator,
            args=training_args,
            callbacks=[LoggingCallback(logger, round_num, overall_start_time)],
        )

        trainer.train()
        logger.flush()
        trainer.model.save_pretrained(round_output_dir)

        # ----------------------------------
        # Add model to ensemble
        # ----------------------------------

        if ensemble_model is None:
            ensemble_model = ModelEnsemble(
                [round_output_dir],
                torch_dtype=torch.bfloat16,
                device_map=config.student_device,
                vocab_size=student_model.vocab_size,
            )
            ensemble_model.requires_grad_(False)
        else:
            ensemble_model.add_model(round_output_dir)

        # ----------------------------------
        # Evaluate and log
        # ----------------------------------

        student_eval_results = evaluate_model(trainer.model, dataset["test"], collator)
        ensemble_eval_results = evaluate_model(
            ensemble_model, dataset["test"], collator
        )

        print(f"\n{'-'*25}")
        print(
            f"Student evaluation for {round_num}: {student_eval_results['eval_loss']}"
        )
        print(
            f"Ensemble evaluation for {round_num}: {ensemble_eval_results['eval_loss']}"
        )
        print(
            f"Teacher evaluation for {round_num}: {teacher_eval_results['eval_loss']}"
        )
        print(f"{'-'*25}")

        round_end_time = time.time()
        round_duration = round_end_time - round_start_time
        overall_elapsed = round_end_time - overall_start_time
        round_duration_str = format_time_elapsed(round_duration)
        overall_elapsed_str = format_time_elapsed(overall_elapsed)
        round_end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{'='*50}")
        print(f"Ending Round {round_num} at: {round_end_datetime}")
        print(f"Completed in: {round_duration_str}")
        print(f"Total training time: {overall_elapsed_str}")
        print(f"{'='*50}\n")

        logger.log(
            function="main",
            round_num=round_num,
            phase="custom_eval",
            role="ensemble",
            eval_loss=ensemble_eval_results["eval_loss"],
            perplexity=ensemble_eval_results["perplexity"],
            round_duration=round_duration,
        )
        logger.log(
            function="main",
            round_num=round_num,
            phase="custom_eval",
            role="student",
            eval_loss=student_eval_results["eval_loss"],
            perplexity=student_eval_results["perplexity"],
            round_duration=round_duration,
        )
        logger.log(
            function="main",
            round_num=round_num,
            phase="custom_eval",
            role="teacher",
            eval_loss=teacher_eval_results["eval_loss"],
            perplexity=teacher_eval_results["perplexity"],
            round_duration=round_duration,
        )

        logger.flush()

        # ----------------------------------
        # Reset student model
        # ----------------------------------

        del student_model
        gc.collect()
        torch.cuda.set_device(config.student_device)
        torch.cuda.empty_cache()

    # ----------------------------------
    # End round
    # ----------------------------------

    # Record overall end time
    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time
    overall_duration_str = format_time_elapsed(overall_duration)
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*50}")
    print(f"Training completed at: {end_datetime}")
    print(f"Total training time: {overall_duration_str}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
