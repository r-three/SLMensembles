import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
import numpy as np

from ensemble import ModelEnsemble


# Model and dataset setup
seed = 42
dataset_path = "/scratch/ssd004/scratch/nkandpa2/slm_ensembles/tulu-3-sft-mixture-pretokenized"
output_path = "/scratch/ssd004/scratch/nkandpa2/slm_ensembles/boosted_distillation_1.5B_teacher_average_fixed_logging"
teacher_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
student_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
ensemble_model_names = []

tokenizer = AutoTokenizer.from_pretrained(student_model_name)
student_model = AutoModelForCausalLM.from_pretrained(student_model_name, torch_dtype=torch.bfloat16, device_map="cuda:0")

teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name, torch_dtype=torch.bfloat16, device_map="cuda:1")
teacher_model.resize_token_embeddings(new_num_tokens=student_model.vocab_size)
teacher_model.requires_grad_(False)

if len(ensemble_model_names) > 0:
    ensemble_model = ModelEnsemble(ensemble_model_names, torch_dtype=torch.bfloat16, device_map="cuda:1", vocab_size=student_model.vocab_size)
    ensemble_model.requires_grad_(False)
else:
    ensemble_model = None
 

dataset = datasets.load_from_disk(dataset_path)
response_template_ids = tokenizer("<|im_start|>assistant\n")["input_ids"]
collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)


class DistillationTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.training_round = kwargs.pop("training_round")
        self.steps_per_round = kwargs.pop("steps_per_round")
        super().__init__(*args, **kwargs)

    def compute_kl_loss(self, student_logits, ensemble_logits, teacher_logits, mask, temperature=1.0):
        """Computes KL divergence loss between teacher and student model logits."""
        teacher_logits = teacher_logits.to(student_logits.device)
        if ensemble_logits is not None:
            num_models = len(ensemble_model.models)
            ensemble_logits = ensemble_logits.to(student_logits.device)
            student_logits = student_logits/(num_models + 1) + ensemble_logits*(num_models/(num_models + 1))
        student_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
        forward_kl = F.kl_div(student_probs, teacher_probs, log_target=True, reduction="none").sum(-1)
        return forward_kl[mask].mean()
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Custom loss function for knowledge distillation."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        
        with torch.no_grad():
            teacher_logits = teacher_model(input_ids.to(teacher_model.device), attention_mask=attention_mask.detach().to(teacher_model.device)).logits
            if ensemble_model is not None:
                ensemble_logits = ensemble_model(input_ids.to(ensemble_model.device), attention_mask=attention_mask.detach().to(ensemble_model.device)).logits
            else:
                ensemble_logits = None
        
        student_logits = model(input_ids, attention_mask=attention_mask).logits
        loss = self.compute_kl_loss(student_logits, ensemble_logits, teacher_logits, labels != -100)
        return (loss, student_outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            student_logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
            if ensemble_model is not None:
                num_models = len(ensemble_model.models)
                ensemble_logits = ensemble_model(input_ids=inputs["input_ids"].to(ensemble_model.device), attention_mask=inputs["attention_mask"].to(ensemble_model.device)).logits.to(student_logits.device)
                student_logits = (student_logits/(num_models + 1) + ensemble_logits*(num_models/(num_models+1))).detach()
            loss = model.module.loss_function(logits=student_logits, labels=inputs["labels"], vocab_size=model.module.config.vocab_size).detach()
            return (loss, None, None) if prediction_loss_only else (loss, student_logits, inputs["labels"])
    
    def log(self, logs, start_time=None):
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            if start_time is not None:
                speed_metrics("train", start_time, num_tokens=self.state.num_input_tokens_seen)

        output = {**logs, **{"step": self.state.global_step + (self.training_round*self.steps_per_round)}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)


steps_per_round = 1000
for training_round in range(0,6):
    dataset["train"] = dataset["train"].shuffle(seed=seed+training_round)
    training_args = SFTConfig(
        output_dir=os.path.join(output_path, f"round_{training_round}"),
        overwrite_output_dir=False,
        report_to="wandb",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,
        warmup_steps=50,
        bf16=True,
        max_steps=steps_per_round,
        eval_strategy="steps",
        eval_steps=100,
        per_device_eval_batch_size=4,
        eval_on_start=True,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="no",
        save_total_limit=0
    )
    trainer = DistillationTrainer(
        training_round=training_round,
        steps_per_round=steps_per_round,
        model=student_model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collator,
        args=training_args,
        tokenizer=tokenizer,
    )
    trainer.train()

    student_model.save_pretrained(os.path.join(output_path, f"round_{training_round}"))
    if ensemble_model is None:
        ensemble_model = ModelEnsemble([os.path.join(output_path, f"round_{training_round}")], torch_dtype=torch.bfloat16, device_map="cuda:1", vocab_size=student_model.vocab_size)
        ensemble_model.requires_grad_(False)
    else:
        ensemble_model.add_model(os.path.join(output_path, f"round_{training_round}"))
    
    del student_model
    student_model = AutoModelForCausalLM.from_pretrained(student_model_name, torch_dtype=torch.bfloat16, device_map="cuda:0")
    student_model.resize_token_embeddings(new_num_tokens=student_model.vocab_size)
