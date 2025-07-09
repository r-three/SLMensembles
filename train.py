import torch
import torch.nn.functional as F
import numpy as np
from transformers import TrainerCallback
from trl import SFTTrainer
import config


class LoggingCallback(TrainerCallback):
    def __init__(self, logger, round_num, overall_start_time):
        self.logger = logger
        self.round_num = round_num
        self.overall_start_time = overall_start_time

    def on_prediction_step_end(self, args, state, control, **kwargs):
        loss = kwargs.get("loss", None)
        if loss is not None:
            self.logger.log(
                function="on_prediction_step_end",
                round_num=self.round_num,
                phase="eval",
                role="student",
                step=state.global_step,
                eval_loss=loss.mean().item(),
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        control = super().on_log(args, state, control, logs=logs, **kwargs)

        self.logger.log(
            function="on_log",
            round_num=self.round_num,
            phase="train",
            role="student",
            step=state.global_step,
            train_loss=logs.get("loss"),
            learning_rate=logs.get("learning_rate"),
            grad_norm=logs.get("grad_norm"),
        )

        return control


class DistillationTrainer(SFTTrainer):
    def __init__(self, ensemble_model, teacher_logits, logger, round_num, overall_start_time, *args, **kwargs):
        self.ensemble_model = ensemble_model
        self.teacher_logits = teacher_logits
        self.logger = logger
        self.round_num = round_num
        self.overall_start_time = overall_start_time
        self.extra_logging_info = {"kl_losses": []}

        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        # -------------------------
        # Compute Student Predictions
        # -------------------------
        student_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        student_logits = student_outputs.logits
        next_token_loss = student_outputs.loss

        # -------------------------
        # Compute Ensemble Predictions
        # -------------------------
        ensemble_logits = None
        if self.ensemble_model is not None:
            with torch.no_grad():
                ensemble_outputs = self.ensemble_model(input_ids=input_ids, attention_mask=attention_mask)
                ensemble_logits = ensemble_outputs.logits

        alpha = config.alpha if not config.synthetic_data else 1
        kl_loss = 0
        if not config.synthetic_data:
            kl_loss = self.compute_kl_loss(student_logits, ensemble_logits, mask=labels != -100)
        hybrid_loss = (1 - alpha) * kl_loss + alpha * next_token_loss

        # -------------------------
        # Log
        # -------------------------
        if self.state.global_step % self.args.logging_steps == 0:
            self.logger.log(
                function="compute_loss",
                round_num=self.round_num,
                phase="train",
                role="student",
                step=self.state.global_step,
                train_loss=hybrid_loss.item(),
                train_next_token_loss=next_token_loss.item(),
                train_kl_loss=kl_loss.item(),
                alpha=alpha,
                learning_rate=self._get_learning_rate(),
            )

        return (hybrid_loss, current_model_logits) if return_outputs else hybrid_loss

    def compute_kl_loss(self, student_logits, ensemble_logits, mask, temperature=1.0):
        """Computes KL divergence loss between teacher and student model logits."""

        # ----------------------------------------
        # Combine model predictions with ensemble
        # ----------------------------------------
        if ensemble_logits is not None:
            num_models = len(self.ensemble_model.models)
            student_logits = student_logits / (num_models + 1) + ensemble_logits * (num_models / (num_models + 1))

        # -----------------------
        # Compute KL Loss
        # -----------------------
        student_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.log_softmax(self.teacher_logits / temperature, dim=-1)
        kl_loss = F.kl_div(student_probs, teacher_probs, log_target=True, reduction="none").sum(-1)
        return kl_loss[mask].mean()

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        # -------------------------
        # Compute Predictions
        # -------------------------
        with torch.no_grad():
            student_logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).logits
            ensemble_logits = None
            if self.ensemble_model is not None:
                ensemble_logits = self.ensemble_model(input_ids=input_ids, attention_mask=attention_mask).logits

            if ensemble_logits is not None:
                num_models = len(self.ensemble_model.models)
                total_ensemble_logits = student_logits / (num_models + 1) + ensemble_logits * (num_models / (num_models + 1))
            else:
                total_ensemble_logits = student_logits

        # ------------------------------
        # Handle potential DDP wrapping
        # ------------------------------
        if hasattr(model, "module"):
            model = model.module

        # ------------------------------
        # Compute Loss
        # ------------------------------
        loss = model.loss_function(
            logits=total_ensemble_logits,
            labels=labels,
            vocab_size=model.config.vocab_size,
        )

        kl_loss = 0
        if not config.synthetic_data:
            kl_loss = self.compute_kl_loss(student_logits, ensemble_logits, mask=labels_s != -100)
            self.extra_logging_info.setdefault("kl_losses", []).append(kl_loss.item())

        # ------------------------------
        # Log
        # ------------------------------
        if self.state.global_step % self.args.logging_steps == 0:
            self.logger.log(
                function="prediction_step",
                round_num=self.round_num,
                phase="eval",
                role="student",
                step=self.state.global_step,
                eval_loss=loss.item(),
                eval_kl_loss=kl_loss.item(),
            )

        return (loss, None, None) if prediction_loss_only else (loss, student_logits, labels)

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

        self.logger.log(
            function="evaluation_loop",
            round_num=self.round_num,
            phase="eval",
            role="student",
            step=self.state.global_step,
            eval_loss=output.metrics["eval_loss"],
            eval_kl_loss=np.mean(self.extra_logging_info["kl_losses"]),
        )
        self.extra_logging_info = {"kl_losses": []}
        return output
