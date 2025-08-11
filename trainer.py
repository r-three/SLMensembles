import torch
import torch.nn.functional as F
import numpy as np
import pdb
from transformers import TrainerCallback
from trl import SFTTrainer
import config
from abc import ABC, abstractmethod
import csv
import sys
import torch.distributed as dist
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from utils import main_print
from tqdm.auto import tqdm
from datetime import datetime

# ---------------------- High-Level Implementation ----------------------

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
    def __init__(self, ensemble_model, logger, round_num, overall_start_time, *args, **kwargs):
        self.ensemble_model = ensemble_model
        self.logger = logger
        self.round_num = round_num
        self.overall_start_time = overall_start_time
        self.extra_logging_info = {"kl_losses": []}

        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        # -------------------------
        # Compute Student Predictions
        # -------------------------
        student_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = student_outputs.logits

        if hasattr(model, "module"):
            model = model.module

        next_token_loss = model.loss_function(
            logits=student_logits,
            labels=labels,
            vocab_size=config.student_vocab_size,
        )

        # ----------------------------
        # Compute Ensemble Predictions
        # ----------------------------
        ensemble_logits = None
        if self.ensemble_model is not None:
            with torch.no_grad():
                ensemble_outputs = self.ensemble_model(input_ids=input_ids, attention_mask=attention_mask)
                ensemble_logits = ensemble_outputs.logits

        # -------------------------
        # Compute Loss
        # -------------------------
        alpha = config.alpha if not config.synthetic_data else 1
        kl_loss = 0
        if not config.synthetic_data:
            kl_loss = self.compute_kl_loss(student_logits, ensemble_logits, mask=labels != -100, inputs=inputs)
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

        return (hybrid_loss, student_logits) if return_outputs else hybrid_loss

    def compute_kl_loss(self, student_logits, ensemble_logits, mask, inputs, temperature=1.0):
        teacher_logit_indices = inputs["logit_indices"]
        teacher_logit_values = inputs["logit_values"]

        # ----------------------------------------
        # Combine model predictions with ensemble
        # ----------------------------------------
        if ensemble_logits is not None:
            num_models = len(self.ensemble_model.models)
            student_logits = student_logits / (num_models + 1) + ensemble_logits * (num_models / (num_models + 1))

        # ------------------------------
        # Reconstruct the teacher logits
        # ------------------------------
        batch_size, seq_len, vocab_size = student_logits.shape  # [8, 1024, 151936]
        teacher_logits = torch.full((batch_size, seq_len, vocab_size), -1e8, device=student_logits.device)
        teacher_logits.scatter_(-1, teacher_logit_indices, teacher_logit_values)

        # -----------------------
        # Compute KL Loss
        # -----------------------
        student_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.log_softmax(teacher_logits / temperature, dim=-1)
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
        if config.ddp and hasattr(model, "module"):
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
            kl_loss = self.compute_kl_loss(student_logits, ensemble_logits, mask=labels != -100, inputs=inputs)
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


# ---------------------- Manual Implementation ----------------------

def _gather(x: torch.Tensor) -> torch.Tensor:
    output_tensors = [x.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, x)
    return torch.cat(output_tensors, dim=0)

class Trainer(ABC):
    def __init__(
        self,
        model,
        optim,
        lr_scheduler,
        config,
        logger=None,
        round_num=0,
        overall_start_time=None,
    ) -> None:
        self.model = model
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.logger = logger
        self.round_num = round_num
        self.overall_start_time = overall_start_time
        self.processed_id = 0
        self.gad = 0    # gradient accumulated incremented before fwd pass.
        self.gas = config.gradient_accumulation_steps
        self.tr_step = 0    # Need to update when read ckpt
        self.rank = dist.get_rank()
        self.min_eval_loss = 1e12
        self.current_eval_loss = 1e12

    def prepare_train(self):
        if dist.get_rank() == 0:
            with open("results.csv", "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Initialized training", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                writer.writerow(["Train step (local)", "Mean eval loss", "Next token loss", "KL loss", "Valid count"]) 
        self.model.train()
        self.model.config.use_cache = False  # avoid cache warnings in training

    def compute_loss(self, batch):
        '''
        reduce loss is sum
        this ensures that we weight all tokens in the dataset equally,
        rather than weighting each overall example equally when
        using high amounts of gradient accumulation.
        this can result in > 5 point improvements in AlpacaEval
        see https://github.com/huggingface/transformers/issues/24725 for
        more discussion and details.
        https://unsloth.ai/blog/gradient

        Return the valid count (num of valid tokens), which is the ratio between mean and sum for each batch.
        '''

        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        labels = batch.pop('labels')
        outputs = self.model(**batch)
        logits = outputs.logits
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
        shift_logits = shift_logits.view(-1, embedding_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        ignore_index = getattr(self.config, "ignore_index", -100)
        valid_mask = shift_labels.ne(ignore_index)
        valid_count = valid_mask.sum()
        return loss, None, None, valid_count

    def step(self, train_batch, eval_dl, epoch):
        train_loss = self.train_step(train_batch, epoch)

        test_loss = None
        if self.tr_step % self.config.logging_steps == 0:
            test_loss = self.eval_step(eval_dl, epoch)
        self.tr_step += 1
        return train_loss, test_loss
    
    def train_step(self, batch, epoch):
        self.model.train()
        batch["input_ids"] = batch["input_ids"].type(torch.LongTensor)
        batch["labels"] = batch["labels"].type(torch.LongTensor)

        self.gad += 1
        self.processed_id += 1
        # TODO: change to proper dataset ckpt

        if (self.tr_step + 1) % self.gas != self.gas - 1:
            # no need to sync while accumulating gradients
            self.model.set_requires_gradient_sync(False) # with (grad = False):
            tr_step_loss, _, _, _ = self.compute_loss(batch)
            (tr_step_loss / self.gas).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)
            self.model.set_requires_gradient_sync(True)
        else:
            # next forward / backward pass will be synced
            self.model.set_requires_gradient_sync(True)
            dist.barrier()
            tr_step_loss, _, _, _ = self.compute_loss(batch)
            (tr_step_loss / self.gas).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)
            self.optim.step()
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                self.lr_scheduler.step(self.metric)
            else:
                self.lr_scheduler.step()
            self.optim.zero_grad()
        gathered_tr_step_loss = _gather(tr_step_loss.reshape(1)).mean().item()
        # gather with sum

        # Log training steps if logger is available
        if self.logger is not None and self.rank == 0:
            self.logger.log(
                function="train_step",
                round_num=self.round_num,
                phase="train",
                role="student",
                step=self.tr_step,
                train_loss=gathered_tr_step_loss,
                learning_rate=self.lr_scheduler.get_last_lr()[0] if hasattr(self.lr_scheduler, 'get_last_lr') else None,
            )

        return gathered_tr_step_loss
    
    def eval_step(self, eval_dl, epoch: int) -> float:
        main_print("Evaluating")
        self.model.eval()
        eval_loss = torch.tensor(0.0).to(torch.cuda.current_device())
        nxt_token_loss = torch.tensor(0.0).to(torch.cuda.current_device())
        kl_loss = torch.tensor(0.0).to(torch.cuda.current_device())
        valid_total = torch.tensor(0).to(torch.cuda.current_device())
        for _, batch in enumerate(tqdm(eval_dl,
                                  disable=self.rank != 0,
                                  file=sys.__stdout__,)):
            with torch.no_grad():
                batch["input_ids"] = batch["input_ids"].type(torch.LongTensor)
                batch["labels"] = batch["labels"].type(torch.LongTensor)
                loss_sum, next_token_sum, kl_sum, valid_cnt = self.compute_loss(batch)
                eval_loss += loss_sum
                nxt_token_loss += next_token_sum
                kl_loss += kl_sum
                valid_total += valid_cnt
        
        # So you don't see eval loss of a few million
        gathered_eval_loss = _gather(eval_loss.reshape(1)).sum().item()
        gathered_nxt_token_loss = _gather(nxt_token_loss.reshape(1)).sum().item()
        gathered_kl_loss = _gather(kl_loss.reshape(1)).sum().item()
        gathered_valid_total = _gather(valid_total.reshape(1)).sum().item()
        # Take the average of eval_loss on both cards, 
        mean_eval_loss = gathered_eval_loss / gathered_valid_total
        mean_nk_loss = gathered_nxt_token_loss / gathered_valid_total
        mean_kl_loss = gathered_kl_loss / gathered_valid_total

        self.metric = gathered_eval_loss

        main_print(f"Step: {self.tr_step}, eval loss: {mean_eval_loss}")

        if dist.get_rank() == 0:
            with open("results.csv", "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([self.tr_step, mean_eval_loss, mean_nk_loss, mean_kl_loss, gathered_valid_total])
            
            # Log evaluation results if logger is available
            if self.logger is not None:
                self.logger.log(
                    function="eval_step",
                    round_num=self.round_num,
                    phase="eval",
                    role="student",
                    step=self.tr_step,
                    eval_loss=mean_eval_loss,
                    eval_next_token_loss=mean_nk_loss,
                    eval_kl_loss=mean_kl_loss,
                )

        self.min_eval_loss = min(mean_eval_loss, self.min_eval_loss)
        self.current_eval_loss = mean_eval_loss
        self.model.train()
        return gathered_eval_loss
    
    
class DistillTrainer(Trainer):
    def __init__(
        self,
        model,
        optim,
        lr_scheduler,
        config,
        ensemble_model,
        logger=None,
        round_num=0,
        overall_start_time=None,
    ) -> None:
        self.ensemble_model = ensemble_model
        super().__init__(model, optim, lr_scheduler, config, logger, round_num, overall_start_time)

    def compute_loss(self, batch):
        '''
        Compute loss with both next token perdiction and kl div with teacher logits.
        '''
        # ----------------------------
        # Compute Ensemble Predictions
        # ----------------------------
        ensemble_logits = None
        if self.ensemble_model is not None:
            with torch.no_grad():
                ensemble_outputs = self.ensemble_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                ensemble_logits = ensemble_outputs.logits

        # ----------------------------
        # Next token prediction and loss (sum)
        # ----------------------------
        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        labels = batch.pop('labels')
        outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        logits = outputs.logits

        if self.ensemble_model is not None:
            num_models = len(self.ensemble_model.models)
            logits += ensemble_logits
            logits /= num_models + 1
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')       # Ignore_index=-100
        shift_logits = shift_logits.view(-1, embedding_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        next_token_loss = loss_fct(shift_logits, shift_labels)
        ignore_index = getattr(self.config, "ignore_index", -100)
        valid_mask = shift_labels.ne(ignore_index)
        valid_count = valid_mask.sum()
        # Only calculate loss for those that are not chat template / question and not padded. 
        # valid_count = batch['attention_mask'].sum() + batch['start_index'].sum()
        
        # -------------------------
        # Compute Loss
        # -------------------------
        alpha = self.config.alpha if not self.config.synthetic_data else 1
        kl_loss = 0
        if (labels != -100).sum == 0:
            print(labels)
        if not self.config.synthetic_data:
            kl_loss = self.compute_kl_loss(logits, mask=labels != -100, inputs=batch)
        hybrid_loss = (1 - alpha) * kl_loss + alpha * next_token_loss
        
        # Log training loss details if logger is available
        if self.logger is not None and self.rank == 0 and self.tr_step % self.config.logging_steps == 0:
            self.logger.log(
                function="compute_loss",
                round_num=self.round_num,
                phase="train",
                role="student",
                step=self.tr_step,
                train_loss=hybrid_loss.item(),
                train_next_token_loss=next_token_loss.item(),
                train_kl_loss=kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
                alpha=alpha,
                learning_rate=self.lr_scheduler.get_last_lr()[0] if hasattr(self.lr_scheduler, 'get_last_lr') else None,
            )

        return hybrid_loss, next_token_loss, kl_loss, valid_count
    
    def compute_kl_loss(self, student_logits, mask, inputs, temperature=1.0):
        
        # -----------------------
        # Compute KL Loss
        # -----------------------
        # sum(len(inputs['logprob_indices'][i])) = mask.sum()
        student_probs = F.log_softmax(student_logits / temperature, dim=-1)
        student_masked_probs = student_probs[mask]        # [valid_count, vocab_size]
        teacher_logprob_values = torch.cat([torch.tensor(inputs['logprob_values'][i]) for i in range(len(inputs['logprob_values']))], dim=0).to(student_logits.device)
        teacher_logprob_indices = torch.cat([torch.tensor(inputs['logprob_indices'][i]) for i in range(len(inputs['logprob_indices']))], dim=0).to(torch.int64).to(student_logits.device, dtype=torch.int64)
        student_selected_probs = student_masked_probs.gather(dim=-1, index=teacher_logprob_indices)
        kl_loss = F.kl_div(student_selected_probs, teacher_logprob_values, log_target=True, reduction="none").sum()

        # Alternatively
        # t = (teacher_logit_values / temperature)
        # t_logZ = torch.logsumexp(t, dim=-1, keepdim=True)                 # [B,T,1]
        # teacher_logp_S = t - t_logZ                                       # [B,T,K]

        # # student log-probs on S but with FULL-vocab normalization
        # s_logZ_full = torch.logsumexp(student_logits / temperature, dim=-1, keepdim=True)  # [B,T,1]
        # student_selected = student_logits.gather(-1, teacher_logit_indices)       # [B,T,K]
        # student_logp_S_full = (student_selected / temperature) - s_logZ_full                # [B,T,K]

        # # KL(teacher || student) over the full vocab equals sum over S of p_T * (log p_T - log q_full)
        # kl_manual = torch.exp(teacher_logp_S) * (teacher_logp_S - student_logp_S_full)
        # kl_manual = kl_manual.sum(-1)[mask].sum()                                         # [B,T]

        # if dist.get_rank() == 0:
        #     print("KL: ", kl_loss.item(), kl_manual.item())
        # Alternative (manual) method is slightly different from the exact method. But experimentally it doesn't save any memory.

        return kl_loss










