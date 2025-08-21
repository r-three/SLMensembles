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
import math
from checkpoint import Checkpointer

# ---------------------- Callbacks ----------------------

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
            epoch_num=getattr(state, "epoch", None),
            phase="eval",
            role="student",
            step=state.global_step,
            eval_loss=loss.mean().item(),
            alpha=config.alpha,
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        control = super().on_log(args, state, control, logs=logs, **kwargs)

        self.logger.log(
            function="on_log",
            round_num=self.round_num,
            epoch_num=getattr(state, "epoch", None),
            phase="train",
            role="student",
            step=state.global_step,
            train_loss=logs.get("loss"),
            learning_rate=logs.get("learning_rate"),
            grad_norm=logs.get("grad_norm"),
            alpha=config.alpha,
        )

        return control

# ---------------------- Trainer ----------------------

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
        checkpointer=None,
        round_num=0,
        overall_start_time=None,
        wandb_run=None,
    ) -> None:
        self.model = model
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.logger = logger
        self.checkpointer = checkpointer
        self.round_num = round_num
        self.overall_start_time = overall_start_time
        self.wandb_run = wandb_run
        self.processed_id = 0
        self.gad = 0    # gradient accumulated incremented before fwd pass.
        self.gas = config.gradient_accumulation_steps
        self.tr_step = 0    # Need to update when read ckpt
        self.rank = dist.get_rank()
        self.min_eval_loss = 1e12
        self.current_eval_loss = 1e12
        # Early stopping state
        self.best_loss = float('inf')
        self.wait = 0
        self.should_stop = False

    def prepare_train(self):
        if dist.get_rank() == 0:
            with open("results.csv", "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Initialized training", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                writer.writerow(["Train step (local)", "Mean eval loss", "Next token loss", "KL loss", "Valid count"]) 
        self.model.train()
        self.model.config.use_cache = False  # avoid cache warnings in training
        # Lightweight init log
        if self.logger is not None and self.rank == 0:
            self.logger.log(
                function="prepare_train",
                round_num=self.round_num,
                epoch_num=getattr(self.state, "epoch", None),
                phase="init",
                role="student",
                step=self.tr_step,
                learning_rate=self.lr_scheduler.get_last_lr()[0] if hasattr(self.lr_scheduler, 'get_last_lr') else None,
                alpha=config.alpha,
            )

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

    def step(self, train_batch, eval_dl, epoch, wandb_run):
        train_loss = self.train_step(train_batch, epoch)

        test_loss = None
        if self.tr_step % self.config.logging_steps == 0:
            test_loss = self.eval_step(eval_dl, epoch)
        
        if self.wandb_run is not None and dist.get_rank() == 0:
            log_dict = {
                "train/loss": train_loss,
                "train/epoch": epoch,
                "train/round": self.round_num,
                "train/step": self.tr_step,
                "train/learning_rate": self.lr_scheduler.get_last_lr()[0] if hasattr(self.lr_scheduler, 'get_last_lr') else None,
            }
            
            if test_loss is not None:
                log_dict.update({
                    "eval/loss": test_loss,
                    "eval/epoch": epoch,
                    "eval/round": self.round_num,
                    "eval/step": self.tr_step,
                })
            
            self.wandb_run.log(log_dict, step=self.tr_step)
        
        if self.tr_step % config.ckpt_save_steps == 0 and dist.get_rank() == 0: self.save_checkpoint()
        dist.barrier()
        
        self.tr_step += 1
        return train_loss, test_loss
    
    def train_step(self, batch, epoch):
        self.model.train()
        batch["input_ids"] = batch["input_ids"].type(torch.LongTensor)
        batch["labels"] = batch["labels"].type(torch.LongTensor)

        self.gad += 1
        self.processed_id += 1

        grad_norm = None
        # Compute loss and backpropagate (supporting grad accumulation)
        if (self.tr_step + 1) % self.gas != self.gas - 1:
            # no need to sync while accumulating gradients
            self.model.set_requires_gradient_sync(False)  # with (grad = False):
            tr_step_loss, next_token_loss, kl_loss, valid_count = self.compute_loss(batch)
            (tr_step_loss / self.gas).backward()
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)
                grad_norm = float(grad_norm)
            except Exception:
                grad_norm = None
            self.model.set_requires_gradient_sync(True)
        else:
            # next forward / backward pass will be synced
            self.model.set_requires_gradient_sync(True)
            dist.barrier()
            tr_step_loss, next_token_loss, kl_loss, valid_count = self.compute_loss(batch)
            (tr_step_loss / self.gas).backward()
            try:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)
                grad_norm = float(grad_norm)
            except Exception:
                grad_norm = None
            self.optim.step()
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                self.lr_scheduler.step(self.metric)
            else:
                self.lr_scheduler.step()
            self.optim.zero_grad()

        loss_sum = _gather(tr_step_loss.reshape(1)).mean().item()
        nt_sum = _gather((next_token_loss if next_token_loss is not None else torch.tensor(0.0, device=tr_step_loss.device)).reshape(1)).mean().item()
        kl_sum = _gather((kl_loss if kl_loss is not None else torch.tensor(0.0, device=tr_step_loss.device)).reshape(1)).mean().item()
        valid_sum = _gather(valid_count.reshape(1)).mean().item()

        # Periodic CSV logging
        if (
            self.logger is not None
            and self.rank == 0
            and (self.tr_step % getattr(self.config, "logging_steps", 1) == 0)
        ):
            self.logger.log(
                function="train_step",
                round_num=self.round_num,
                epoch_num=getattr(self.state, "epoch", None),
                phase="train",
                role="student",
                step=self.tr_step,
                train_loss=loss_sum,
                train_next_token_loss=nt_sum,
                train_kl_loss=kl_sum,
                grad_norm=grad_norm,
                learning_rate=self.lr_scheduler.get_last_lr()[0] if hasattr(self.lr_scheduler, 'get_last_lr') else None,
                alpha=config.alpha,
            )

        return loss_sum
    
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
                if next_token_sum is not None:
                    nxt_token_loss += next_token_sum
                if kl_sum is not None:
                    kl_loss += kl_sum
                valid_total += valid_cnt
        
        # So you don't see eval loss of a few million
        gathered_eval_loss = _gather(eval_loss.reshape(1)).sum().item()
        gathered_nxt_token_loss = _gather(nxt_token_loss.reshape(1)).sum().item()
        gathered_kl_loss = _gather(kl_loss.reshape(1)).sum().item()
        gathered_valid_total = _gather(valid_total.reshape(1)).sum().item()
        # Take the average of eval_loss on both cards, 
        mean_eval_loss = gathered_eval_loss / gathered_valid_total
        mean_nk_loss = gathered_nxt_token_loss / gathered_valid_total if gathered_valid_total > 0 else None
        mean_kl_loss = gathered_kl_loss / gathered_valid_total if gathered_valid_total > 0 else None

        self.metric = mean_eval_loss

        main_print(f"Step: {self.tr_step}, eval loss: {mean_eval_loss}")

        if dist.get_rank() == 0:
            with open("results.csv", "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([self.tr_step, mean_eval_loss, mean_nk_loss, mean_kl_loss, gathered_valid_total])
            
            if self.logger is not None:
                self.logger.log(
                    function="eval_step",
                    round_num=self.round_num,
                    epoch_num=epoch,
                    phase="eval",
                    role="student",
                    step=self.tr_step,
                    eval_loss=mean_eval_loss,
                    eval_next_token_loss=mean_nk_loss,
                    eval_kl_loss=mean_kl_loss,
                    perplexity=math.exp(mean_eval_loss) if mean_eval_loss is not None else None,
                    learning_rate=self.lr_scheduler.get_last_lr()[0] if hasattr(self.lr_scheduler, 'get_last_lr') else None,
                    alpha=config.alpha,
                )
                try:
                    self.logger.flush()
                except Exception:
                    pass

        self.min_eval_loss = min(mean_eval_loss, self.min_eval_loss)
        self.current_eval_loss = mean_eval_loss

        # -------- Early stopping --------
        if self.best_loss - mean_eval_loss < config.early_stop_min_delta:
            self.best_loss = mean_eval_loss
            self.wait = 0
        else:
            self.wait += 1
        if self.wait >= config.early_stop_patience:
            main_print(f"Early stopping triggered: no improvement for {self.wait} evaluations.")
            self.should_stop = True
        
        self.model.train()
        return mean_eval_loss

    def save_checkpoint(self):
        """Save model+optim via DCP and rotate per-round."""

        round_num = int(getattr(self, "round_num", 0))
        step = int(getattr(self, "global_step", getattr(self, "tr_step", 0)))
        loss = float(getattr(self, "eval_loss", getattr(self, "train_loss", float("inf"))))

        training_state = {
            "epoch": getattr(self, "epoch", 0),
            "round_num": round_num,
            "global_step": step,
            "loss": loss,
            "rng": torch.random.get_rng_state().tolist() if torch.random else None,
        }

        self.checkpointer.save(self.model, self.optim, round_num, step, loss, training_state)

# ---------------------- DistillTrainer ----------------------

class DistillTrainer(Trainer):
    def __init__(
        self,
        model,
        optim,
        lr_scheduler,
        config,
        ensemble_model,
        logger=None,
        checkpointer=None,
        round_num=0,
        overall_start_time=None,
        wandb_run=None,
    ):
        super().__init__(
            model,
            optim,
            lr_scheduler,
            config,
            logger,
            checkpointer,
            round_num,
            overall_start_time,
            wandb_run,
        )
        self.ensemble_model = ensemble_model

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
        loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
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

        if self.logger is not None:
            self.logger.log(
                function="train_step",
                round_num=self.round_num,
                epoch_num=getattr(self.state, "epoch", None),
                phase="train",
                role="student",
                step=self.tr_step,
                train_loss=hybrid_loss,
                train_next_token_loss=next_token_loss,
                train_kl_loss=kl_loss,
                perplexity=math.exp(hybrid_loss) if hybrid_loss is not None else None,
                learning_rate=self.lr_scheduler.get_last_lr()[0] if hasattr(self.lr_scheduler, 'get_last_lr') else None,
                alpha=config.alpha,
            )

        return hybrid_loss, next_token_loss, kl_loss, valid_count

    def compute_kl_loss(self, student_logits, mask, inputs):
        # -----------------------
        # Compute KL Loss
        # -----------------------

        # sum(len(inputs['logprob_indices'][i])) = mask.sum()
        student_probs = F.log_softmax(student_logits / config.kl_temperature, dim=-1)
        student_masked_probs = student_probs[mask]        # [valid_count, vocab_size]
        
        teacher_logprob_values = torch.cat([torch.tensor(inputs['logprob_values'][i]) for i in range(len(inputs['logprob_values']))], dim=0).to(student_logits.device)
        teacher_logprob_indices = torch.cat([torch.tensor(inputs['logprob_indices'][i]) for i in range(len(inputs['logprob_indices']))], dim=0).to(torch.int64).to(student_logits.device, dtype=torch.int64)
        
        student_selected_probs = student_masked_probs.gather(dim=-1, index=teacher_logprob_indices)
        
        kl_loss = F.kl_div(student_selected_probs, teacher_logprob_values, log_target=True, reduction="none").sum()
        
        return kl_loss

