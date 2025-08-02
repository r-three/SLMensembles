from abc import ABC, abstractmethod
import sys
import csv
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
import torch.nn.functional as F
from utils import main_print
from tqdm.auto import tqdm
from datetime import datetime

def _gather(x: torch.Tensor) -> torch.Tensor:
    output_tensors = [x.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, x)
    return torch.cat(output_tensors, dim=0)

class Trainer(ABC):
    def __init__(
        self,
        model,
        tokenizer,
        optim,
        lr_scheduler,
        config,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.processed_id = 0
        self.gad = 0    # gradient accumulated incremented before fwd pass.
        self.gas = config.gradient_accumulation_steps
        self.tr_step = 0    # Need to update when read ckpt
        self.rank = dist.get_rank()

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
            self.model.set_requires_gradient_sync(False)
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

        # No logging for train steps.

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
        print("Mean eval loss: ", mean_eval_loss)

        self.metric = gathered_eval_loss

        main_print(f"Step: {self.tr_step}, eval loss: {mean_eval_loss}")

        if dist.get_rank() == 0:
            with open("results.csv", "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([self.tr_step, mean_eval_loss, mean_nk_loss, mean_kl_loss, gathered_valid_total])

        self.model.train()
        return gathered_eval_loss
    
class DistillTrainer(Trainer):
    def __init__(
        self,
        model,
        tokenizer,
        optim,
        lr_scheduler,
        config,
        ensemble_model,
    ) -> None:
        self.ensemble_model = ensemble_model
        super().__init__(model, tokenizer, optim, lr_scheduler, config)

    def compute_loss(self, batch):
        '''
        Compute loss with both next token perdiction and kl div with teacher logits.
        '''
        # ----------------------------
        # Next token prediction and loss (sum)
        # ----------------------------
        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        labels = batch.pop('labels')
        outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        logits = outputs.logits
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


        # ----------------------------
        # Compute Ensemble Predictions
        # ----------------------------
        ensemble_logits = None
        if self.ensemble_model is not None:
            raise NotImplementedError
            with torch.no_grad():
                ensemble_outputs = self.ensemble_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                ensemble_logits = ensemble_outputs.logits
        
        # -------------------------
        # Compute Loss
        # -------------------------
        alpha = self.config.alpha if not self.config.synthetic_data else 1
        kl_loss = 0
        if not self.config.synthetic_data:
            kl_loss = self.compute_kl_loss(logits, ensemble_logits, mask=labels != -100, inputs=batch)
        hybrid_loss = (1 - alpha) * kl_loss + alpha * next_token_loss

        return hybrid_loss, next_token_loss, kl_loss, valid_count
    
    def compute_kl_loss(self, student_logits, ensemble_logits, mask, inputs, temperature=1.0):
        
        # ----------------------------------------
        # Combine model predictions with ensemble
        # ----------------------------------------
        if ensemble_logits is not None:
            raise NotImplementedError
            num_models = len(self.ensemble_model.models)
            student_logits = student_logits / (num_models + 1) + ensemble_logits * (num_models / (num_models + 1))

        # -----------------------
        # Compute KL Loss
        # -----------------------
        # sum(len(inputs['logprob_indices'][i])) = mask.sum()
        student_probs = F.log_softmax(student_logits / temperature, dim=-1)
        student_masked_probs = student_probs[mask]        # [valid_count, vocab_size]
        teacher_logprob_values = torch.cat([torch.tensor(inputs['logprob_values'][i]) for i in range(len(inputs['logprob_values']))], dim=0).to(student_logits.device)
        teacher_logprob_indices = torch.cat([torch.tensor(inputs['logprob_indices'][i]) for i in range(len(inputs['logprob_indices']))], dim=0).to(torch.int64).to(student_logits.device, dtype=torch.int64)
        student_selected_probs = student_masked_probs.gather(dim=-1, index=teacher_logprob_indices)
        kl_loss = F.kl_div(student_selected_probs, teacher_logprob_values, log_target=True, reduction="none").sum(-1)
        kl_loss = kl_loss[mask].sum()

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