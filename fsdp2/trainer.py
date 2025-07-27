from abc import ABC, abstractmethod
import sys
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from utils import main_print
from tqdm.auto import tqdm

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
        return loss, valid_count

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
            tr_step_loss, _ = self.compute_loss(batch)
            (tr_step_loss / self.gas).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)
            self.model.set_requires_gradient_sync(True)
        else:
            # next forward / backward pass will be synced
            self.model.set_requires_gradient_sync(True)
            dist.barrier()
            tr_step_loss, _ = self.compute_loss(batch)
            (tr_step_loss / self.gas).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)
            self.optim.step()
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                self.lr_scheduler.step(self.metric)
            else:
                self.lr_scheduler.step()
            self.optim.zero_grad()
        gathered_tr_step_loss = _gather(tr_step_loss.reshape(1)).mean().item()

        # TODO: add back logging.

        return gathered_tr_step_loss
    
    def eval_step(self, eval_dl, epoch: int) -> float:
        main_print("Evaluating")
        self.model.eval()
        eval_loss = torch.tensor(0.0).to(torch.cuda.current_device())
        valid_total = torch.tensor(0).to(torch.cuda.current_device())
        for _, batch in enumerate(tqdm(eval_dl,
                                  disable=self.rank != 0,
                                  file=sys.__stdout__,)):
            with torch.no_grad():
                batch["input_ids"] = batch["input_ids"].type(torch.LongTensor)
                batch["labels"] = batch["labels"].type(torch.LongTensor)
                loss_sum, valid_cnt = self.compute_loss(batch)
                eval_loss += loss_sum
                valid_total += valid_cnt
        
        eval_loss /= valid_total
        # So you don't see eval loss of a few million
        gathered_eval_loss = _gather(eval_loss.reshape(1)).mean().item()
        # Take the average of eval_loss on both cards, 
        mean_eval_loss = gathered_eval_loss
        print("Mean eval loss: ", mean_eval_loss)

        self.metric = gathered_eval_loss

        main_print(f"Step: {self.tr_step}, eval loss: {mean_eval_loss}")
        # if self.wandb_logging:
        #     self.log(mean_eval_loss, epoch, "eval")

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
    ) -> None:
        super().__init__(model, tokenizer, optim, lr_scheduler, config)

    def compute_loss(self, batch):
        '''
        Compute loss with both next token perdiction and kl div with teacher logits.
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
        next_token_loss = loss_fct(shift_logits, shift_labels)
        ignore_index = getattr(self.config, "ignore_index", -100)
        valid_mask = shift_labels.ne(ignore_index)
        valid_count = valid_mask.sum()
        return next_token_loss, kl_loss, hybrid_loss, valid_count