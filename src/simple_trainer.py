import torch
import torch.nn.functional as F
import torch.distributed as dist
from tqdm.auto import tqdm
import sys

from simple_config import config
from simple_utils import is_main_process, main_print


def _gather(x: torch.Tensor) -> torch.Tensor:
    """Gather tensors across all processes."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return x
    
    device = x.device
    world_size = dist.get_world_size()
    
    # Gather tensor sizes first
    local_size = torch.tensor(x.size(), device=device)
    size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    
    # Pad tensors to max size
    max_size = torch.stack(size_list).max(dim=0).values
    padded = torch.zeros(max_size.tolist(), dtype=x.dtype, device=device)
    padded[:x.size(0)] = x
    
    # Gather padded tensors
    tensor_list = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(tensor_list, padded)
    
    # Unpad and concatenate
    output = []
    for i, size in enumerate(size_list):
        output.append(tensor_list[i][:size[0].item()])
    
    return torch.cat(output, dim=0)


class DistillTrainer:
    """Simplified trainer for single teacher-student distillation."""
    
    def __init__(
        self,
        student_model,
        teacher_model,
        optimizer,
        lr_scheduler,
        checkpointer=None,
    ):
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.checkpointer = checkpointer
        self.global_step = 0
        
    def compute_distillation_loss(self, student_logits, teacher_logits, labels):
        """
        Compute hybrid loss combining:
        1. KL divergence between student and teacher distributions
        2. Cross-entropy loss on true labels
        """
        # Shift for next-token prediction
        shift_student_logits = student_logits[..., :-1, :].contiguous()
        shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten
        vocab_size = student_logits.size(-1)
        shift_student_logits = shift_student_logits.view(-1, vocab_size)
        shift_teacher_logits = shift_teacher_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        
        # Move to same device
        shift_labels = shift_labels.to(shift_student_logits.device)
        
        # Create mask for valid positions (not padding)
        mask = shift_labels != -100
        
        # KL Divergence Loss
        if mask.sum() > 0:
            student_log_probs = F.log_softmax(shift_student_logits[mask] / config.temperature, dim=-1)
            teacher_probs = F.softmax(shift_teacher_logits[mask] / config.temperature, dim=-1)
            kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='sum')
            
            # Scale KL loss by temperature squared (as per standard practice)
            kl_loss = kl_loss * (config.temperature ** 2)
        else:
            kl_loss = torch.tensor(0.0, device=shift_student_logits.device)
        
        # Cross Entropy Loss
        ce_loss = F.cross_entropy(shift_student_logits, shift_labels, ignore_index=-100, reduction='sum')
        
        # Combine losses
        total_loss = config.alpha * ce_loss + (1 - config.alpha) * kl_loss
        
        # Return average loss and token count for proper averaging across GPUs
        valid_tokens = mask.sum()
        
        return total_loss, valid_tokens, ce_loss, kl_loss
    
    def train_step(self, batch):
        """Single training step with distillation."""
        self.student_model.train()
        self.teacher_model.eval()
        
        # Move batch to GPU
        input_ids = batch["input_ids"].to(torch.cuda.current_device())
        attention_mask = batch["attention_mask"].to(torch.cuda.current_device())
        labels = batch["labels"].to(torch.cuda.current_device())
        
        # Teacher forward pass (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            teacher_logits = teacher_outputs.logits
        
        # Student forward pass
        student_outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        student_logits = student_outputs.logits
        
        # Compute loss
        loss, valid_tokens, ce_loss, kl_loss = self.compute_distillation_loss(
            student_logits, teacher_logits, labels
        )
        
        # Normalize by tokens for gradient accumulation
        normalized_loss = loss / valid_tokens if valid_tokens > 0 else loss
        
        # Backward
        normalized_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), config.max_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        
        self.global_step += 1
        
        # Gather losses across GPUs for logging
        total_loss = _gather(loss.unsqueeze(0)).sum().item()
        total_tokens = _gather(valid_tokens.unsqueeze(0)).sum().item()
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        
        return avg_loss
    
    def evaluate(self, eval_dataloader):
        """Evaluate the student model."""
        self.student_model.eval()
        self.teacher_model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating", disable=dist.get_rank() != 0):
                # Move batch to GPU
                input_ids = batch["input_ids"].to(torch.cuda.current_device())
                attention_mask = batch["attention_mask"].to(torch.cuda.current_device())
                labels = batch["labels"].to(torch.cuda.current_device())
                
                # Teacher forward
                teacher_outputs = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                teacher_logits = teacher_outputs.logits
                
                # Student forward
                student_outputs = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                student_logits = student_outputs.logits
                
                # Compute loss
                loss, valid_tokens, _, _ = self.compute_distillation_loss(
                    student_logits, teacher_logits, labels
                )
                
                total_loss += loss.item()
                total_tokens += valid_tokens.item()
        
        # Gather across GPUs
        total_loss_tensor = torch.tensor(total_loss, device=torch.cuda.current_device())
        total_tokens_tensor = torch.tensor(total_tokens, device=torch.cuda.current_device())
        
        gathered_loss = _gather(total_loss_tensor.unsqueeze(0)).sum().item()
        gathered_tokens = _gather(total_tokens_tensor.unsqueeze(0)).sum().item()
        
        avg_loss = gathered_loss / gathered_tokens if gathered_tokens > 0 else 0.0
        
        self.student_model.train()
        
        return avg_loss
