import os
import torch
from dataclasses import dataclass

@dataclass
class DistillationConfig:
    # Model configurations
    teacher_model_name = "allenai/OLMo-2-1124-7B-SFT"
    student_model_name = "allenai/OLMo-2-0425-1B-SFT"
    student_vocab_size = 100278
    tokenizer_name = "allenai/OLMo-2-1124-7B-SFT" 

    # Dataset and Path
    dataset_path = "/scratch/klambert/dataset/tulu-3-sft-mixture-pretokenized"
    output_dir = "/scratch/klambert/model_log/singular"
    dataset_name = "allenai/tulu-3-sft-mixture"
    
    # Training parameters
    num_epochs: int = 3
    batch_size: int = 8
    eval_batch_size: int = 16
    learning_rate: float = 5e-5
    max_grad_norm: float = 1.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4  # Number of steps to accumulate gradients
    
    # Distillation parameters
    alpha: float = 0.5  # Weight for CE loss vs KL loss (0 = pure KL, 1 = pure CE)
    kl_temperature: float = 3.0  # Temperature for distillation
    
    # Checkpointing and logging
    save_steps: int = 500
    eval_steps: int = 100
    resume_from_checkpoint: bool = False
    
    # System
    seed: int = 42

    # Wandb logging
    wandb_project: str = "slm-distillation"
    wandb_run_name: str = None  # Auto-generated if None
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


# Global config instance
config = DistillationConfig()
