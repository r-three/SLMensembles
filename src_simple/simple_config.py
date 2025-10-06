"""
Simplified configuration for single teacher-student distillation.
"""
import os
from dataclasses import dataclass


@dataclass
class DistillationConfig:
    # Model configurations
    teacher_model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    student_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Training parameters
    num_epochs: int = 3
    batch_size: int = 8
    eval_batch_size: int = 16
    learning_rate: float = 5e-5
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1  # Number of steps to accumulate gradients
    
    # Distillation parameters
    alpha: float = 0.5  # Weight for CE loss vs KL loss (0 = pure KL, 1 = pure CE)
    temperature: float = 3.0  # Temperature for distillation
    
    # Dataset
    dataset_name: str = "HuggingFaceTB/cosmopedia-100k"
    max_seq_length: int = 512
    
    # Checkpointing and logging
    output_dir: str = "./outputs/distillation"
    save_steps: int = 500
    eval_steps: int = 100
    resume_from_checkpoint: bool = False
    
    # Early stopping (optional)
    early_stop_patience: int = None  # Set to int to enable early stopping
    early_stop_min_delta: float = 0.0  # Minimum improvement to reset patience
    
    # System
    seed: int = 42
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


# Global config instance
config = DistillationConfig()
