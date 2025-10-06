# Simplified Teacher-Student Distillation

This is a streamlined implementation of knowledge distillation from a teacher model to a student model, optimized for simplicity and efficiency.

## Key Features

- **Single Teacher-Student Distillation**: Direct knowledge transfer from one teacher to one student
- **FSDP Parallelization**: Efficient multi-GPU training with PyTorch FSDP
- **Simple Checkpointing**: Basic checkpoint saving/loading functionality
- **Minimal Dependencies**: Only essential components included

## Removed Complexities

The following features were removed to create a minimal, focused codebase:
- Ensemble functionality
- Round-based training
- ID tracking and loss logging
- Manifest management
- Complex checkpoint rotation
- Signal handling
- CSV logging
- Early stopping
- Wandb integration (can be easily added back if needed)

## File Structure

```
src/
├── main_simple.py         # Main training script
├── simple_trainer.py      # Distillation trainer
├── simple_config.py       # Configuration
├── simple_utils.py        # Utilities
└── simple_checkpoint.py   # Checkpoint management
```

## Usage

### 1. Configure Training

Edit `src/simple_config.py` to set:
- Teacher and student model names
- Training hyperparameters
- Dataset configuration
- Output directory

### 2. Run Training

```bash
# Single GPU
python src/main_simple.py

# Multi-GPU (recommended)
./run_simple_distillation.sh
```

### 3. Key Parameters in Config

- `alpha`: Balance between CE loss and KL loss (0=pure KL, 1=pure CE)
- `temperature`: Softmax temperature for distillation
- `num_epochs`: Number of training epochs
- `batch_size`: Training batch size per GPU

## Training Process

1. **Dataset Loading**: Loads and tokenizes the dataset
2. **Model Setup**: 
   - Loads frozen teacher model
   - Initializes trainable student model
   - Applies FSDP sharding for memory efficiency
3. **Training Loop**:
   - Teacher generates soft targets
   - Student learns from both soft targets (KL) and hard labels (CE)
   - Hybrid loss: `α * CE_loss + (1-α) * KL_loss`
4. **Checkpointing**: Saves model periodically and keeps last 3 checkpoints

## Customization

This minimal codebase is designed to be easily extended. Common modifications:

1. **Add Wandb Logging**: 
   ```python
   import wandb
   wandb.init(project="distillation")
   wandb.log({"loss": loss})
   ```

2. **Change Models**: Update model names in `simple_config.py`

3. **Modify Loss Function**: Edit `compute_distillation_loss()` in `simple_trainer.py`

4. **Add Evaluation Metrics**: Extend the `evaluate()` method

## Requirements

- PyTorch >= 2.0
- Transformers
- Datasets
- tqdm

## Notes

- The codebase uses BF16 mixed precision by default for efficiency
- Gradient accumulation can be added if needed for larger effective batch sizes
- The teacher model is kept frozen throughout training
