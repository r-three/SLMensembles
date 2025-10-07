# Simplified Teacher-Student Distillation

This is a streamlined implementation of knowledge distillation from a teacher model to a student model, optimized for simplicity and efficiency.

## Key Features

- **Single Teacher-Student Distillation**: Direct knowledge transfer from one teacher to one student
- **FSDP Parallelization**: Efficient multi-GPU training with PyTorch FSDP
- **Simple Checkpointing**: Basic checkpoint saving/loading functionality
- **Wandb Integration**: Built-in logging of training metrics to Weights & Biases
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
- Early stopping (configurable)

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
- `wandb_project`: Wandb project name (default: "slm-distillation")
- `wandb_run_name`: Custom run name (auto-generated if None)

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

## Wandb Integration

The pipeline includes built-in Weights & Biases logging for experiment tracking:

### Setup Wandb

```bash
# Install wandb
pip install wandb

# Login to wandb (first time only)
wandb login
```

### Configure Wandb

Edit `simple_config.py` to customize wandb settings:

```python
# Enable/disable wandb
use_wandb: bool = True

# Set your project name
wandb_project: str = "my-distillation-project"

# Set your entity (username or team name)
wandb_entity: str = "my-username"  # or None

# Custom run name (auto-generated if None)
wandb_run_name: str = None
```

### Logged Metrics

The trainer automatically logs:

**Training Metrics (per step):**
- `train/loss`: Total training loss
- `train/ce_loss`: Cross-entropy loss component
- `train/kl_loss`: KL divergence loss component
- `train/learning_rate`: Current learning rate
- `train/grad_norm`: Gradient norm (when optimizer step occurs)
- `train/epoch`: Current epoch
- `train/step`: Global training step

**Evaluation Metrics:**
- `eval/loss`: Total evaluation loss
- `eval/ce_loss`: Cross-entropy loss on eval set
- `eval/kl_loss`: KL divergence loss on eval set
- `eval/min_loss`: Minimum eval loss achieved so far
- `eval/epoch`: Current epoch
- `eval/step`: Global step when evaluation occurred

**Configuration:**
All hyperparameters are logged to wandb config including models, learning rates, batch sizes, distillation parameters, etc.

### Disable Wandb

To disable wandb logging, set in `simple_config.py`:
```python
use_wandb: bool = False
```

Or if wandb is not installed, the trainer will automatically run without logging.

## Customization

This minimal codebase is designed to be easily extended. Common modifications:

1. **Change Models**: Update model names in `simple_config.py`

2. **Modify Loss Function**: Edit `compute_loss()` in `simple_trainer.py`

3. **Add Evaluation Metrics**: Extend the `eval_step()` method

4. **Custom Wandb Metrics**: Add more wandb.log() calls in the trainer

## Requirements

- PyTorch >= 2.0
- Transformers
- Datasets
- tqdm
- wandb (optional, for logging)

## Notes

- The codebase uses BF16 mixed precision by default for efficiency
- Gradient accumulation can be added if needed for larger effective batch sizes
- The teacher model is kept frozen throughout training
