# Quick Start Guide - Refactored Simple Trainer

## What Changed?

The `simple_trainer.py` has been refactored to match the original `trainer.py` architecture while staying simple:

### ✅ Added (from original trainer.py)
- Proper gradient accumulation with FSDP sync control
- Sum reduction for losses (better gradient accumulation)
- Early stopping support
- Robust distributed gathering
- Integrated checkpointing
- Base `Trainer` class architecture

### ❌ Removed (kept simple)
- Callbacks and complex logging
- Ensemble support
- ID tracking
- Multi-round training
- Sparse logprobs

## Running the Training

### 1. Check Configuration

Edit `simple_config.py`:

```python
# Key parameters
gradient_accumulation_steps: int = 1  # Increase for memory savings
alpha: float = 0.5  # 0=pure KL, 1=pure CE
temperature: float = 3.0  # Distillation temperature

# Optional early stopping
early_stop_patience: int = None  # Set to int to enable (e.g., 5)
early_stop_min_delta: float = 0.0  # Minimum improvement
```

### 2. Launch Training

```bash
# Single GPU
python main_simple.py

# Multi-GPU (FSDP)
torchrun --nproc_per_node=2 main_simple.py --mixed-precision

# Or with custom config
# (Modify config values in simple_config.py first)
torchrun --nproc_per_node=4 main_simple.py
```

### 3. Monitor Progress

The trainer will print:
- Training loss per batch
- Evaluation loss every `eval_steps`
- Epoch summaries
- Early stopping notifications (if enabled)

## Key Features

### Gradient Accumulation

Now fully supported! Set in config:

```python
config.gradient_accumulation_steps = 4
```

The trainer handles:
- Gradient sync control (FSDP compatible)
- Loss normalization
- Optimizer stepping only on final accumulation step

### Early Stopping

Enable in config:

```python
config.early_stop_patience = 5  # Stop after 5 evals without improvement
config.early_stop_min_delta = 0.001  # Minimum improvement threshold
```

Training will stop automatically when no improvement is detected.

### Checkpointing

Automatic checkpointing every `save_steps`:

```python
config.save_steps = 500  # Save every 500 steps
```

Checkpoints include:
- Model state
- Optimizer state
- LR scheduler state
- Training metadata (epoch, step, loss)

Resume from checkpoint:

```python
config.resume_from_checkpoint = True
```

## Code Structure

### Trainer Hierarchy

```
Trainer (Abstract Base)
├── train_step(batch) - Handles gradient accumulation
├── eval_step(dataloader) - Full evaluation
├── save_checkpoint() - Save model state
└── compute_loss(batch) - Abstract method

    ↓ (inherits from)

DistillTrainer
└── compute_loss(batch) - Hybrid distillation loss
```

### Training Loop (main_simple.py)

```python
# Initialize
trainer = DistillTrainer(student, teacher, optimizer, scheduler, checkpointer)

# Train
for epoch in range(num_epochs):
    for batch in train_loader:
        loss = trainer.train_step(batch)  # Handles everything internally
        
        if trainer.global_step % eval_steps == 0:
            eval_loss = trainer.eval_step(eval_loader)
            
            if trainer.should_stop:  # Early stopping
                break
        
        if trainer.global_step % save_steps == 0:
            trainer.save_checkpoint()
```

## Loss Computation

The trainer computes a hybrid loss:

```
Total Loss = α × CE_loss + (1 - α) × KL_loss
```

Where:
- **CE Loss**: Cross-entropy with true labels (standard LM training)
- **KL Loss**: KL divergence with teacher logits (distillation)
- **α (alpha)**: Weight between the two (0 to 1)

Both losses use `reduction='sum'` for proper gradient accumulation.

## Distributed Training

Fully compatible with:
- **FSDP**: Fully Sharded Data Parallel
- **Mixed Precision**: bfloat16 training
- **Multi-GPU**: Proper synchronization and gathering

The `_gather()` function handles distributed tensor gathering safely.

## Troubleshooting

### Issue: OOM (Out of Memory)

**Solution:**
1. Increase `gradient_accumulation_steps`
2. Decrease `batch_size`
3. Decrease `max_seq_length`

### Issue: Training not stopping early

**Solution:**
- Check `early_stop_patience` is set (not `None`)
- Verify `eval_steps` is frequent enough
- Check if loss is actually improving

### Issue: Checkpoint not saving

**Solution:**
- Verify `output_dir` is writable
- Check `save_steps` interval
- Ensure you're running past step 0

### Issue: Loss is NaN

**Solution:**
- Reduce learning rate
- Check `max_grad_norm` (default: 1.0)
- Verify data has no NaN values
- Check temperature is reasonable (2-5 typical)

## Comparison to Original

| Feature | Original trainer.py | New simple_trainer.py |
|---------|-------------------|----------------------|
| Architecture | Base + Subclass | ✅ Base + Subclass |
| Gradient Accum | ✅ | ✅ |
| Sum Reduction | ✅ | ✅ |
| Early Stopping | ✅ | ✅ |
| FSDP Compatible | ✅ | ✅ |
| Callbacks | ✅ | ❌ (simplified) |
| CSV Logging | ✅ | ❌ (simplified) |
| Ensemble | ✅ | ❌ (simplified) |
| Multi-round | ✅ | ❌ (simplified) |

## Next Steps

1. **Test with small dataset** - Verify everything works
2. **Enable early stopping** - Set `early_stop_patience`
3. **Tune hyperparameters** - Adjust `alpha`, `temperature`, `learning_rate`
4. **Scale up** - Increase `batch_size` and `gradient_accumulation_steps`
5. **Monitor** - Watch eval loss for convergence

## Documentation

- `TRAINER_REFACTORING.md` - Detailed technical documentation
- `CHANGES_SUMMARY.md` - Complete list of changes
- `MIGRATION_GUIDE.md` - Migration from old src_simple (if exists)

## Questions?

The refactored trainer maintains the robust training logic from `src/trainer.py` while keeping the simplicity of the original `src_simple/`. It's production-ready for single teacher-student distillation!

