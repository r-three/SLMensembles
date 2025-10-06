# Trainer Refactoring Documentation

## Overview

The `simple_trainer.py` has been refactored to closely match the original `trainer.py` implementation while maintaining simplicity and removing unnecessary complexity.

## Key Changes

### Architecture

**Before:**
- Single `DistillTrainer` class
- Basic distillation logic

**After:**
- Abstract base `Trainer` class with common training logic
- `DistillTrainer` subclass for distillation-specific implementation
- Mirrors the structure of `src/trainer.py`

### Features Preserved from Original

1. **Gradient Accumulation Support**
   - Proper handling of gradient sync with FSDP
   - `set_requires_gradient_sync()` integration
   - Dividing loss by accumulation steps

2. **Loss Computation with Sum Reduction**
   - All losses use `reduction='sum'` for proper gradient accumulation
   - Valid token counting for accurate loss averaging
   - Per-token normalization

3. **Distributed Training**
   - Robust `_gather()` function with error handling
   - Proper GPU synchronization with `dist.barrier()`
   - Periodic memory cleanup

4. **Early Stopping**
   - Configurable patience and minimum delta
   - Tracks best loss across evaluations
   - Graceful training termination

5. **Checkpointing**
   - Integrated checkpoint saving
   - State tracking (epoch, global_step, loss)
   - Proper synchronization across processes

6. **Temperature-based KL Divergence**
   - Temperature scaling for both student and teacher logits
   - KL loss scaled by temperature squared (standard practice)

### Features Removed (Simplifications)

1. **No Callbacks**
   - Removed `LoggingCallback` and `TrainerCallback` complexity
   - Direct evaluation instead of callback-based logging

2. **No CSV Logging**
   - Removed `AsyncLossLogger` and detailed CSV logging
   - Simple print statements for progress tracking

3. **No Ensemble Support**
   - Single teacher-student setup only
   - No ensemble model averaging

4. **No Sparse Logprob Support**
   - Uses full teacher logits instead of sparse logprobs
   - Simpler memory footprint

5. **No Round-based Training**
   - Single continuous training run
   - No multi-round distillation

## Configuration

### New Config Parameters

```python
# Gradient accumulation
gradient_accumulation_steps: int = 1

# Early stopping (optional)
early_stop_patience: int = None  # Set to int to enable
early_stop_min_delta: float = 0.0
```

## Usage

### Basic Training Loop

```python
# Initialize trainer
trainer = DistillTrainer(
    student_model=student_model,
    teacher_model=teacher_model,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    checkpointer=checkpointer,
)

# Training loop
for epoch in range(num_epochs):
    trainer.epoch = epoch
    
    for batch in train_dataloader:
        # Train step handles gradient accumulation internally
        loss = trainer.train_step(batch)
        
        # Periodic evaluation
        if trainer.global_step % eval_steps == 0:
            eval_loss = trainer.eval_step(eval_dataloader)
            
            # Check early stopping
            if trainer.should_stop:
                break
        
        # Periodic checkpointing
        if trainer.global_step % save_steps == 0:
            trainer.save_checkpoint()
```

## Compatibility with FSDP

The trainer is fully compatible with PyTorch FSDP (Fully Sharded Data Parallel):

1. **Gradient Sync Control**
   - Uses `model.set_requires_gradient_sync()` for gradient accumulation
   - Only syncs on final accumulation step

2. **Distributed Barriers**
   - Proper synchronization before evaluation
   - Safe checkpointing with barriers

3. **Mixed Precision**
   - Works with FSDP `MixedPrecisionPolicy`
   - Handles bfloat16 training

## Loss Computation Details

### Hybrid Distillation Loss

```
total_loss = α * CE_loss + (1 - α) * KL_loss
```

Where:
- `α` (alpha): Weight between cross-entropy and KL divergence
- `CE_loss`: Cross-entropy on true labels (sum reduction)
- `KL_loss`: KL divergence with teacher (sum reduction, temperature-scaled)

### Normalization Strategy

- Losses use `reduction='sum'` during computation
- Divided by gradient accumulation steps for backprop
- Final averaging uses gathered valid token counts

## Key Methods

### `Trainer` (Base Class)

- `train_step(batch)`: Single training step with gradient accumulation
- `eval_step(eval_dataloader)`: Full evaluation pass
- `save_checkpoint(loss)`: Save model state
- `compute_loss(batch)`: Abstract method for loss computation

### `DistillTrainer` (Subclass)

- `compute_loss(batch)`: Implements hybrid distillation loss
  - Returns: `(total_loss, valid_count, ce_loss, kl_loss)`

## Differences from Original `trainer.py`

| Feature | Original | Simplified | Notes |
|---------|----------|------------|-------|
| Base class | ✅ | ✅ | Both have abstract Trainer |
| Gradient accumulation | ✅ | ✅ | Identical logic |
| Loss reduction | sum | sum | Same approach |
| Early stopping | ✅ | ✅ | Same implementation |
| Callbacks | ✅ | ❌ | Removed for simplicity |
| CSV logging | ✅ | ❌ | Removed for simplicity |
| ID tracking | ✅ | ❌ | Not needed for simple case |
| Ensemble support | ✅ | ❌ | Single teacher only |
| Sparse logprobs | ✅ | ❌ | Full logits only |
| WandB integration | ✅ | ❌ | Moved to main script |

## Testing

The refactored trainer maintains compatibility with:
- FSDP model parallelization
- DistributedSampler for data parallelization
- Mixed precision (bfloat16)
- Checkpoint resume functionality
- Multi-GPU training with proper synchronization

## Future Extensions

To extend this trainer:

1. **Add new trainer types**: Subclass `Trainer` and implement `compute_loss()`
2. **Add logging**: Implement lightweight logging in `train_step()` or `eval_step()`
3. **Add metrics**: Extend `eval_step()` to compute additional metrics
4. **Add callbacks**: Create optional callback hooks if needed

