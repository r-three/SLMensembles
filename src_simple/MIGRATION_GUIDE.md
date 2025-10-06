# Migration Guide: From Complex to Simple Distillation

This guide helps you transition from the original complex codebase to the simplified version.

## Key Changes

### 1. Main Training Script
- **Old**: `src/main.py` - 533 lines with ensemble, rounds, manifests
- **New**: `src/main_simple.py` - ~200 lines focused on core distillation

### 2. Trainer
- **Old**: `src/trainer.py` - Complex with ID tracking, callbacks, ensemble support
- **New**: `src/simple_trainer.py` - Basic distillation logic only

### 3. Configuration
- **Old**: Complex config with ensemble parameters, round settings, CSV columns
- **New**: `src/simple_config.py` - Simple dataclass with essential parameters

### 4. Removed Features

| Feature | Old Codebase | New Codebase | How to Add Back |
|---------|--------------|--------------|-----------------|
| Ensemble Models | `ModelEnsemble` class | Single teacher only | Keep teacher frozen, no ensemble |
| Round-based Training | Multiple rounds with model updates | Single training run | Run multiple times with different configs |
| ID Tracking | `AsyncLossLogger`, per-example tracking | None | Add simple dict tracking if needed |
| Manifest Files | Complex metadata tracking | None | Use simple JSON file if needed |
| Wandb Logging | Integrated | None | Add 3-4 lines in trainer |
| CSV Logging | `CSVLogger` class | None | Use Python's csv module |
| Signal Handling | SLURM signal handlers | None | Add signal.signal() if needed |
| Early Stopping | Patience-based | None | Track eval loss and break |

### 5. Equivalent Functionality

#### Running Training
```bash
# Old
torchrun --nproc_per_node=4 src/main.py --explicit-prefetching --mixed-precision

# New
torchrun --nproc_per_node=4 src/main_simple.py --mixed-precision
```

#### Checkpointing
- Old: Complex rotation with manifests
- New: Simple save/load with automatic cleanup

#### Loss Computation
- Old: Complex with ensemble, ID tracking, multiple loss types
- New: Clean hybrid loss: `α * CE + (1-α) * KL`

### 6. For Experiments

The simplified codebase is ideal for branching and experimentation:

1. **Routing Experiments**: Add a router module in the trainer
2. **Different Loss Functions**: Modify `compute_distillation_loss()`
3. **Data Selection**: Modify `get_dataset()` in utils
4. **Multi-Teacher**: Extend trainer to handle list of teachers

### 7. Performance Considerations

The simplified version maintains the same core performance optimizations:
- FSDP sharding for memory efficiency
- Mixed precision training
- Distributed data loading
- Gradient clipping

### 8. Quick Start for Migration

1. Copy your model names to `simple_config.py`
2. Adjust hyperparameters (learning rate, batch size, etc.)
3. Run with: `./run_simple_distillation.sh`
4. Monitor loss in console output
5. Add back specific features as needed

### 9. When to Use Which Version

**Use the Original** when you need:
- Ensemble distillation
- Complex experiment tracking
- Round-based iterative training
- Per-example loss tracking

**Use the Simplified** when you need:
- Clean baseline for experiments
- Quick prototyping
- Single teacher-student distillation
- Minimal overhead
