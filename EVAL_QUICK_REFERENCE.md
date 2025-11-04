# Quick Reference: Evaluating Checkpoints

## Three Commands You Need

```bash
# 1. Evaluate training checkpoint (directory)
python src_simple/simple_eval.py \
    --model_path outputs/checkpoints/checkpoint_epoch0_step5000

# 2. Evaluate final model (.pt file)
python src_simple/simple_eval.py \
    --model_path outputs/final_model/model.pt

# 3. Evaluate HuggingFace model
python src_simple/simple_eval.py \
    --model_name allenai/OLMo-2-1B
```

## Quick Examples

### Find latest checkpoint and evaluate
```bash
LATEST=$(ls -t outputs/checkpoints/ | head -1)
python src_simple/simple_eval.py --model_path outputs/checkpoints/$LATEST
```

### Evaluate all checkpoints
```bash
for ckpt in outputs/checkpoints/checkpoint_*; do
    python src_simple/simple_eval.py --model_path $ckpt
done
```

### Compare baseline vs final
```bash
echo "=== Baseline ===" && \
python src_simple/simple_eval.py --model_name allenai/OLMo-2-1B && \
echo "=== Trained ===" && \
python src_simple/simple_eval.py --model_path outputs/final_model/model.pt
```

## What Gets Printed

```
Loading from checkpoint: outputs/checkpoints/checkpoint_epoch0_step5000
Detected distributed checkpoint format
✓ Loaded checkpoint - Epoch: 0, Step: 5000
Checkpoint info: Epoch 0, Step 5000, Loss 2.3456

Evaluating model...
Model: outputs/checkpoints/checkpoint_epoch0_step5000
Dataset: allenai/tulu-v2-sft-mixture
Test examples: 1000
Batches processed: 125
Cross-Entropy Loss: 2.3456
Perplexity: 10.44
```

## Key Points

✅ Works on **single GPU** (no torchrun needed)
✅ **Auto-detects** format (directory vs file)
✅ **No changes** needed to your training code
✅ Shows **training metadata** (epoch, step, loss)

## See Also

- Full guide: `CHECKPOINT_EVALUATION_GUIDE.md`
- Checkpoint format details: `simple_checkpoint.py`

