#!/bin/bash

# Simple single-node multi-GPU training script for teacher-student distillation

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Adjust based on your GPU availability
export WORLD_SIZE=4  # Number of GPUs

# Run training
torchrun \
    --nproc_per_node=$WORLD_SIZE \
    --master_port=29500 \
    src/main_simple.py \
    --mixed-precision
