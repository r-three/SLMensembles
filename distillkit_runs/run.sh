#!/bin/bash

#SBATCH --job-name=test_distillkit
#SBATCH --output=/home/ehghaghi/scratch/ehghaghi/distilkit_runs/run_logs/%x_%j.out
#SBATCH --error=/home/ehghaghi/scratch/ehghaghi/distilkit_runs/run_logs/%x_%j.err
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --account=aip-craffel
#SBATCH --time=23:00:00

# srun -c 4 --gres=gpu:2 --partition l40 --mem=10GB --pty --time=16:00:00 bash

echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU resources: $CUDA_VISIBLE_DEVICES"

# Load modules or activate environment if needed
module load cuda/12.6
module load gcc arrow/19.0.1 python/3.11
source /home/ehghaghi/projects/aip-craffel/ehghaghi/SLMensembles/DistillKit/myenv/bin/activate  # Replace with your actual virtual environment path

# Export cache dirs
export SCRATCH="/home/ehghaghi/scratch/ehghaghi"
export OUTPUT_DIR=$SCRATCH/results
export HUGGINGFACE_HUB_CACHE=$SCRATCH/.cache
export WANDB_DIR="$SCRATCH/wandb"
export TMPDIR="$SCRATCH/tmp"
mkdir -p $SCRATCH/wandb $SCRATCH/tmp

export WANDB_API_KEY=""
export WANDB_ENTITY="raffel-reports"


# Run your Python distillation script
python -u distil_logits.py