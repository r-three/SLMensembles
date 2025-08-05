#!/bin/bash

#SBATCH --job-name=test_distillkit
#SBATCH --output=/home/ehghaghi/scratch/ehghaghi/distilkit_runs/run_logs/%x_%j.out
#SBATCH --error=/home/ehghaghi/scratch/ehghaghi/distilkit_runs/run_logs/%x_%j.err
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --account=aip-craffel
#SBATCH --time=23:00:00

# srun -c 4 --gres=gpu:2 --partition l40 --mem=128GB --pty --time=16:00:00 bash

echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU resources: $CUDA_VISIBLE_DEVICES"

echo "Running on node: $(hostname)"
nvidia-smi -q -d ECC

# Load modules or activate environment if needed
module load cuda/12.6
module load gcc arrow/19.0.1 python/3.11
source /home/ehghaghi/projects/aip-craffel/ehghaghi/c-btm-distillation/uv-x86_64-unknown-linux-gnu/distill_env/bin/activate  # Replace with your actual environment path

# Export cache dirs
export SCRATCH="/home/ehghaghi/scratch/ehghaghi"
export N_CLUSTERS=8
export OUTPUT_DIR=$SCRATCH/distillation_results/$N_CLUSTERS
export HUGGINGFACE_HUB_CACHE=$SCRATCH/.cache
export WANDB_DIR="$SCRATCH/wandb"
export TMPDIR="$SCRATCH/tmp"
mkdir -p $SCRATCH/wandb $SCRATCH/tmp

export WANDB_API_KEY=""
export WANDB_ENTITY="raffel-reports"
# export WANDB_MODE=offline
# export WANDB_DISABLE_SERVICE=true
# export WANDB_DISABLED=true
# export WANDB_PROJECT="SLMEnsembles"

# Try different wandb versions with pip
# echo "Installing wandb..."
# pip uninstall -y wandb
# pip install wandb==0.18.0

# Run your Python distillation script
python -u distil_logits.py