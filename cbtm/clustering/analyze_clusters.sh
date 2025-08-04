#!/bin/bash

#SBATCH --job-name=test_analyze_clusters
#SBATCH --output=/home/ehghaghi/scratch/ehghaghi/cbtm_runs/run_logs/%x_%j.out
#SBATCH --error=/home/ehghaghi/scratch/ehghaghi/cbtm_runs/run_logs/%x_%j.err
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --account=aip-craffel
#SBATCH --time=23:00:00

echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU resources: $CUDA_VISIBLE_DEVICES"

# ----------------------------
# Module & environment setup
# ----------------------------

module load cuda/12.6
module load gcc arrow/19.0.1 python/3.11

# Check what's loaded
echo "Loaded modules:"
module list

# Activate your Python environment
source /home/ehghaghi/projects/aip-craffel/ehghaghi/c-btm-distillation/uv-x86_64-unknown-linux-gnu/.venv/bin/activate

# Verify Python environment
echo "Python path: $(which python)"
echo "Testing numpy import..."
python -c "import numpy; print(f'numpy {numpy.__version__} works!')" || echo "numpy failed!"


# ----------------------------
# Environment variables
# ----------------------------

export SCRATCH="/home/ehghaghi/scratch/ehghaghi"
export HUGGINGFACE_HUB_CACHE="$SCRATCH/.cache"
export OUTPUT_DIR="$SCRATCH/results"

# ----------------------------
# Experiment variables
# ----------------------------

NUM_CLUSTERS=64
DATASET="allenai/tulu-3-sft-mixture"
SAMPLE_SIZE=10000
KMEANS_DIR="$SCRATCH/clusters"

# ----------------------------
# Run clustering script
# ----------------------------

echo "Running clustering with ${NUM_CLUSTERS} clusters on ${SAMPLE_SIZE} samples..."
python -u -m analyze_clusters \
  --dataset-name "${DATASET}" \
  --num-clusters "${NUM_CLUSTERS}" \
  --sample-size "${SAMPLE_SIZE}" \
  --output-dir "${KMEANS_DIR}/${DATASET}/${NUM_CLUSTERS}/" 


echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) completed at $(date)"