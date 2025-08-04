#!/bin/bash

#SBATCH --job-name=test_cluster
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

# Activate your Python environment
source /home/ehghaghi/projects/aip-craffel/ehghaghi/c-btm-distillation/uv-x86_64-unknown-linux-gnu/.venv/bin/activate

# ----------------------------
# Environment variables
# ----------------------------

export SCRATCH="/home/ehghaghi/scratch/ehghaghi"
export HUGGINGFACE_HUB_CACHE="$SCRATCH/.cache"
export OUTPUT_DIR="$SCRATCH/results"

# ----------------------------
# Experiment variables
# ----------------------------
DATASET="allenai/tulu-3-sft-mixture"
NUM_CLUSTER=64
HF_USERNAME="Malikeh1375"
REPO_NAME="clustered_tulu_3"
SAMPLE_SIZE=939343 #FULL DATASET
MODLE_PATH="$SCRATCH/clusters/allenai/tulu-3-sft-mixture/$NUM_CLUSTER"
TEST_SIZE=0.2

# ----------------------------
# Run clustering script
# ----------------------------

echo "Running clustering with ${NUM_CLUSTERS} clusters on ${SAMPLE_SIZE} samples..."
python -u -m cluster \
  --dataset-name "${DATASET}" \
  --vectorizer-path "${MODLE_PATH}/tfidf.pkl" \
  --kmeans-path "${MODLE_PATH}/kmeans.pkl" \
  --hf-username "${HF_USERNAME}" \
  --repo-name "${REPO_NAME}_${NUM_CLUSTER}" \
  --sample-size "${SAMPLE_SIZE}" \
  --test-size "${TEST_SIZE}"\
  --output-dir "${MODEL_PATH}" \


echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) completed at $(date)"