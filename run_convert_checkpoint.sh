#!/bin/bash
#SBATCH --job-name=dcp_to_pt
#SBATCH --output=/scratch/klambert/run_logs/%x_%j.out
#SBATCH --error=/scratch/klambert/run_logs/%x_%j.err
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --account=aip-craffel
#SBATCH --time=01:00:00

# ---- user vars (edit these) ----
CHECKPOINT_DIR="/scratch/klambert/model_log/singular/new_models/checkpoints/checkpoint_epoch0_step3000"
MODEL_NAME="allenai/OLMo-2-0425-1B-SFT"   # e.g. "allenai/OLMo-2-0425-1B-SFT" or local cfg dir
OUT_PATH="/scratch/klambert/model_log/singular/ensemble_boosting_runs/first_run/model_full.pt"
OWNER_RANK=0
MASTER_PORT="${MASTER_PORT:-29500}"

# World size (defaults to 1 if SLURM doesn’t set it)
WORLD_SIZE="${SLURM_GPUS_ON_NODE:-1}"

echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) on $(hostname)"
echo "WORLD_SIZE=$WORLD_SIZE OWNER_RANK=$OWNER_RANK"

# Single-process path (fastest if you only need one rank)
if [ "$WORLD_SIZE" -eq 1 ]; then
  python dcp_to_pt.py \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --student-model-name "$MODEL_NAME" \
    --out "$OUT_PATH" \
    --owner-rank "$OWNER_RANK"
  exit $?
fi

# ---- env ----
module load gcc arrow/18.1.0
source /home/klambert/projects/aip-craffel/shared/slm_ensemble/prj/bin/activate

# Multi-GPU: spawn ranks, only OWNER_RANK actually loads+saves (subgroup inside script)
torchrun \
  --nproc_per_node="$WORLD_SIZE" \
  --master_port="$MASTER_PORT" \
  dcp_to_pt.py \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --student-model-name "$MODEL_NAME" \
    --out "$OUT_PATH" \
    --owner-rank "$OWNER_RANK"
