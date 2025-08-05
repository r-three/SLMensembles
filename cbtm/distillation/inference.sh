#!/bin/bash

#SBATCH --job-name=test_distillkit
#SBATCH --output=/home/ehghaghi/scratch/ehghaghi/inference/run_logs/%x_%j.out
#SBATCH --error=/home/ehghaghi/scratch/ehghaghi/inference/run_logs/%x_%j.err
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --account=aip-craffel
#SBATCH --time=23:00:00

# srun -c 4 --gres=gpu:2 --partition l40 --mem=128GB --pty --time=16:00:00 bash

echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU resources: $CUDA_VISIBLE_DEVICES"

# Load modules or activate environment if needed
module load cuda/12.6
module load gcc arrow/19.0.1 python/3.11
source /home/ehghaghi/projects/aip-craffel/ehghaghi/c-btm-distillation/uv-x86_64-unknown-linux-gnu/distill_env/bin/activate  # Replace with your actual environment path

# Export cache dirs
export SCRATCH="/home/ehghaghi/scratch/ehghaghi"
export HUGGINGFACE_HUB_CACHE=$SCRATCH/.cache


# Configuration - UPDATE THESE VALUES
MODEL_PATH="$SCRATCH/distillation_results/8_config1_coding_25K/checkpoint-500"
TEST_INPUT="Write a Python function to calculate fibonacci numbers recursively."

echo "🚀 Testing Fine-tuned Model"
echo "=========================="
echo "📂 Model: $MODEL_PATH"
echo "📝 Input: $TEST_INPUT"
echo ""

# Run the inference script
python3 inference.py \
  --model_path "$MODEL_PATH" \
  --input "$TEST_INPUT"

echo ""
echo "✅ Test completed!"



# # Check your current setup
# sbatch run_cbtm.sh check

# # Generate text with C-BTM ensemble
# sbatch run_cbtm.sh infer "Solve this differential equation" 0.1 4 50

# # Just show ensemble weights (no generation)
# sbatch run_cbtm.sh weights "Explain machine learning"

# # Run comprehensive analysis
# sbatch run_cbtm.sh