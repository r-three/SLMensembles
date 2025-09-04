#!/bin/bash
#SBATCH --job-name=olmo2_teacher_logit
#SBATCH --nodes=4
#SBATCH --mem=256G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:4
#SBATCH --output=logs/olmo2_teacher_logit.%j.out
#SBATCH --error=logs/olmo2_teacher_logit.%j.err
#SBATCH --open-mode=append
#SBATCH --wait-all-nodes=1
#SBATCH --account=aip-craffel
#SBATCH --time=1-00:00:00

deactivate
module load gcc arrow/18.1.0
source /home/lfy/projects/aip-craffel/lfy/venvs/prj/bin/activate
export TORCH_HOME=/scratch/lfy/cache/prj/torch
export HF_HOME=/scratch/lfy/cache/prj/huggingface


export MASTER_ADDR="$(hostname --fqdn)"
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"
export RDVZ_ID=$RANDOM
echo "RDZV Endpoint $MASTER_ADDR:$MASTER_PORT"

export NCCL_IB_DISABLE=1  # Our cluster does not have InfiniBand. We need to disable usage using this flag.
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN

# export TORCH_DISTRIBUTED_DEBUG=DETAIL  # Uncomment these flags for debugging communication
# export TORCH_CPP_LOG_LEVEL=INFO
export LOGLEVEL=INFO
export PYTHONFAULTHANDLER=1
# export CUDA_LAUNCH_BLOCKING=0

srun -p $SLURM_JOB_PARTITION \
    -c $SLURM_CPUS_ON_NODE \
    -N $SLURM_JOB_NUM_NODES \
    --mem=256G \
    --gres=gpu:l40s:$SLURM_GPUS_ON_NODE \
    bash -c 'torchrun \
    --nproc-per-node=$SLURM_GPUS_ON_NODE \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --rdzv-endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv-id $RDVZ_ID \
    --rdzv-backend c10d \
    src/main.py --mixed-precision'