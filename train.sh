#!/bin/bash

#SBATCH --job-name=slm_ensembles                                                            
#SBATCH --output=/scratch/ssd004/scratch/klambert/slm_ensembles/logs/%x_%j.out                
#SBATCH --error=/scratch/ssd004/scratch/klambert/slm_ensembles/logs/%x_%j.err 
#SBATCH --partition=a40                                                                       
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4                                                                     
#SBATCH --mem=16GB                                                  
#SBATCH --time=12:00:00 

# srun -c 4 --gres=gpu:1 --partition a40 --mem=10GB --pty --time=8:00:00 bash
# cd /scratch/ssd004/scratch/klambert/slm_ensembles/logs/

echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU resources: $CUDA_VISIBLE_DEVICES"

module load python/3.10.12 
module load cuda-12.1
source /scratch/ssd004/scratch/klambert/slm_ensembles/venv/bin/activate
wandb login

python -u /h/klambert/slm_ensembles/train.py
