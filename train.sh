#!/bin/bash

#SBATCH --job-name=slm_ensembles                                                      
#SBATCH --output=/scratch/ssd004/scratch/klambert/slm_ensembles/run_logs/%x_%j.out                
#SBATCH --error=/scratch/ssd004/scratch/klambert/slm_ensembles/run_logs/%x_%j.err 
#SBATCH --partition=a40                                                                       
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4                                                                     
#SBATCH --mem=16GB                                                  
#SBATCH --time=16:00:00 

# srun -c 4 --gres=gpu:1 --partition a40 --mem=10GB --pty --time=24:00:00 bash
# cd /scratch/ssd004/scratch/klambert/slm_ensembles/run_logs
# cd /scratch/ssd004/scratch/klambert/slm_ensembles/csv_logs
# cd /projects/distilling_llms/model_log

# from IPython.core.page import page
# from tabulate import tabulate
# import pandas as pd

# df = pd.read_csv("metrics.csv")
# table = tabulate(df.tail(50), headers="keys", tablefmt="fancy_grid")
# page(table)

# TODO: launch script with different alpha values
# configure the logging to be occasional so that there's no overhead
# add metadata

echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU resources: $CUDA_VISIBLE_DEVICES"

# Load modules if needed on your cluster
module load python/3.10.12 
module load cuda-12.1

# Activate the virtual environment
source /scratch/ssd004/scratch/klambert/slm_ensembles/venv/bin/activate

# Run script
python -u /h/klambert/slm_ensembles/train.py