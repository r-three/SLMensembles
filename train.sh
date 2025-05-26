#!/bin/bash

#SBATCH --job-name=slm_ensembles                                                            
#SBATCH --output=/scratch/ssd004/scratch/klambert/slm_ensembles/logs/%x_%j.out                
#SBATCH --error=/scratch/ssd004/scratch/klambert/slm_ensembles/logs/%x_%j.err 
#SBATCH --partition=a40                                                                       
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4                                                                     
#SBATCH --mem=16GB                                                  
#SBATCH --time=08:00:00 

# Project notes:
 # compute the teachers loss on the validation set
    # compute the ensemble's loss on the validation set
    # we want one loss for the teacher - how well does the teacher predict stuff
    # at the end of every training cycle, compute the ensemble's loss and see how well it performs on the validation set
    # graph the baseline teacher loss to the ensemble performance as we add more models
    # does the training and ensemble adding decrease the validation loss and the similarity to the teacher
    # two benchmarks - validation loss (how good it is at predicting instructions and similarity to the ground truth); similarity of ensemble and the teacher 
    # dump numerical results into a file and then upldoa into a notebook
    # put in a colab - ask Claude how to best store evaluation metrics
    # also maybe mark the time it takes to train and evaluate 
    # Setup wandb for this round
    
    # timeline
        # have the results which would go int the paper by the end of the internship so that I can write up a paper
        # need to see which distilation approaches work the best
        # scale up methods after we found the best ones that work
        # reinforcement learning for ensembles
        # right now trying to get some signal on what works
    # plan to run code for the night; forsight; work on the code during the day
    # have a clear goal of what I need to do next
    # what are the immediate todos are on my part?
        # have the code in running condition - if I run it trains something, and trains it correctly
        # implement the validation loss and the teacher-student-kl loss and comparison
        # evaluate what happnes with the first run when I evaluate ensemble members
        # read over my paper notes and search for other methods and approaches
        # validation loss isn't going down? hwo similar it is to the teacher? -> will tell us where to go next
    # neurips, iclr, icml, (nlp: emnlp, acl, etc.), colm (on language modeling)
    # how do you start a conference?

# Note: need to run `srun -c 4 --gres=gpu:1 --partition a40 --mem=10GB --pty --time=8:00:00 bash`

# Set up terminal output mirroring
# SLURM_JOB_ID=16262193
# SLURM_JOB_NAME='slm_ense'
# exec > >(tee -a /scratch/ssd004/scratch/klambert/slm_ensembles/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out) 2> >(tee -a /scratch/ssd004/scratch/klambert/slm_ensembles/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err >&2)

# Print a message when the job starts
echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) started at $(date)"

# Print the name of the node and job ID for debugging
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU resources: $CUDA_VISIBLE_DEVICES"

# Load modules if needed on your cluster
module load python/3.10.12 
module load cuda-12.1

# Activate the virtual environment - uncomment and adjust the path as needed
# module 
source /scratch/ssd004/scratch/klambert/slm_ensembles/venv/bin/activate
wandb login

# Run script
python -u /h/klambert/slm_ensembles/train.py
