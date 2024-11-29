#!/bin/bash
# SWBATCH DIRECTIVES
#SBATCH --partition=gpu           # Use GPU partition
#SBATCH --gres=gpu:2              # Request 1 GPU
#SBATCH --time=12:00:00           # Runtime limit
#SBATCH --job-name=train_model    # Name of the job
#SBATCH --output=train_output.log # Output logs
#SBATCH --error=train_error.log   # Error logs
SBATCH --nodelist=hal07          # Request a specific node (hal06)


# Activate your environment

# Install dependencies (if needed)
pip3 install -Uq fastai "timm==0.6.13"
pip3 install huggingface_hub

# Run your training script
python3 train_test.py --gpu_id 0 --data_root /home/abhia2/data/newData/2011_09_26

