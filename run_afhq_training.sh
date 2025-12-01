#!/bin/bash
#SBATCH -A uppmax2025-2-346
#SBATCH -J afhq_clip_train
#SBATCH -t 12:00:00
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH -o afhq_train_%j.out
#SBATCH -e afhq_train_%j.err

# Load modules
module load python/3.11.8
module load python_ML_packages/3.11.8-gpu

# Activate your environment if you have one
# source ~/myenv/bin/activate

# Go to working directory
cd /home/junming/private/ClipBased-SyntheticImageDetection

# Activate virtual environment
source venv/bin/activate

# Run training
python train_afhq_clip.py

echo "Training completed!"
