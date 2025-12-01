#!/bin/bash
#SBATCH -A uppmax2025-2-346
#SBATCH -J afhq_baselines
#SBATCH -t 08:00:00
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH -o baselines_%j.out
#SBATCH -e baselines_%j.err

# Go to working directory
cd /home/junming/private/ClipBased-SyntheticImageDetection

# Activate virtual environment
source venv/bin/activate

echo "=========================================="
echo "AFHQ Baseline Experiments"
echo "=========================================="

echo ""
echo "1. Running Zero-Shot CLIP evaluation..."
echo "------------------------------------------"
python zero_shot_afhq.py

echo ""
echo "2. Training ResNet50 baseline..."
echo "------------------------------------------"
python train_resnet_baseline.py

echo ""
echo "=========================================="
echo "Baseline Experiments Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - zero_shot_results.json"
echo "  - best_resnet_afhq.pth"
echo "  - resnet_training_results.csv"