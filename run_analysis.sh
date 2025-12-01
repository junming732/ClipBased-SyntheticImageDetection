#!/bin/bash
#SBATCH -A uppmax2025-2-346
#SBATCH -J afhq_analysis
#SBATCH -t 03:00:00
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH -o afhq_analysis_%j.out
#SBATCH -e afhq_analysis_%j.err

# Go to working directory
cd /home/junming/private/ClipBased-SyntheticImageDetection

# Activate virtual environment
source venv/bin/activate

echo "=========================================="
echo "Starting AFHQ Analysis"
echo "=========================================="

echo ""
echo "1. Running overall performance analysis..."
echo "------------------------------------------"
python analyze_afhq_results.py

echo ""
echo "2. Running misclassification visualization..."
echo "------------------------------------------"
python visualize_misclassifications.py

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo ""
echo "Generated files:"
ls -lh *.png *.csv misclassified_images.txt 2>/dev/null | grep -E "(png|csv|txt)"