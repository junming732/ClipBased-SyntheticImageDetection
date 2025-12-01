#!/bin/bash
#SBATCH -A uppmax2025-2-346
#SBATCH -J cifake_clip_finetune
#SBATCH -t 6:00:00
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH -o cifake_finetune_%j.out
#SBATCH -e cifake_finetune_%j.err

# Print job info
echo "=========================================="
echo "CIFAKE CLIP Fine-tuning Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Navigate to project directory
cd /domus/h1/junming/private/ClipBased-SyntheticImageDetection

# Activate virtual environment
source venv/bin/activate

# Set environment variable for dataset
export CIFAKE_DATA_PATH=/home/junming/nobackup_junming/CIFAKE

# Print Python and package versions
echo "Python version:"
python --version
echo ""
echo "PyTorch version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Run fine-tuning
echo "=========================================="
echo "Starting fine-tuning..."
echo "=========================================="

python train_cifake_clip.py

# Print completion info
echo ""
echo "=========================================="
echo "Job completed!"
echo "End time: $(date)"
echo "=========================================="

# List generated files
echo "Generated files:"
ls -lh *.pt *.csv *.png 2>/dev/null || echo "No output files found"