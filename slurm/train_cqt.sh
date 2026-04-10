#!/bin/bash --login
#SBATCH --job-name=autoeq_cqt
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=v100:1
#SBATCH --output=logs/cqt_%j.out
#SBATCH --error=logs/cqt_%j.err

cd /mnt/scratch/meadowm1/AutoEQ

module purge
module load Miniforge3
module load CUDA/12.4.0
conda activate autoeq

mkdir -p logs checkpoints
python train.py --data data/processed_cqt_combined --run cqt_combined

