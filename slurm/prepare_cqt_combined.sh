#!/bin/bash --login
#SBATCH --job-name=prep_cqt_combined
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/prep_cqt_combined_%j.out
#SBATCH --error=logs/prep_cqt_combined_%j.err

cd /mnt/scratch/meadowm1/AutoEQ

module purge
module load Miniforge3
conda activate autoeq

mkdir -p logs
python data/prepare_data.py --mode cqt --dataset combined --workers 16
