#!/bin/bash --login
#SBATCH --job-name=prep_cqt
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/prep_cqt_%j.out
#SBATCH --error=logs/prep_cqt_%j.err

cd /mnt/scratch/meadowm1/AutoEQ

module purge
module load Miniforge3
conda activate autoeq

mkdir -p logs
python data/prepare_data.py --mode cqt

