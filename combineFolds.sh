#!/bin/bash
#PBS -l walltime=1:00:00
#PBS -l select=1:ncpus=4:mem=8gb
#PBS -N AMOS_combine_folds

cd ${PBS_O_WORKDIR}

# Launch virtual environment
module load anaconda3/personal
source activate nnUNetv2

python3 combineFolds.py