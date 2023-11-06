#!/bin/bash
#PBS -l walltime=1:00:00
#PBS -l select=1:ncpus=12:mem=32gb
#PBS -N AMOS_organ_volumes

cd ${PBS_O_WORKDIR}

# Launch virtual environment
module load anaconda3/personal
source activate nnUNetv2

python3 getOrganVolumes.py