#!/bin/bash
#PBS -l walltime=10:00:00
#PBS -l select=1:ncpus=12:mem=40gb
#PBS -N predict_900

cd ${PBS_O_WORKDIR}

# Launch virtual environment
module load anaconda3/personal
source activate nnUNetv2

## Verify install:
python -c "import torch;print(torch.cuda.is_available())"

# Set environment variables
ROOT_DIR='/rds/general/user/kc2322/home/data/AMOS_3D/'

datasets=("Dataset900_Age4" "Dataset901_Age4" "Dataset902_Age4")

export nnUNet_raw=$ROOT_DIR"nnUNet_raw"
export nnUNet_preprocessed=$ROOT_DIR"nnUNet_preprocessed"
export nnUNet_results=$ROOT_DIR"nnUNet_results"

for number in {0..2}; do
    DATASET=${datasets[number]}
    TASK=${DATASET:7:3}

    # Inference
    INPUT_FOLDER=$ROOT_DIR"nnUNet_raw/"$DATASET"/imagesTs"
    OUTPUT_FOLDER=$ROOT_DIR"inference/"$DATASET"/all"

    echo $TASK
    echo $DATASET
    echo $INPUT_FOLDER
    echo $OUTPUT_FOLDER

    #nnUNetv2_predict -i $INPUT_FOLDER -o $OUTPUT_FOLDER -d $TASK -c 3d_fullres -f all -chk checkpoint_best.pth

    # Run python script to evaluate results
    python3 processResults.py -d $DATASET
done
