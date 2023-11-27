#!/bin/bash
#PBS -l walltime=10:00:00
#PBS -l select=1:ncpus=15:mem=120gb:ngpus=1:gpu_type=RTX6000
#PBS -N predict_800

cd ${PBS_O_WORKDIR}

# Launch virtual environment
module load anaconda3/personal
source activate nnUNetv2

## Verify install:
python -c "import torch;print(torch.cuda.is_available())"

# Set environment variables
ROOT_DIR='/rds/general/user/kc2322/home/data/AMOS_3D/'

datasets=("Dataset800_Age3" "Dataset801_Age3" "Dataset802_Age3")

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
