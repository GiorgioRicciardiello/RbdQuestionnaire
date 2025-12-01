#!/bin/bash
#BSUB -J copy_cwa
#BSUB -P acc_vascbrain
#BSUB -q long
#BSUB -n 8
#BSUB -R "rusage[mem=16000]"
#BSUB -R "span[hosts=1]"
#BSUB -W 48:00
#BSUB -oo /sc/arion/projects/sleeplab/ActigraphyUKBB/analysis_giocrm/minerva/outputs/copy_cwa.out
#BSUB -eo /sc/arion/projects/sleeplab/ActigraphyUKBB/analysis_giocrm/minerva/outputs/copy_cwa.err
#BSUB -L /bin/bash

# Load modules
module purge


# Activate existing conda environment
source activate rbd_env

# Debug info
echo "Running on $(hostname)"
echo "CPUs: $LSB_DJOB_NUMPROC"
python --version

# Run script
python -u /sc/arion/projects/sleeplab/ActigraphyUKBB/analysis_giocrm/src/copy_cwa_files.py
