#!/bin/bash
#BSUB -J "ml_actig_multi_night[1-15]%4"   # 15 nights, max 4 concurrent
#BSUB -P acc_vascbrain
#BSUB -q gpu
#BSUB -n 4                               # CPUs per task
#BSUB -R "rusage[mem=8000]"              # 8 GB RAM per task
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:j_exclusive=yes:mode=shared"
#BSUB -W 7:00
#BSUB -oo /sc/arion/work/riccig01/during/RbdQuestionnaire/minerva/output_errors/ml_actig_%J_%I.out
#BSUB -eo /sc/arion/work/riccig01/during/RbdQuestionnaire/minerva/output_errors/ml_actig_%J_%I.err
#BSUB -L /bin/bash

module purge


module load cuda/11.8
source activate rbd_env

# Cap BLAS threads to CPUs per task
export OMP_NUM_THREADS=${LSB_DJOB_NUMPROC}
export MKL_NUM_THREADS=${LSB_DJOB_NUMPROC}
export OPENBLAS_NUM_THREADS=${LSB_DJOB_NUMPROC}
export NUMEXPR_NUM_THREADS=${LSB_DJOB_NUMPROC}

echo "Host: $(hostname)"
echo "Night (array index): ${LSB_JOBINDEX}"
echo "CPUs: ${LSB_DJOB_NUMPROC}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
python --version

RESULTS_ROOT=/sc/arion/work/riccig01/during/RbdQuestionnaire/results/ml_actigraphy_night_sensitivity

python -u /sc/arion/work/riccig01/during/RbdQuestionnaire/ml_actigraphy_multiple_nights.py \
  --night ${LSB_JOBINDEX} \
  --results-root ${RESULTS_ROOT} \
  --cpus ${LSB_DJOB_NUMPROC} \
  --use-gpu \
  --trials 200 \
  --outer-splits 10 \
  --inner-splits 5 \
  --seed 42 \
  --maximize spec \
  --target label