"""
============================================================
Train Night Actigraphy Pipeline (LSF Array Compatible)

This script is designed to be run on Minerva (LSF scheduler)
as part of an array job, where each array index corresponds
to a different number of nights sampled per subject.

This avoids making a for loop across the nights in the .py, instead we run each night as a separate task

Why this structure?
-------------------
- Instead of looping over all nights inside Python, we map
  each array index (`$LSB_JOBINDEX`) to one `--night` value.
  This makes jobs embarrassingly parallel and lets the LSF
  scheduler manage concurrency (`[1-15]%4`).

- Each run samples up to N nights per subject at random
  (without replacement) and trains a nested CV pipeline
  with Optuna hyperparameter optimization.

- Output is saved under `results_root/nights_{N}`, ensuring
  clean separation of results across different night counts.

- Using an array job improves throughput, avoids redundant
  computation, and makes better use of Minerva resources.

Key parameters:
---------------
--night         Number of nights to sample per subject (from LSF index)
--results-root  Directory for storing results
--cpus          Number of CPUs to use (threads capped to this)
--use-gpu       Flag to enable GPU acceleration (if available)
--trials        Number of Optuna trials per inner CV
--outer-splits  Number of outer CV folds
--inner-splits  Number of inner CV folds
--seed          Random seed (e.g., offset with night for uniqueness)

Example (Minerva):
------------------
bsub -J "ml_actig_multi_night[1-15]%4" ...
python train_night.py --night ${LSB_JOBINDEX} --results-root /path/to/results ...

============================================================
"""

import argparse
from pathlib import Path
import pandas as pd
import optuna
from config.config import config
from library.ml_actigraphy.training import run_nested_cv_with_optuna_parallel

# --- Define features--
APPROVED_FEATURES = [
    "TST", "WASO", "SE", "On", "Off", "AI10", "AI10_w", "AI10_REM", "AI10_REM_w",
    "AI10_NREM", "AI10_NREM_w", "AI30", "AI30_w", "AI30_REM", "AI30_REM_w",
    "AI30_NREM", "AI30_NREM_w", "AI60", "AI60_w", "AI60_REM", "AI60_REM_w",
    "AI60_NREM", "AI60_NREM_w", "TA0.5", "TA0.5_w", "TA0.5_REM", "TA0.5_REM_w",
    "TA0.5_NREM", "TA0.5_NREM_w", "TA1", "TA1_w", "TA1_REM", "TA1_REM_w",
    "TA1_NREM", "TA1_NREM_w", "TA1.5", "TA1.5_w", "TA1.5_REM", "TA1.5_REM_w",
    "TA1.5_NREM", "TA1.5_NREM_w", "SIB0", "SIB0_w", "SIB0_REM", "SIB0_REM_w",
    "SIB0_NREM", "SIB0_NREM_w", "SIB1", "SIB1_w", "SIB1_REM", "SIB1_REM_w",
    "SIB1_NREM", "SIB1_NREM_w", "SIB5", "SIB5_w", "SIB5_REM", "SIB5_REM_w",
    "SIB5_NREM", "SIB5_NREM_w", "LIB60", "LIB60_w", "LIB60_REM", "LIB60_REM_w",
    "LIB60_NREM", "LIB60_NREM_w", "LIB120", "LIB120_w", "LIB120_REM",
    "LIB120_REM_w", "LIB120_NREM", "LIB120_NREM_w", "LIB300", "LIB300_w",
    "LIB300_REM", "LIB300_REM_w", "LIB300_NREM", "LIB300_NREM_w", "MMAS",
    "MMAS_w", "MMAS_REM", "MMAS_REM_w", "MMAS_NREM", "MMAS_NREM_w", "T_avg",
    "T_avg_w", "T_avg_REM", "T_avg_REM_w", "T_avg_NREM", "T_avg_NREM_w",
    "T_std", "T_std_w", "T_std_REM", "T_std_REM_w", "T_std_NREM", "T_std_NREM_w",
    "HP_A_ac", "HP_A_ac_w", "HP_A_ac_REM", "HP_A_ac_REM_w", "HP_A_ac_NREM",
    "HP_A_ac_NREM_w", "HP_M_ac", "HP_M_ac_w", "HP_M_ac_REM", "HP_M_ac_REM_w",
    "HP_M_ac_NREM", "HP_M_ac_NREM_w", "HP_C_ac", "HP_C_ac_w", "HP_C_ac_REM",
    "HP_C_ac_REM_w", "HP_C_ac_NREM", "HP_C_ac_NREM_w"
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--night", type=int, required=True)
    parser.add_argument("--results-root", type=str, required=True)
    parser.add_argument("--cpus", type=int, default=4)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--outer-splits", type=int, default=10)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maximize", type=str, default="spec")
    parser.add_argument("--target", type=str, default="label")
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(config.get('data_path').get('pp_actig'))
    target = args.target

    output_dir = Path(args.results_root).joinpath(f"nights_{args.night}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- sample nights per subject ---
    df_grouped = (
        df.groupby("subject_id", group_keys=False)
          .apply(lambda x: x.sample(
              n=min(args.night, len(x)),
              replace=False,
              random_state=args.seed + args.night  # reproducible but varied
          ))
    )
    print(f"Running {args.night} nights | Dimensions: {df_grouped.shape}")

    if not (output_dir.joinpath('predictions_actig.csv').exists() and
            output_dir.joinpath('metrics_sctg.csv').exists()):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        optuna_sampler = optuna.samplers.TPESampler()

        (df_metrics,
         df_predictions,
         df_inner_val_records) = run_nested_cv_with_optuna_parallel(
            df=df_grouped,
            target_col=target,
            continuous_cols=None,
            feature_cols=APPROVED_FEATURES,
            col_id="subject_id",
            model_types=["xgboost"],
            n_outer_splits=args.outer_splits,
            n_inner_splits=args.inner_splits,
            n_trials=args.trials,
            pos_weight=False,
            study_sampler=optuna_sampler,
            maximize=args.maximize,
            results_dir=output_dir,
        )
