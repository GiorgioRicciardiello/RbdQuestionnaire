#!/usr/bin/env python
import os
import argparse
from pathlib import Path

import optuna
import pandas as pd

from config.config import config  # your existing config
from library.ml_actigraphy.training import run_nested_cv_with_optuna_parallel

# keep your list; shortened here for space — use your full APPROVED_FEATURES
APPROVED_FEATURES = [
    "TST","WASO","SE","On","Off","AI10","AI10_w","AI10_REM","AI10_REM_w",
    "AI10_NREM","AI10_NREM_w","AI30","AI30_w","AI30_REM","AI30_REM_w",
    "AI30_NREM","AI30_NREM_w","AI60","AI60_w","AI60_REM","AI60_REM_w",
    "AI60_NREM","AI60_NREM_w","TA0.5","TA0.5_w","TA0.5_REM","TA0.5_REM_w",
    "TA0.5_NREM","TA0.5_NREM_w","TA1","TA1_w","TA1_REM","TA1_REM_w",
    "TA1_NREM","TA1_NREM_w","TA1.5","TA1.5_w","TA1.5_REM","TA1.5_REM_w",
    "TA1.5_NREM","TA1.5_NREM_w","SIB0","SIB0_w","SIB0_REM","SIB0_REM_w",
    "SIB0_NREM","SIB0_NREM_w","SIB1","SIB1_w","SIB1_REM","SIB1_REM_w",
    "SIB1_NREM","SIB1_NREM_w","SIB5","SIB5_w","SIB5_REM","SIB5_REM_w",
    "SIB5_NREM","SIB5_NREM_w","LIB60","LIB60_w","LIB60_REM","LIB60_REM_w",
    "LIB60_NREM","LIB60_NREM_w","LIB120","LIB120_w","LIB120_REM",
    "LIB120_REM_w","LIB120_NREM","LIB120_NREM_w","LIB300","LIB300_w",
    "LIB300_REM","LIB300_REM_w","LIB300_NREM","LIB300_NREM_w","MMAS",
    "MMAS_w","MMAS_REM","MMAS_REM_w","MMAS_NREM","MMAS_NREM_w","T_avg",
    "T_avg_w","T_avg_REM","T_avg_REM_w","T_avg_NREM","T_avg_NREM_w",
    "T_std","T_std_w","T_std_REM","T_std_REM_w","T_std_NREM","T_std_NREM_w",
    "HP_A_ac","HP_A_ac_w","HP_A_ac_REM","HP_A_ac_REM_w","HP_A_ac_NREM",
    "HP_A_ac_NREM_w","HP_M_ac","HP_M_ac_w","HP_M_ac_REM","HP_M_ac_REM_w",
    "HP_M_ac_NREM","HP_M_ac_NREM_w","HP_C_ac","HP_C_ac_w","HP_C_ac_REM",
    "HP_C_ac_REM_w","HP_C_ac_NREM","HP_C_ac_NREM_w"
]

def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--night", type=int, required=True, help="Night index (1..N).")
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--outer-splits", type=int, default=10)
    p.add_argument("--inner-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--maximize", choices=["sens","spec"], default="sens")
    p.add_argument("--target", type=str, default="label")
    p.add_argument("--use-gpu", action="store_true")
    p.add_argument("--cpus", type=int, default=None)
    args = p.parse_args()

    # Resolve CPUs (LSF exposes LSB_DJOB_NUMPROC)
    cpus = args.cpus or env_int("LSB_DJOB_NUMPROC", os.cpu_count() or 1)

    # Cap BLAS/OMP threads to avoid oversubscription inside the task
    os.environ["OMP_NUM_THREADS"]      = str(cpus)
    os.environ["MKL_NUM_THREADS"]      = str(cpus)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpus)
    os.environ["NUMEXPR_NUM_THREADS"]  = str(cpus)

    # Reduce Optuna chatter
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Load data from your config
    df_path = config["data_path"]["pp_actig"]
    df = pd.read_csv(df_path)
    target = args.target

    out_dir = args.results_root / f"nights_{args.night}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[night {args.night}] host={os.uname().nodename} cpus={cpus} gpu={os.environ.get('CUDA_VISIBLE_DEVICES','')}")
    print(f"[night {args.night}] df shape = {df.shape} -> {out_dir}")

    # IMPORTANT:
    # - Inside run_nested_cv_with_optuna_parallel:
    #   * set study.optimize(..., n_jobs=1)
    #   * pass n_jobs=cpus into XGBoost via your _get_model_and_space
    #   * enable device='cuda' when args.use_gpu is True (or tree_method='gpu_hist' for XGB 1.x)

    sampler = optuna.samplers.TPESampler(seed=args.seed + args.night)

    df_metrics, df_predictions, df_inner = run_nested_cv_with_optuna_parallel(
        df=df,
        target_col=target,
        continuous_cols=None,
        feature_cols=APPROVED_FEATURES,
        col_id="subject_id",
        model_types=["xgboost"],
        random_seed=args.seed,
        n_outer_splits=args.outer_splits,
        n_inner_splits=args.inner_splits,
        n_trials=args.trials,
        study_sampler=sampler,
        pos_weight=True,
        min_sens=0.6,
        min_spec=0.6,
        maximize=args.maximize,
        results_dir=out_dir,
        # Ensure your model factory (_get_model_and_space) reads n_jobs=cpus and use_gpu
        # e.g., _get_model_and_space(..., n_jobs=cpus, use_gpu=args.use_gpu)
    )

    # You already save CSVs inside run_..., but also emit canonical names if you like:
    df_metrics.to_csv(out_dir / "metrics_sctg.csv", index=False)
    df_predictions.to_csv(out_dir / "predictions_actig.csv", index=False)
    df_inner.to_csv(out_dir / "inner_val_records.csv", index=False)

    print(f"[night {args.night}] done -> {out_dir}")

if __name__ == "__main__":
    main()
