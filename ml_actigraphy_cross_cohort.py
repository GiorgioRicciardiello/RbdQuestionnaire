# main.py
from config.config import config_actigraphy, config
from typing import List, Tuple
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
# Local imports from your library
# from library.ml_actigraphy.training import train_nested_cv_xgb_optuna, train_nested_cv_xgb_optuna_parallel_cpu_only, run_nested_cv_with_optuna_parallel
from library.ml_actigraphy.training import  run_nested_cv_with_optuna_parallel
from library.ml_actigraphy.evaluation import  compute_feature_importance
import optuna

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
    # %% data and path
    df = pd.read_csv(config.get('data_path').get('pp_actig'))
    target = 'label'
    print(f'Dataset fo dimension: {df.shape}')
    # %% Include the demographic features
    df_quest = pd.read_csv(config.get('data_path').get('pp_questionnaire'))
    df_quest = df_quest.loc[df_quest['actig'] == 1, :]

    df = pd.merge(left=df,
                  right=df_quest[['subject_id', 'age', 'bmi', 'gender', 'cohort']],
                  on='subject_id',
                  how='left',
                  )

    APPROVED_FEATURES = APPROVED_FEATURES + ['age', 'gender', 'bmi']

    # define the continuous features to normalize
    feat_cont = {f"{feat}": df[feat].max() for feat in APPROVED_FEATURES}
    features_to_normalize  = [feat for feat, max_ in feat_cont.items() if max_ > 1]

    # %% output path
    output_dir = config.get('results_path').get('results').joinpath(f'ml_actigraphy_pos_weight_yes_dem_yes')
    # df.index = df['subject_id
    random_seed = 42  # Random seed for reproducibility

    n_outer_splits = 10
    n_inner_splits = 5
    n_trials = 250

    model_types = ["xgboost"]
    # --- Preprocessing Pipeline ---
    # preprocessor = ColumnTransformer(
    #     transformers=[("num", SimpleImputer(strategy="median"), APPROVED_FEATURES)],
    #     remainder="drop"
    # )

    optuna_sampler = optuna.samplers.TPESampler()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if not(output_dir.joinpath('predictions_actig.csv').exists() and
           output_dir.joinpath('metrics_actg.csv').exists()):
        # --- Training ---
        print("Starting nested CV training...")

        (df_metrics,
         df_predictions,
         df_inner_val_records) =  run_nested_cv_with_optuna_parallel(
            df=df,
            target_col=target,
            continuous_cols=None,
            feature_cols=APPROVED_FEATURES,
            col_id='subject_id',  # this will be dropped from the data matrix
            model_types=model_types,
            n_outer_splits=n_outer_splits,
            n_inner_splits=n_inner_splits,
            n_trials=n_trials,
            pos_weight=False,
            outer_use_es=True,
            outer_es_val_frac=0.20,
            study_sampler=optuna_sampler,
            maximize="spec",
            results_dir=output_dir,
        )


    else:
        df_metrics = pd.read_csv(output_dir / "metrics_outer_folds.csv")
        df_predictions = pd.read_csv(output_dir / "predictions_outer_folds.csv",)
        df_inner_val_records = pd.read_csv(output_dir / "inner_val_records.csv")








