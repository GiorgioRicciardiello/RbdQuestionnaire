"""
SELECT: ML ACTIGRAPHY


Select the best model given the stages specificaitons as the models are intetented to be selected as
a series of diagnostic tests

    stages = {
        'first_stage': {
            'sens': 90,
            'spc': 60
        },
        'second_stage': {
            'sens':60,
            'spc': 90
        }
    }

"""

import pandas as pd
from typing import Tuple, Dict, Optional
from pathlib import Path


def _get_actig_met_pred(dir_path: Path, modality: str = 'avg') -> Tuple[pd.DataFrame, pd.DataFrame]:
    if modality == 'avg':
        file_name_metrics = r'metrics_avg_scores.csv'
        file_name_pred = r'predictions_avg_scores.csv'
    else:
        file_name_metrics = r'metrics_majority_voting.csv'
        file_name_pred = r'predictions_majority_voting.csv'

    # get predictions
    df_pred_actigraphy = pd.read_csv(dir_path.joinpath('per_subject', file_name_pred))
    df_metrics_actigraphy = pd.read_csv(dir_path.joinpath('per_subject', file_name_metrics))
    return df_metrics_actigraphy, df_pred_actigraphy

# Get best second stage
def _get_best_actigraphy_config(df, stages, stage='second_stage'):
    """
    Selects best actigraphy model config based on sensitivity and specificity thresholds.

    Parameters:
    - df: DataFrame with actigraphy metrics
    - stages: dict with thresholds
    - stage: 'second_stage'

    Returns:
    - DataFrame with best configuration(s)
    """
    sens_thr = stages[stage]['sens']
    spec_thr = stages[stage]['spc']

    valid_configs = []

    for opt in df['optimization'].unique():
        for thr_type in df['threshold_type'].unique():
            subset = df[
                (df['optimization'] == opt) &
                (df['threshold_type'] == thr_type)
                ]

            meets_criteria = subset[
                (subset['sensitivity'] >= sens_thr) &
                # (subset['sensitivity'] < 100) &
                (subset['specificity'] >= spec_thr)
                # (subset['specificity'] < 100)
                ]

            if not meets_criteria.empty:
                valid_configs.append({
                    'model_type': 'xgboost',  # fixed in your data
                    'optimization': opt,
                    'threshold_type': thr_type,
                    'mean_sens': meets_criteria['sensitivity'].mean(),
                    'mean_spec': meets_criteria['specificity'].mean()
                })

    df_valid = pd.DataFrame(valid_configs)
    if df_valid.empty:
        return pd.DataFrame()

    df_valid['score_sum'] = df_valid['mean_sens'] + df_valid['mean_spec']
    return df_valid.sort_values(by='score_sum', ascending=False)


def _get_predictions_for_actigraphy_config(df_pred, best_config_row):
    """
    Retrieves predictions for best actigraphy config.

    Parameters:
    - df_pred: DataFrame with actigraphy predictions
    - best_config_row: Series with best config info

    Returns:
    - DataFrame with predictions
    """
    opt = best_config_row['optimization'].values[0]
    thr_type = best_config_row['threshold_type'].values[0]
    pred_col = f'y_pred_{thr_type}'.replace('youden_j', 'youden')  # e.g. y_pred_youden
    thresh_col = f'thr_{thr_type}'.replace('youden_j', 'youden')

    df_filtered = df_pred[df_pred['optimization'] == opt]
    df_filtered = df_filtered.rename(columns={'y_pred': 'y_score'})
    return df_filtered[['subject_id', 'y_true', 'y_score', pred_col, thresh_col]].rename(columns={pred_col: 'y_pred_acitg'})


def _get_most_actigraphy_config(df,
                                stages:Optional[Dict] = None,
                                stage='second_stage',
                                criteria:str='specificity') -> pd.DataFrame:
    """
    Selects the actigraphy model config with the highest specificity < 100%
    that meets sensitivity and specificity thresholds.

    Parameters:
    - df: DataFrame with actigraphy metrics
    - stages: dict with thresholds
    - stage: 'second_stage'

    Returns:
    - DataFrame with best configuration(s) sorted by specificity
    """
    if stages:
        sens_thr = stages[stage].get('sens')
        spec_thr = stages[stage].get('spc')
    else:
        sens_thr = None
        spec_thr = None

    if criteria == 'specificity':
        df_best = df.loc[df[criteria] > spec_thr,]
        if df_best.empty:
            raise ValueError(f'Actigraphy no metrics in specificity are > than {spec_thr}')

        df_best = df_best[df_best['sensitivity'] == df_best['sensitivity'].max()]

    elif criteria == 'sensitivity':
        df_best = df.loc[df[criteria] > sens_thr,]
        if df_best.empty:
            raise ValueError(f'Actigraphy no metrics in specificity are > than {spec_thr}')

        df_best = df_best[df_best['specificity'] == df_best['sensitivity'].max()]
    elif criteria == 'auc':
        df_best = df.loc[df[criteria] == df[criteria].max(),]
        if stage == 'first_stage':
            df_best = df_best[df_best['sensitivity'] == df_best['sensitivity'].max]
        elif stage == 'second_stage':
            df_best = df_best[df_best['specificity'] == df_best['specificity'].max]

    elif  criteria == 'youden':
        df_best = df.loc[df[criteria] == df[criteria].max(),]
    else:
        raise ValueError(f'Invalid criteria: {criteria}')

    df_best['model_type'] = 'xgboost'
    return df_best


def filter_actig_ml(path_acti_pred:Path,
                    stages:Dict[str, Dict[str, int]] = None,
                    stage='second_stage',
                    modality: str = 'avg',
                    criteria: str = 'specificity',) -> Tuple[pd.DataFrame,pd.DataFrame, pd.DataFrame]:

    df_metrics_actig, df_predictions_actig = _get_actig_met_pred(dir_path=path_acti_pred, modality=modality)

    best_config_actig = _get_most_actigraphy_config(df=df_metrics_actig,
                                                    stages=stages,
                                                    stage=stage,
                                                    criteria=criteria)

    df_preds_actig_best = _get_predictions_for_actigraphy_config(df_pred=df_predictions_actig,
                                                                 best_config_row=best_config_actig)


    return df_predictions_actig, df_preds_actig_best, best_config_actig