"""
SELECT: ML QUESTIONNAIRE

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
import re


def get_single_question(path_quest_raw:Path,
                        col_quest:str='q1_rbd',
                        col_subject:str='subject_id') -> pd.DataFrame:
    df_raw_quest = pd.read_csv(path_quest_raw)
    if not col_quest in df_raw_quest.columns:
        raise ValueError('The questionnaire does not contain a column named {}'.format(col_quest))

    # col_quest = [col for col in df_raw_quest.columns if col.startswith('q') ]
    df_raw_quest = df_raw_quest[[col_subject, 'diagnosis', col_quest]]
    # consistency on the target column with the predictions
    y_pred = f'y_pred_{col_quest}'
    df_raw_quest.rename(columns={'diagnosis': 'y_true',
                                 col_quest: y_pred}, inplace=True)
    df_raw_quest = df_raw_quest.loc[~df_raw_quest[y_pred].isna()]
    df_raw_quest[y_pred] = df_raw_quest[y_pred].astype(int)
    return df_raw_quest



def get_quest_met_pred(dir_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_metrics_quest = pd.read_csv(dir_path.joinpath('metrics_outer_folds_ci.csv'))
    df_predictions_quest = pd.read_csv(dir_path.joinpath('predictions_outer_folds.csv'))
    return df_metrics_quest, df_predictions_quest



def get_best_model_config(df, stages, stage='first_stage'):
    """
    Filters model configurations that meet sensitivity and specificity thresholds
    for a given stage, and returns the best one based on combined score.

    Parameters:
    - df: DataFrame containing model metrics
    - stages: dict with sensitivity and specificity thresholds
    - stage: key in stages dict ('first_stage' or 'second_stage')

    Returns:
    - DataFrame with best configuration(s)
    """
    sens_thr = stages[stage]['sens']
    spec_thr = stages[stage]['spc']

    thr_methods = [col.split('thr_at_')[1] for col in df.columns if 'thr_at_' in col]
    opt_methods = df['optimization'].unique()
    model_types = df['model_type'].unique()

    valid_configs = []

    for model in model_types:
        for opt in opt_methods:
            for thr in thr_methods:
                sens_col = f'sensitivity_at_{thr}'
                spec_col = f'specificity_at_{thr}'

                if sens_col in df.columns and spec_col in df.columns:
                    subset = df[
                        (df['model_type'] == model) &
                        (df['optimization'] == opt)
                        ]

                    meets_criteria = subset[
                        (subset[sens_col] >= sens_thr) &
                        (subset[spec_col] >= spec_thr)
                        ]

                    if not meets_criteria.empty:
                        valid_configs.append({
                            'model_type': model,
                            'optimization': opt,
                            'threshold': thr,
                            'mean_sens': meets_criteria[sens_col].mean(),
                            'mean_spec': meets_criteria[spec_col].mean()
                        })

    df_valid = pd.DataFrame(valid_configs)
    if df_valid.empty:
        return pd.DataFrame()  # No configs met criteria

    df_valid['score_sum'] = df_valid['mean_sens'] + df_valid['mean_spec']
    return df_valid.sort_values(by='score_sum', ascending=False)


def get_most_sensitive_model_config(df, stages:Dict=None,
                                    stage:str| None ='first_stage',
                                    criteria:str|None = None) -> pd.DataFrame:
    """
    Filters model configurations that meet sensitivity and specificity thresholds
    for a given stage, and returns the one with highest mean sensitivity.

    Parameters:
    - df: DataFrame containing model metrics
    - stages: dict with sensitivity and specificity thresholds
    - stage: key in stages dict ('first_stage' or 'second_stage')

    Returns:
    - DataFrame with best configuration(s) sorted by sensitivity
    """
    if stages:
        sens_thr = stages[stage]['sens']
        spec_thr = stages[stage]['spc']
    elif criteria and stages is None:
        # select from criteria
        # because we are given the metrics of each fold, we will use the ones that have the CI values
        col_metrics = [col for col in df.columns if col.endswith('_ci')]
        df_unique = df[['model_type', 'optimization'] + col_metrics]
        df_unique = df_unique.drop_duplicates(keep='first')

        # col_search = [col for col in df_unique.columns if criteria in col]
        df_unique = df_unique.loc[df_unique['model_type'] == 'xgboost', :]
        for colm in col_metrics:
            df_unique[f'{colm}_mean'.replace('_ci', '')] = df_unique[colm].apply(
                lambda s: float(re.search(r'[-+]?\d*\.\d+|\d+', s).group()) if isinstance(s, str) and re.search(
                    r'[-+]?\d*\.\d+|\d+', s) else None
            ).copy()
        # get row with highest mean metrics
        mean_cols = [col for col in df_unique.columns if col.endswith('_mean')]
        # Find the column with the highest value in each row
        df_unique['max_mean_value'] = df_unique[mean_cols].max(axis=1)
        df_unique['max_mean_metric'] = df_unique[mean_cols].idxmax(axis=1)

        # Get the row with the highest overall mean value
        best_row = df_unique.loc[df_unique['max_mean_value'].idxmax()]
        best_row = pd.DataFrame(best_row).T
        # get all the rows
        col_missing = [col for col in df.columns if not col in best_row.columns]
        best_row = pd.merge(best_row, df[col_missing], left_index=True, right_index=True ,how='left')

        return best_row

    thr_methods = [col.split('thr_at_')[1] for col in df.columns if 'thr_at_' in col]
    opt_methods = df['optimization'].unique()
    model_types = df['model_type'].unique()

    valid_configs = []

    for model in model_types:
        for opt in opt_methods:
            for thr in thr_methods:
                sens_col = f'sensitivity_at_{thr}'
                spec_col = f'specificity_at_{thr}'

                if sens_col in df.columns and spec_col in df.columns:
                    subset = df[
                        (df['model_type'] == model) &
                        (df['optimization'] == opt)
                        ]

                    meets_criteria = subset[
                        (subset[sens_col] >= sens_thr) &
                        (subset[spec_col] >= spec_thr)
                        ]

                    if not meets_criteria.empty:
                        valid_configs.append({
                            'model_type': model,
                            'optimization': opt,
                            'threshold': thr,
                            'mean_sens': meets_criteria[sens_col].mean(),
                            'mean_spec': meets_criteria[spec_col].mean()
                        })

    df_valid = pd.DataFrame(valid_configs)
    if df_valid.empty:
        return pd.DataFrame()
    df_valid = df_valid.sort_values(by='mean_sens', ascending=False)
    df_valid.reset_index(inplace=True, drop=True)
    return df_valid


def get_predictions_for_best_config(df_pred, best_config_row):
    """
    Retrieves predictions from df_pred for the best model configuration.

    Parameters:
    - df_pred: DataFrame with prediction outputs
    - best_config_row: single-row DataFrame or Series with best config info

    Returns:
    - DataFrame with matching predictions
    """
    model = best_config_row['model_type']
    opt = best_config_row['optimization']
    thr = best_config_row['threshold']
    pred_col = f'y_pred_at_{thr}'

    # Filter predictions
    df_filtered = df_pred[
        (df_pred['model_type'] == model) &
        (df_pred['optimization'] == opt)
        ]

    # Return only relevant columns
    return df_filtered[['subject_id', 'y_true', 'y_score', pred_col]].rename(columns={pred_col: 'y_pred_quest'})



def filter_quest_ml(path_quest_pred:Path,
                    stages:Optional[Dict[str, Dict[str, int]]] = None,
                    stage:str|None ='first_stage',
                    criteria:str='youden_j'):
    df_metrics_quest, df_pred_quest = get_quest_met_pred(dir_path=path_quest_pred)

    # best_config_quest = get_best_model_config(df_metrics_quest, stages, stage='first_stage').iloc[0]

    best_config_quest = get_most_sensitive_model_config(df=df_metrics_quest,
                                                        stages=stages,
                                                        stage=stage,
                                                        criteria=criteria)
    if best_config_quest.shape[0] > 1:
        # when selecting from stages
        best_config_quest = best_config_quest.iloc[0]
        df_preds_best_quest = get_predictions_for_best_config(df_pred=df_pred_quest, best_config_row=best_config_quest)
    else:
        # when selecting from criteria
        model_type = best_config_quest.model_type.values[0]
        opt = best_config_quest.optimization.values[0]
        df_preds_best_quest = df_pred_quest.loc[(df_pred_quest['model_type'] == model_type) &
                                (df_pred_quest['optimization'] == opt), :]

        # if we have multiple thresholds is because of the folds, we need to compute the mean
        col_thr = [col for col in df_preds_best_quest.columns if col.startswith('thr_at_')]
        for col in col_thr:
            df_preds_best_quest[col] = df_preds_best_quest[col].mean()

    return df_preds_best_quest, best_config_quest
































