import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
import pandas as pd
from config.config import config
import ast
import re
from typing import List, Dict, Union, Any, Tuple, Optional
from sklearn.experimental import enable_iterative_imputer  # needed to use IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge


def visualize_table(df: pd.DataFrame, group_by: List[str]) -> pd.DataFrame:
    """
    Count the unique pair combinations in a dataframe within the grouped by columns.
    :param df: Input DataFrame
    :param group_by: List of column names to group by
    :return: DataFrame showing counts of unique combinations
    """
    df_copy = df.copy()
    print("Distribution before modification:")

    # Only fill NaN with 'NaN' in object (string) columns to avoid dtype issues
    df_plot_before = df_copy.copy()
    for col in df_plot_before.select_dtypes(include='object'):
        df_plot_before[col] = df_plot_before[col].fillna('NaN')

    grouped_counts_before = df_plot_before.groupby(group_by).size().reset_index(name='Counts')

    print(tabulate(grouped_counts_before, headers='keys', tablefmt='grid'))
    print(f'Remaining Rows: {df_copy.shape[0]}')
    return grouped_counts_before



# %% Main
if __name__ == "__main__":
    df_raw = pd.read_excel(config.get('data_path').get('raw_questionnaire'))

    # %% format column names
    df_raw.columns = [
        re.sub(r'[^\w]+', '_', col.strip().lower()).strip('_')
        for col in df_raw.columns
    ]

    # %% map variables
    mappers = {
        'diagnosis': {
            'Control': 0,
            'iRBD': 1
        },
        'gender': {
            'M': 1,
            'F': 0
        },
        'dataset': {
            'May Clinic Data': 'May',
            'April Cinic Data': 'April',
            'Clinic - Clean Data': 'Clinic',
            'SHAS': 'SHAS',
            'Stanford': 'Stanford',
        }
    }

    df_raw['diagnosis'] = df_raw['diagnosis'].map(mappers['diagnosis'])
    df_raw['gender'] = df_raw['gender'].map(mappers['gender'])
    df_raw['data_set'] = df_raw['data_set'].map(mappers['dataset'])

    #%% expand the other disease columns (optional, we need to clean the strings first, similar names)
    # # Step 1: Normalize the text (optional but recommended)
    # df_raw['other_neuro_sleep_diagnosis'] = df_raw['other_neuro_sleep_diagnosis'].str.lower().str.strip()
    #
    # # Step 2: Extract all unique diagnoses
    # # Split each string into a list
    # split_diag = df_raw['other_neuro_sleep_diagnosis'].str.split(',\s*')
    #
    # # Flatten and get unique labels
    # unique_diags = sorted(set(d for sublist in split_diag.dropna() for d in sublist if d))
    #
    # # Step 3: Create binary columns
    # for diag in unique_diags:
    #     col_name = f'other_diag_{diag.replace(" ", "_")}'
    #     df_raw[col_name] = df_raw['other_neuro_sleep_diagnosis'].apply(lambda x: diag in x if isinstance(x, str) else False)
    #
    df_raw.drop(columns=['actigraphy_prediction_score',
                         'other_neuro_sleep_diagnosis'],
                inplace=True)

    # %% sanity checks
    col_q = [col for col in df_raw.columns if col.startswith('q')]
    visualize_table(df_raw, ['data_set', 'diagnosis'])

    # %%
    df_raw.reset_index(drop=True, inplace=True)

    df_raw.to_csv(config.get('data_path').get('pp_questionnaire'), index=False)













