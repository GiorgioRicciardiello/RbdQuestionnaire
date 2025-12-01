"""
Select patients from the Stanford ADRC and include them in the dataset.
"""
import pandas as pd
import optuna
from config.config import config, config_actigraphy
import numpy as np


if __name__ == "__main__":
    # %% data and path
    input_dir = config.get('data_path').get('data').joinpath('stanford_adrc')
    df_responses = pd.read_csv(input_dir.joinpath('B9Form_SharingData_ED_N215_2025-06-09.csv'))
    df_dem_diagnosis = pd.read_csv(input_dir.joinpath('SharingData_ED_N215_2025-06-09 (1).csv'))

    # %% Read the master files
    df_quest_master = pd.read_csv(config.get('data_path').get('pp_questionnaire'))
    df_actig_master = pd.read_csv(config.get('data_path').get('pp_actig'))

    # %% Column formating
    # we will need this copy for the merging
    df_dem = df_dem_diagnosis.copy()
    df_dem = df_dem.rename(columns={'file_name': 'ID', 'nacc_dx': 'label'})
    df_dem = df_dem[['ID', 'race', 'age', 'sex', 'label']]

    # 1. Based on the selection criteria, filter the ADRC subjects to include in the analysis
    col_rem_question = 'b9_berem'
    controls_lable = 'CU'
    col_diagnosis = 'nacc_dx'
    col_id = 'file_name'
    cohort_adrc = 'ADRC'
    print(f'Counts of responses for REM question {col_rem_question}: \n{df_responses[col_rem_question].value_counts(dropna=False)}:')

    df_pat_no_to_rem = df_responses.loc[~df_responses[col_rem_question].isin([1, 9])]
    print(f'Patients who answered No to REM question {col_rem_question}: {df_pat_no_to_rem.shape[0]} patients')

    # get the diagnosis of those patients
    count_diagnosis = df_dem_diagnosis[col_diagnosis].value_counts()
    print(count_diagnosis)

    df_true_controls = df_dem_diagnosis.loc[(df_dem_diagnosis[col_diagnosis] == controls_lable) &
                         (df_dem_diagnosis[col_id].isin(df_pat_no_to_rem[col_id].values))]
    print(f'Resulting controls {df_true_controls.shape[0]} patients')

    df_true_controls=df_true_controls.loc[(df_true_controls['age'] >= 40) & (df_true_controls['age'] <= 80)]

    print(f'Resulting controls with age ge 40: {df_true_controls.shape[0]} patients')
    # Use this file name to run the feature extractor from the .cwa files
    files = df_true_controls.file_name

    # trim the dem dataframe to the 78 subjects
    df_dem = df_dem[df_dem['ID'].isin(files.to_list())]
    df_dem['ID'] = df_dem.ID.str.replace('.cwa', '')
    # we should have only controls
    df_dem['label'] = df_dem['label'].map({controls_lable: 0})
    assert df_dem['label'].isna().sum() == 0

    # %% Structure the ADRC
    # Confirm we have extracted the features and included the file in the project
    print(f'======= Merging Actigraphy Dataset =======')
    if not config_actigraphy.get('raw_actigraphy_adrc_wlabels').is_file():
        # read the raw version of the ADRC actigraphy recordings
        df_actig_adrc = pd.read_csv(config_actigraphy.get('raw_adrc'))

        # df_actig_adrc['cohort'] = cohort_adrc
        # include the diagnosis columns (named label in the actigraphy data
        df_actig_adrc = pd.merge(left=df_actig_adrc,
                                      right=df_dem[['ID', 'label']],)

        df_actig_adrc = df_actig_adrc.drop(columns=[ 'Class', 'Date',])
        # generate columns to match the actigraphy dataset
        # df_actig_adrc['has_questionnare'] = 0  # these records have no questionnaire
        df_actig_adrc['site'] = 'ADRC'
        # df_actig_adrc['Date'] = np.nan  # we do not have the dite in this version
        # df_actig_adrc['good_night'] = np.nan  # we do not have this as well
        df_actig_adrc = df_actig_adrc.rename(columns={'ID': 'subject_id'})
        # lets put the night sequence, this will not be very prcise since we do not actually know it
        df_actig_adrc['night_seq'] = df_actig_adrc.groupby('subject_id').cumcount() + 1
        # sanity check
        cols_different = set(df_actig_master.columns) ^ set(df_actig_adrc.columns)
        if not len(cols_different) == 0:
            raise ValueError(f'To merge we need same columns in both: \nColumns different: {cols_different}')

        # save the pre-processing version
        df_actig_adrc.to_csv(config_actigraphy.get('raw_actigraphy_adrc_wlabels'), index=False)
        print(f"ADRC formated to the cohort and ID column names {config_actigraphy.get('raw_actigraphy_adrc_wlabels')}")
        # merging is done in the pre_processing_actigraphy.py
    else:
        df_actig_adrc = pd.read_csv(config_actigraphy.get('raw_actigraphy_adrc_wlabels'))

    # %% Filter and merge with the meta non-actigraphy table
    if not cohort_adrc in df_quest_master.cohort.unique():
        # 2. make the dem table compatible with the current structure
        df_dem['cohort'] = cohort_adrc
        df_dem['vasc_brain'] = 0
        df_dem['actig'] = 1
        df_dem['has_quest'] = 0
        df_dem['data_set'] = cohort_adrc
        df_dem['sex'] = df_dem['sex'].map({'Male': 1, 'Female': 0})

        # race levels
        nan_race = df_dem.race.isna().sum()  # 0

        levels_valid = df_quest_master.race.unique()
        levels_current = df_dem.race.unique()

        race_mapper = {
            'Asian': 'Asian',
            'White': 'White',
            'Unknown/ Not Reported': np.nan,
            'More Than One Race': 'Mixed',
            'Black or African American':  'Black or African American',
            'American Indian/ Alaska Native': 'Other',
        }
        expected_nans = (df_dem['race'] == 'Unknown/ Not Reported').sum()
        df_dem['race'] = df_dem['race'].map(race_mapper)

        assert nan_race == expected_nans

        # code the same as the original
        race_to_num = (
            df_quest_master[["race", "race_num"]]
            .dropna()
            .drop_duplicates()
            .set_index("race")["race_num"]
            .to_dict()
        )

        df_dem['race_num'] = df_dem['race'].map(race_to_num)

        assert df_dem['race_num'].isna().sum() ==  df_dem['race'].isna().sum()

        df_dem.rename(columns={'ID': 'subject_id', 'label': 'diagnosis', 'sex': 'gender'}, inplace=True)

        # now we need to include the actigraphy measures
        col_actig = ['TST', 'WASO', 'SE', 'T_avg', 'nw_night']
        df_actig_mean = (
            df_actig_adrc
            .groupby("ID")[col_actig]
            .mean()
            .reset_index()
        )
        df_dem_actig = pd.merge(left=df_dem,
                          right=df_actig_mean,
                          left_on='subject_id',
                          right_on='ID',)
        df_dem_actig.drop(columns=['ID'], inplace=True)

        df_master_combined = pd.concat([df_quest_master, df_dem_actig], axis=0)
        df_master_combined = df_master_combined.reset_index(drop=True)
        df_master_combined.to_csv(config.get('data_path').get('pp_questionnaire'))
        print(f'\nADRC cohort included in the questionnaire dataset')
    else:
        print(f'\nADRC already exists the questionnaire dataset:')




















