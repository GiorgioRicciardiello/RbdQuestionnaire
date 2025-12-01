import pathlib

from config.config import config, config_actigraphy
import pandas as pd
from library.questionnaire.ml_models import (compute_cross_validation,
                                             models,
                                             )
from library.questionnaire.plot_functions import (plot_elastic_net_model_coefficients,
                                                  plot_model_performance_grid,
                                                  plot_dcurves_per_fold, multi_ppv_plot_combined,
                                                  multi_calibration_plot)
import pickle
import numpy as np

def _construct_folds(df: pd.DataFrame,
                     path_actigraphy_folds: pathlib.Path,
                     features: list[str],
                     target: str,
                     output_path: pathlib.Path,):
    """
    From the folds generated in the actigraphy model, we will use the same junction of subjects in the training and
    validation fold. Prints per-fold patient counts and class distribution.
    """

    def _summary_print(folds_inner):
        """
        Summary printout per fold to count the observations and total number of unique subjects
        :param folds_inner:
        :return:
        """
        print("\nPatient count and class distribution per fold:")
        all_subjects = set()
        for idx, fold in enumerate(folds_inner, start=1):
            # IDs
            train_ids = fold['train_data'].merge(df[['study_id']], left_index=True, right_index=True)[
                'study_id'].unique()
            val_ids = fold['val_data'].merge(df[['study_id']], left_index=True, right_index=True)['study_id'].unique()
            all_subjects.update(train_ids)
            all_subjects.update(val_ids)

            # Diagnosis counts
            train_diag_counts = fold['train_labels'].value_counts().to_dict()
            val_diag_counts = fold['val_labels'].value_counts().to_dict()

            print(f"  Fold {idx}: "
                  f"Train = {len(train_ids)} patients {train_diag_counts} | "
                  f"Val = {len(val_ids)} patients {val_diag_counts}")
        print(f'Number of unique subjects: {len(all_subjects)}')
        return all_subjects

    if output_path.exists():
        with open(output_path, 'rb') as f:
            fold = pickle.load(f)
            _summary_print(folds_inner=fold)
            return fold

    with open(path_actigraphy_folds, 'rb') as f:
        folds = pickle.load(f)


    folds_questionnaire = []
    used_patients = []
    for idx, meta in folds.items():
        # get subjects in the train and validation folds
        train_subj = meta['train'].get('subject_id').unique()
        val_subj = meta['validation'].get('subject_id').unique()
        used_patients.extend(train_subj)
        used_patients.extend(val_subj)

        # construct the X matrix for train and validation
        train_data = df.loc[df['study_id'].isin(train_subj), features]
        val_data = df.loc[df['study_id'].isin(val_subj), features]

        # construct the validation matrix
        train_labels = df.loc[df['study_id'].isin(train_subj), target]
        val_labels = df.loc[df['study_id'].isin(val_subj), target]

        # sanity check
        assert train_data.shape[0] == train_labels.shape[0]
        assert val_labels.shape[0] == val_labels.shape[0]

        folds_questionnaire.append({
            'train_data': train_data,
            'val_data': val_data,
            'train_labels': train_labels,
            'val_labels': val_labels
        })

    # Remaining patients
    df_remaining = df.loc[~df['study_id'].isin(used_patients)].copy()
    remaining_ids = df_remaining['study_id'].unique()
    np.random.shuffle(remaining_ids)

    # Spread them across folds as evenly as possible
    fold_count = len(folds_questionnaire)
    for i, study_id in enumerate(remaining_ids):
        fold_idx = i % fold_count  # cycle through folds
        # Alternate between assigning to train and val to keep balance
        if (i // fold_count) % 2 == 0:
            # Assign to train
            folds_questionnaire[fold_idx]['train_data'] = pd.concat([
                folds_questionnaire[fold_idx]['train_data'],
                df_remaining.loc[df_remaining['study_id'] == study_id, features]
            ])
            folds_questionnaire[fold_idx]['train_labels'] = pd.concat([
                folds_questionnaire[fold_idx]['train_labels'],
                df_remaining.loc[df_remaining['study_id'] == study_id, target]
            ])
        else:
            # Assign to val
            folds_questionnaire[fold_idx]['val_data'] = pd.concat([
                folds_questionnaire[fold_idx]['val_data'],
                df_remaining.loc[df_remaining['study_id'] == study_id, features]
            ])
            folds_questionnaire[fold_idx]['val_labels'] = pd.concat([
                folds_questionnaire[fold_idx]['val_labels'],
                df_remaining.loc[df_remaining['study_id'] == study_id, target]
            ])

    all_subjects = _summary_print(folds_inner=folds_questionnaire)
    # ---- Sanity check ----
    total_unique_df = df['study_id'].nunique()
    if len(all_subjects) != total_unique_df:
        raise ValueError(f"Sanity check failed! "
                         f"Expected {total_unique_df} unique subjects, "
                         f"but found {len(all_subjects)} in folds.")
    else:
        print(f"\n✅ Sanity check passed: {len(all_subjects)} unique subjects across all folds.")

    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump(folds_questionnaire, f)

    return folds_questionnaire

if __name__ == '__main__':
    # %% Read data
    df_data = pd.read_csv(config.get('data_path').get('pp_questionnaire'))
    optimize = False
    # %% Cohort selection (questionnaire only, questionnaire and actigrapht)
    cohort = 'all_subjects'  # 'quest_actigraphy'  # 'quest_actigraphy' #  'quest_actigraphy'
    if cohort == 'quest_actigraphy':
        df_data = df_data[df_data['actig'] == 1]
        print(f'Selecting only quest + actig {df_data.shape[0]}')
    else:
        print(f'Selecting all questionnaires data {df_data.shape[0]}')

    # %% output paths
    OVERWRITE = False
    base_path = config.get('results_path').get('results').joinpath(f'TwoStageQuestionnaire_{cohort}')

    if not base_path.exists():
        base_path.mkdir(parents=True, exist_ok=True)

    # %% Define the output paths
    path_avg_metrics_config          = base_path.joinpath(f'avg_metrics_config.csv')
    path_classifications_config      = base_path.joinpath(f'classifications_config.csv')
    path_feature_importance          = base_path.joinpath(f'feature_importance.csv')
    path_pred_prob_elastic           = base_path.joinpath(f'pred_prob_elasticnet.csv')
    path_models_pred_prob            = base_path.joinpath(f'pred_prob_all_models.csv')

    # Paths for full‐dataset (no cross‐val) results
    path_full_metrics                = base_path.joinpath(f'full_dataset_metrics.csv')
    path_full_classifications        = base_path.joinpath(f'full_dataset_classifications.csv')
    path_full_elastic_params         = base_path.joinpath(f'full_dataset_elastic_params.csv')


    path_avg_paper = config.get('results_path').get('results').joinpath(f'avg_paper.csv')
    path_model_metrics = config.get('results_path').get('results').joinpath(f'model_metrics.png')
    path_plot_best_model = config.get('results_path').get('results').joinpath('best_model')
    path_plot_best_model.mkdir(parents=True, exist_ok=True)

    path_fp_fp_tn_fn_tab = config.get('results_path').get('results').joinpath('tab_fp_fp_tn_fn.csv')

    # full dataset model
    path_full_dataset_metrics = config.get('results_path').get('results').joinpath(f'full_dataset_metrics.csv')
    path_full_dataset_classifications = config.get('results_path').get('results').joinpath(f'full_elastic_params.csv')
    path_full_dataset_elastic_params = config.get('results_path').get('results').joinpath(f'full_elastic_params.csv')

    # pred and prob of full dataset concat with k-folds
    path_models_pred_pob = config.get('results_path').get('results').joinpath(f'pred_pob.csv')

    path_folds_output = config.get('results_path').get('results').joinpath(f'folds_output_{cohort}_questionnaire.pkl')
    # %% Select columns and drop columns with nans
    target = 'diagnosis'

    categorical_var = [col for col in df_data.columns if col.startswith('q')]
    categorical_var.append('gender')

    continuous_var = ['age', 'bmi']
    columns = list(set(categorical_var + continuous_var + [target]))

    df_data = df_data.reindex(sorted(df_data.columns), axis=1)
    print(f'Dataset dimension: {df_data.shape}')

    # %% configuration of different models to run
    configurations = {
        "Full Questionnaire": {
            'features': [col for col in columns if col != target],
            'target': target,
        },

    }
    features = [col for col in columns if col != target]
    conf_name = "Questionnaire"

    # %% Use the same patients in the training and validation fold
    path_actigraphy_folds = config_actigraphy.get('results_dir').joinpath(f'folds_patient_ids.pkl')
    folds = None
    k=10
    if path_actigraphy_folds.exists():
        folds = _construct_folds(df=df_data,
                                 path_actigraphy_folds=path_actigraphy_folds,
                                 features=features,
                                 target=target,
                                 output_path=path_folds_output)
        k = len(folds)
    # %% run the analysis if the files do not exist or it overwrite is set to True
    if not (path_avg_metrics_config.is_file() and path_classifications_config.is_file()) or OVERWRITE:
        print(f'Running configuration: {conf_name}')
        # 3) K‐FOLD CROSS‐VALIDATION
        (
            df_avg_metrics_fold,
            df_classifications_fold,
            df_elastic_params_fold,
            df_elastic_preds_fold,
            df_fold_model_probs_fold
        ) = compute_cross_validation(
            models=models,
            df_model=df_data,
            features=features,
            target=target,
            optimize=optimize,
            k=k,
            continuous_features=['age', 'bmi'] if 'age' in features else None,
            folds=folds
        )

        # Collect the k-fold cross validation for each feature set
        df_avg_metrics_fold['config'] = conf_name
        df_classifications_fold['config'] = conf_name
        df_elastic_params_fold['config'] = conf_name
        df_elastic_preds_fold['config'] = conf_name
        df_fold_model_probs_fold['config']      = conf_name
  
        # 2) Save cross‐validation results
        df_avg_metrics_fold.to_csv(path_avg_metrics_config, index=False)
        df_classifications_fold.to_csv(path_classifications_config, index=False)
        df_elastic_params_fold.to_csv(path_feature_importance, index=False)
        df_elastic_preds_fold.to_csv(path_pred_prob_elastic, index=False)
        df_fold_model_probs_fold.to_csv(path_models_pred_prob, index=False)

    else:
        df_avg_metrics_fold = pd.read_csv(path_avg_metrics_config)
        df_classifications_fold = pd.read_csv(path_classifications_config)
        df_elastic_params_fold = pd.read_csv(path_feature_importance)
        df_fold_model_probs_fold = pd.read_csv(path_pred_prob_elastic)

    # sort the frames
    df_avg_metrics_fold = df_avg_metrics_fold.sort_values(by=['config', 'config', 'specificity'],
                                                          ascending=[False, True, False]
                                                          )
    # %%
    #       Evaluation Best Model - Elastic net
    #       Using the true, pred, pred_prob obtained for each fold and for each feature set
    #
    # %% Feature importance plot
    feature_name_mapper = {
        'q1_rbd': 'Q1 RBD',
        'age': 'Age',
        'q4_constipation': 'Q4 Constipation',
        'q2_smell': 'Q2 Smell',
        'q5_orthostasis': 'Q5 Orthostasis',
        'gender': 'Gender',
        'bmi': 'BMI',
    }
    df_elastic_params_fold['Feature'] = df_elastic_params_fold['Feature'].map(feature_name_mapper)
    plot_elastic_net_model_coefficients(df_params=df_elastic_params_fold,
                                        figsize=(10,6),
                                        output_path=path_plot_best_model)

    # %% decison curve
    # Assume you have a DataFrame "results" with columns 'true_label' and 'pred_prob'.
    prevalence = 30 / 100000  # i.e., 0.0003
    for feature_set_config in df_fold_model_probs_fold['config'].unique():
        plot_dcurves_per_fold(df_results=df_fold_model_probs_fold.copy(),
                              prevalence=prevalence,
                              configuration=feature_set_config,
                              output_path=path_plot_best_model)

    # %% Prevalance plot
    best_model = 'Elastic Net'
    # all in one figure
    multi_ppv_plot_combined(df_predictions_model=df_fold_model_probs_fold,
                   figsize=(10,6),
                    population_prevalence = 30 / 100000,  # 0.0003
                   output_path=path_plot_best_model)

    #%% Plot AUC
    plot_model_performance_grid(df=df_classifications_fold,
                                selected_configs=df_classifications_fold.config.unique(),
                                selected_models=['Elastic Net', 'XGBoost'],
                                output_path=base_path.joinpath(f'auc_best_models_across_config.png'))
    # %% ROC curve


    # %% calibration plot
    # all in one figure
    df_brier_scores = multi_calibration_plot(df_predictions=df_fold_model_probs_fold,
                           model_name=best_model,
                           rows=2,
                           output_path=path_plot_best_model)


    df_grouped = df_brier_scores.groupby('config').agg(
        brier_score=('brier_score', lambda x: f"{x.mean():.4f} ({x.std():.4f})"),
        log_loss=('log_loss', lambda x: f"{x.mean():.4f} ({x.std():.4f})"),
        auc=('auc', lambda x: f"{x.mean():.4f} ({x.std():.4f})")
    ).reset_index()

    df_grouped.to_csv(path_plot_best_model.joinpath('loss_metrics_brier_auc_scores_elastic_net.csv'), index=False)

