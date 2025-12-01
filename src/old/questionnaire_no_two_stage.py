from config.config import config
import pandas as pd
from library.questionnaire.ml_models import (compute_cross_validation,
                                             models,
                                             )
from library.questionnaire.plot_functions import (plot_elastic_net_model_coefficients,
                                                  plot_model_performance_grid,
                                                  plot_dcurves_per_fold, multi_ppv_plot_combined,
                                                  multi_calibration_plot)



if __name__ == '__main__':
    # %% Read data
    df_data = pd.read_csv(config.get('data_path').get('pp_questionnaire'))

    # %% Cohort selection (questionnaire only, questionnaire and actigrapht)
    df_data = df_data[df_data['has_quest'] == 1]
    print(f'Selecting only quest + actig {df_data.shape[0]}')

    # %% output paths
    TEST = True
    OVERWRITE = True
    PLOT_VEN = False
    base_path = config.get('results_path').get('results').joinpath(f'metrics_questionnaire')

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


    # %% Select columns and drop columns with nans
    target = 'diagnosis'

    categorical_var = [col for col in df_data.columns if col.startswith('q')]
    categorical_var.append('gender')

    continuous_var = ['age']
    columns = list(set(categorical_var + continuous_var + [target]))

    df_data = df_data.reindex(sorted(df_data.columns), axis=1)
    print(f'Dataset dimension: {df_data.shape}')

    # %% configuration of different models to run
    configurations = {
        "Full Questionnaire": {
            'features': [col for col in columns if col != target],
            'target': target,
        },

        'Questionnaire Only': {
            'features': [col for col in columns if not col in ['age', 'gender', 'bmi', target]],
            'target': target,
        },

        'Demographics': {
            'features': ['age', 'gender', 'bmi'],
            'target': target,
        },
    }

    # %% run the analysis if the files do not exist or it overwrite is set to True
    if not (path_avg_metrics_config.is_file() and path_classifications_config.is_file()) or OVERWRITE:
        # Accumulators across all configurations
        df_fold_classifications = pd.DataFrame()
        df_all_metrics_fold    = pd.DataFrame()
        df_all_feature_imp     = pd.DataFrame()
        df_all_elastic_preds   = pd.DataFrame()
        df_all_full_metrics    = pd.DataFrame()
        df_all_full_classif    = pd.DataFrame()
        df_all_full_elastic    = pd.DataFrame()
        df_all_model_probs     = pd.DataFrame()
        df_kfold_model_probs    = pd.DataFrame()
        k = 5  # number of folds
        for conf_name, conf_values in configurations.items():
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
                features=conf_values['features'],
                target=conf_values['target'],
                k=k,
                continuous_features=['age'] if 'age' in conf_values['features'] else None,
                bias_estimate_threshold=True

            )

            # Collect the k-fold cross validation for each feature set
            df_avg_metrics_fold['config'] = conf_name
            df_all_metrics_fold = pd.concat([df_all_metrics_fold, df_avg_metrics_fold], ignore_index=True)

            df_classifications_fold['config'] = conf_name
            df_fold_classifications = pd.concat([df_fold_classifications, df_classifications_fold], ignore_index=True)

            df_elastic_params_fold['config'] = conf_name
            df_all_feature_imp = pd.concat([df_all_feature_imp, df_elastic_params_fold], ignore_index=True)

            df_elastic_preds_fold['config'] = conf_name
            df_all_elastic_preds = pd.concat([df_all_elastic_preds, df_elastic_preds_fold], ignore_index=True)

            # Combine the fold & full‐data probability DataFrames
            # df_full_model_probs['config']      = conf_name
            df_fold_model_probs_fold['config']      = conf_name
            df_kfold_model_probs = pd.concat(
                [df_kfold_model_probs, df_fold_model_probs_fold],
                ignore_index=True
            )


            # df_combined_probs = pd.concat(
            #     [df_fold_model_probs_fold, df_full_model_probs],
            #     ignore_index=True
            # )
            # df_all_model_probs = pd.concat([df_all_model_probs, df_combined_probs], ignore_index=True)

            # Collect full‐data results
            # df_full_metrics['config']         = conf_name
            # df_all_full_metrics = pd.concat([df_all_full_metrics, df_full_metrics], ignore_index=True)

            # df_full_classifications['config'] = conf_name
            # df_all_full_classif = pd.concat([df_all_full_classif, df_full_classifications], ignore_index=True)

            # df_full_elastic_params['config']  = conf_name
            # df_all_full_elastic = pd.concat([df_all_full_elastic, df_full_elastic_params], ignore_index=True)



        # ---------------------------
        # Post‐processing & saving
        # ---------------------------
        # replace the index with the id of the dataset
        df_fold_classifications['subject_id'] = df_fold_classifications['index'].map(df_data['subject_id'].to_dict())

        # 1) Finalize cross‐val metrics ordering
        cols = ['config'] + [col for col in df_all_metrics_fold.columns if col != 'config']
        df_all_metrics_fold = df_all_metrics_fold[cols]
        df_all_metrics_fold = df_all_metrics_fold.sort_values(
            by=['config', 'specificity'], ascending=[False, False]
        )

        # 2) Save cross‐validation results
        df_all_metrics_fold.to_csv(path_avg_metrics_config, index=False)
        df_fold_classifications.to_csv(path_classifications_config, index=False)
        df_all_feature_imp.to_csv(path_feature_importance, index=False)
        df_all_elastic_preds.to_csv(path_pred_prob_elastic, index=False)
        df_all_model_probs.to_csv(path_models_pred_prob, index=False)


        # 3) Save full‐data (no cross‐val) results
        # df_all_full_metrics.to_csv(path_full_metrics, index=False)
        # df_all_full_classif.to_csv(path_full_classifications, index=False)
        # df_all_full_elastic.to_csv(path_full_elastic_params, index=False)
    else:
        df_all_metrics_fold = pd.read_csv(path_avg_metrics_config)
        df_fold_classifications = pd.read_csv(path_classifications_config)
        df_all_feature_imp = pd.read_csv(path_feature_importance)
        df_all_elastic_preds = pd.read_csv(path_pred_prob_elastic)

    # sort the frames
    df_all_metrics_fold = df_all_metrics_fold.sort_values(by=['config', 'config', 'specificity'],
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
        'bmi': 'BMI'
    }
    df_all_feature_imp['Feature'] = df_all_feature_imp['Feature'].map(feature_name_mapper)
    plot_elastic_net_model_coefficients(df_all_feature_imp,
                                        figsize=(10,6),
                                        output_path=path_plot_best_model)

    # %% decison curve
    # Assume you have a DataFrame "results" with columns 'true_label' and 'pred_prob'.
    prevalence = 30 / 100000  # i.e., 0.0003
    for feature_set_config in df_all_elastic_preds['config'].unique():
        plot_dcurves_per_fold(df_results=df_all_elastic_preds.copy(),
                              prevalence=prevalence,
                              configuration=feature_set_config,
                              output_path=path_plot_best_model)

    # %% Prevalance plot
    best_model = 'Elastic Net'
    # all in one figure
    multi_ppv_plot_combined(df_predictions_model=df_all_elastic_preds,
                   figsize=(10,6),
                    population_prevalence = 30 / 100000,  # 0.0003
                   output_path=path_plot_best_model)

    #%% Plot AUC
    plot_model_performance_grid(df=df_fold_classifications,
                                selected_configs=df_fold_classifications.config.unique(),
                                selected_models=['Elastic Net', 'XGBoost'],
                                output_path=base_path.joinpath(f'auc_best_models_across_config.png'))
    # %% ROC curve


    # %% calibration plot
    # all in one figure
    df_brier_scores = multi_calibration_plot(df_predictions=df_all_elastic_preds,
                           model_name=best_model,
                           rows=2,
                           output_path=path_plot_best_model)


    df_grouped = df_brier_scores.groupby('config').agg(
        brier_score=('brier_score', lambda x: f"{x.mean():.4f} ({x.std():.4f})"),
        log_loss=('log_loss', lambda x: f"{x.mean():.4f} ({x.std():.4f})"),
        auc=('auc', lambda x: f"{x.mean():.4f} ({x.std():.4f})")
    ).reset_index()

    df_grouped.to_csv(path_plot_best_model.joinpath('loss_metrics_brier_auc_scores_elastic_net.csv'), index=False)

