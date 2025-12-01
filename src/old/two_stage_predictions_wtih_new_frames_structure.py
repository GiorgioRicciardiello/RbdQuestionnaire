"""
Two stage prediction combining the actihraphy results and the questionnare results
"""
import pandas as pd
from config.config import config
from library.two_stage.model_selection import filter_actig_ml
from library.two_stage.model_selection import filter_quest_ml, get_single_question
from library.two_stage.old.visualizations import plot_confusion_matrices_style_bar, plot_single_screening
from pathlib import Path


if __name__ == "__main__":
    # %%  read the input predictions
    modality = 'avg'
    path_acti_pred = config.get('results_path').get('results').joinpath(f'ml_questionnaire_minerva')
    # Select the ML Questionnaire predictions
    # path_quest_pred = config.get('results_path').get('results').joinpath('ml_questionnaire')
    path_quest_pred = config.get('results_path').get('results').joinpath('ml_actigraphy_pos_weight_no_dem_no')

    # %% output path
    output_path = config.get('results_path').get('results').joinpath(f'two_stage_updates')
    output_path.mkdir(parents=True, exist_ok=True)

    # %% ===================================================================
    # ========================== PLOT SINGLE MODELS ========================
    # ======================================================================


    # questionnaire andre
    # df_quest_andre = pd.read_csv(config.get('results_path').get('results').joinpath('ml_quest_andre\old\predictions_outer_folds.csv'))
    # df_quest_andre = df_quest_andre.loc[(df_quest_andre['model_type'] == 'random_forest') &
    #                    (df_quest_andre['scoring_strategy'] == 'youden_j')
    #                     , ['subject_id', 'y_score', 'y_pred_at_tau_inner', 'tau_inner_youden', 'y_true' ]]
    # df_quest_andre.rename(columns={'y_score': 'y_pred',
    #                                'tau_inner_youden': 'thr_opt'}, inplace=True)
    #
    #
    # plot_single_screening(
    #     df_predictions=df_quest_andre,
    #     subject_col="subject_id",
    #     class_names={0: "Control", 1: "iRBD"},  # labels for CM axes
    #     figsize=(18, 5),
    #     font_size_title=14,
    #     font_size_big_title=18,
    #     font_size_label=12,
    #     font_size_legend=10,
    #     font_size_cm=16,
    #     modality='ML Questionnaire',
    #     results_dir=None,
    # )
    # ------------------------- questionnaire------------------------------------------
    criteria = 'youden'
    df_preds_best_quest, best_config_quest = filter_quest_ml(path_quest_pred=path_quest_pred,
                                                             stages=None,
                                                             criteria=criteria,
                                                             stage=None)

    quest_ypred = [col for col in df_preds_best_quest.columns if 'y_pred' in col and criteria in col][0]
    quest_thr = [col for col in df_preds_best_quest.columns if 'thr_' in col and criteria in col  ][0]


    df_preds_best_quest.rename(columns={
                                            # quest_ypred: 'y_pred_optimized',
                                            # quest_thr: 'thr_opt',
                                            quest_ypred: 'y_pred_optimized',
                                            quest_thr: 'thr_opt',
                                            'y_score': 'y_pred'
                                        },
                                            inplace=True)
    # thresholds = {
    #     "standard": 0.5,
    #     "opt": df_avg["thr_opt"].iloc[0],
    # }
    # rename the columns for compatibility
    plot_single_screening(
        df_predictions=df_preds_best_quest,
        subject_col="subject_id",
        results_dir=None,  # optional: saves CSVs + plots here
        class_names={0: "Control", 1: "iRBD"},  # labels for CM axes
        figsize=(18, 5),
        font_size_title=14,
        font_size_big_title=18,
        font_size_label=12,
        font_size_legend=10,
        font_size_cm=16,
        modality='ML Questionnaire',
    )


    # -------------------------actigraphy------------------------------------------
    # we want the best model for youden
    _, df_preds_actig_best, best_config_actig_single = filter_actig_ml(path_acti_pred=path_acti_pred,
                                                             stages=None,
                                                             criteria='youden',
                                                            # criteria='specificity',
                                                             modality='avg',
                                                             stage=None)
    actig_ypred = [col for col in df_preds_actig_best.columns if 'y_pred' in col][0]
    actig_thr = [col for col in df_preds_actig_best.columns if 'thr_' in col ][0]


    df_preds_actig_best.rename(columns={actig_ypred: 'y_pred_optimized',
                                            actig_thr: 'thr_opt',
                                        'y_score': 'y_pred'}, inplace=True)

    plot_single_screening(
        df_predictions=df_preds_actig_best,
        subject_col="subject_id",
        results_dir=None,  # optional: saves CSVs + plots here
        class_names={0: "Control", 1: "iRBD"},  # labels for CM axes
        figsize=(18, 5),
        font_size_title=14,
        font_size_big_title=18,
        font_size_label=12,
        font_size_legend=10,
        font_size_cm=16,
        modality='ML Actigraphy at the Subject Level',
    )


    # %% =================================================================================
    # ========================== GET & PLOT TWO STAGE BEST MODELS ========================
    # ====================================================================================
    # Get best First Stage
    stages = {
        'first_stage': {
            'sens': 90, # sensitivity higher than this, get the best spc
            'spc': 60 # spec higher than this, get the best sens
        },
        'second_stage': {
            'sens':20,   # sensitivity higher than this, get the best spc
            'spc': 90  # spec higher than this, get the best sens
        }
    }
    # quetionnaire
    df_questionnaire = get_single_question(path_quest_raw=config.get('data_path').get('pp_questionnaire'),
                                           col_quest='q1_rbd',
                                            col_subject='subject_id')
    df_questionnaire.drop(columns='y_true', inplace=True)

    # questionnaire ML
    df_preds_best_quest, best_config_quest = filter_quest_ml(path_quest_pred=path_quest_pred,
                                                             stages=stages,
                                                             stage='first_stage')

    # manual selection of the questionnaire (This works do not delete)
    def get_manual_quest(opt_num:int=0, criteria_num:int=0, quest_path:Path=None) -> pd.DataFrame:
        df_predictions_quest = pd.read_csv(quest_path)
        opt = df_predictions_quest['optimization'].unique()[opt_num]
        col_ypred = ['y_pred_at_0p5', 'y_pred_at_sens_max', 'y_pred_at_youden']
        df_predictions_quest = df_predictions_quest.loc[(df_predictions_quest['model_type'] == 'xgboost') &
                                                        (df_predictions_quest['optimization'] == opt),
                                                        ['subject_id', 'y_score', 'y_true', col_ypred[criteria_num]]]
        return df_predictions_quest.rename(columns={col_ypred[criteria_num]: 'y_pred_quest'})

    for opt in [0, 1, 2]:
        for criteria in [0, 1, 2]:
            df_filter = get_manual_quest(quest_path=path_quest_pred.joinpath('predictions_outer_folds.csv'),
                             opt_num=opt,
                             criteria_num=0,
                             )

            df_preds_two_stage = pd.merge(left=df_filter[['subject_id','y_true', 'y_pred_quest']],
                                          right=df_preds_actig_best[['subject_id', 'y_pred_acitg']],
                                          on='subject_id',
                                          how='inner')
            df_preds_two_stage['serial_test'] = df_preds_two_stage['y_pred_quest'] & df_preds_two_stage['y_pred_acitg']

            plot_confusion_matrices_style_bar(df=df_preds_two_stage,
                                              y_true_col='y_true',
                                              class_names={0: "Control", 1: "iRBD"},
                                              methods=['y_pred_quest', 'y_pred_acitg', 'serial_test'],
                                              titles=['Questionnaire', 'Actigraphy', 'Serial Test'],
                                              comparison=f' {opt}- {criteria} ML Questionnaire & Actigraphy',
                                              figsize=(12, 6),
                                              output_dir=output_path)


    # actigraphy ML
    df_predictions_actig, df_preds_actig_best, best_config_actig = filter_actig_ml(path_acti_pred=path_acti_pred,
                                                             stages=stages,
                                                             # criteria='youden',
                                                            criteria='specificity',
                                                             modality='avg',
                                                             stage='second_stage')

    df_preds_two_stage = pd.merge(left=df_preds_best_quest[['subject_id','y_true', 'y_pred_quest']],
                                  right=df_preds_actig_best[['subject_id', 'y_pred_acitg']],
                                  on='subject_id',
                                  how='inner')

    df_preds_two_stage_quest_actig = pd.merge(left=df_preds_two_stage,
                                  right=df_questionnaire,
                                  on='subject_id',
                                  how='inner'
                                  )

    # %% Questionnaire ML + Actigraphy ML
    df_preds_two_stage['serial_test'] = df_preds_two_stage['y_pred_quest'] & df_preds_two_stage['y_pred_acitg']

    plot_confusion_matrices_style_bar(df=df_preds_two_stage,
                            y_true_col='y_true',
                            class_names={0: "Control", 1: "iRBD"},
                            methods=['y_pred_quest', 'y_pred_acitg', 'serial_test'],
                            titles=['Questionnaire', 'Actigraphy', 'Serial Test'],
                            comparison=f'ML Questionnaire & Actigraphy',
                            figsize=(12, 6),
                            output_dir=output_path)

    # %% Questionnaire + Actigraphy ML
    df_preds_two_stage_quest_actig['serial_test_quest'] = df_preds_two_stage_quest_actig['y_pred_q1_rbd'] & df_preds_two_stage_quest_actig['y_pred_acitg']

    plot_confusion_matrices_style_bar(df=df_preds_two_stage_quest_actig,
                                        y_true_col='y_true',
                                        class_names={0: "Control", 1: "iRBD"},
                                        methods=['y_pred_q1_rbd', 'y_pred_acitg', 'serial_test_quest'],
                                        titles=['Questionnaire', 'Actigraphy', 'Serial Test'],
                                        comparison=f'Q1 RBD & Actigraphy',
                                        figsize=(12, 6),
                                        output_dir=output_path)


    # # %% Initial models - To delete
    # df_quest_andre = pd.read_csv(config.get('results_path').get('results').joinpath('ml_quest_andre\old\predictions_outer_folds.csv'))
    # df_quest_andre = df_quest_andre.loc[(df_quest_andre['model_type'] == 'random_forest') &
    #                    (df_quest_andre['scoring_strategy'] == 'youden_j')
    #                     , ['subject_id', 'y_pred_at_tau_inner', 'tau_inner_youden', 'y_true' ]]
    #
    # df_quest_andre['y_pred_quest'] = (df_quest_andre['y_pred_at_tau_inner'] >= df_quest_andre['tau_inner_youden']).astype(int)
    # df_first = pd.merge(left=df_quest_andre[['y_true', 'subject_id', 'y_pred_quest']],
    #                     right=df_preds_actig_best[['subject_id', 'y_pred_acitg']],
    #                     on='subject_id',
    #                     how='inner')
    # df_first['serial_test'] = df_first['y_pred_quest'] & df_first['y_pred_acitg']
    # plot_confusion_matrices_style_bar(df=df_first,
    #                         y_true_col='y_true',
    #                         class_names={0: "Control", 1: "iRBD"},
    #                         methods=['y_pred_quest', 'y_pred_acitg', 'serial_test'],
    #                         titles=['Questionnaire', 'Actigraphy', 'Serial Test'],
    #                         comparison=f'ML Questionnaire Andre & Actigraphy',
    #                         figsize=(12, 6),
    #                         output_dir=Path(r'C:\Users\giorg\OneDrive - Fundacion Raices Italo Colombianas\projects\MSinai\RbdQuestionnaire\results\ml_quest_andre\old\best_model'))
    #
    #
    #









