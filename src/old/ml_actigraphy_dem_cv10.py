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
from library.ml_actigraphy.evaluation import plot_threshold_summary, per_subject_evaluation, compute_feature_importance
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
                  right=df_quest[['subject_id', 'age', 'bmi', 'gender']],
                  on='subject_id',
                  how='left',
                  )

    APPROVED_FEATURES = APPROVED_FEATURES + ['age', 'gender', 'bmi']

    # define the continuous features to normalize
    feat_cont = {f"{feat}": df[feat].max() for feat in APPROVED_FEATURES}
    features_to_normalize  = [feat for feat, max_ in feat_cont.items() if max_ > 1]

    # %% output path
    output_dir = config.get('results_path').get('results').joinpath(f'ml_actigraphy_dem_cv10')
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
           output_dir.joinpath('metrics_sctg.csv').exists()):
        # --- Training ---
        print("Starting nested CV training...")

        (df_metrics,
         df_predictions,
         df_inner_val_records) =  run_nested_cv_with_optuna_parallel(
            df=df,
            target_col=target,
            continuous_cols=features_to_normalize,
            feature_cols=APPROVED_FEATURES,
            col_id='subject_id',  # this will be dropped from the data matrix
            model_types=model_types,
            n_outer_splits=n_outer_splits,
            n_inner_splits=n_inner_splits,
            n_trials=n_trials,
            pos_weight=True,
            outer_use_es=True,
            outer_es_val_frac=0.20,
            study_sampler=optuna_sampler,
            maximize="spec",
            results_dir=output_dir,
        )

        # df_metrics, df_predictions = train_nested_cv_xgb_optuna(
        #     df=df,
        #     target_col="label",
        #     feature_cols=APPROVED_FEATURES,
        #     group_col="subject_id",
        #     preprocessor=preprocessor,
        #     results_dir=output_dir,
        #     n_jobs=-1,
        #     random_seed=42,
        #     n_outer_splits=n_outer_splits,  # e.g., 10
        #     n_inner_splits=n_inner_splits,  # e.g., 5
        #     n_trials=n_trials,  # e.g., 50
        #     min_sens=0.6,
        #     maximize="spec"
        # )
        #
        # df_metrics, df_predictions = train_nested_cv_xgb_optuna_parallel_cpu_only(
        #     df=df,
        #     target_col="label",
        #     feature_cols=APPROVED_FEATURES,
        #     group_col="subject_id",
        #     preprocessor=preprocessor,
        #     results_dir=output_dir,
        #     n_jobs=-1,
        #     n_jobs_outer=16,
        #     random_seed=random_seed,
        #     n_outer_splits=n_outer_splits,  # e.g., 10
        #     n_inner_splits=n_inner_splits,  # e.g., 5
        #     n_trials=n_trials,  # e.g., 50
        #     min_sens=0.6,
        #     maximize="spec"
        # )

    else:
        df_metrics = pd.read_csv(output_dir / "metrics_outer_folds.csv")
        df_predictions = pd.read_csv(output_dir / "predictions_outer_folds.csv",)
        df_inner_val_records = pd.read_csv(output_dir / "inner_val_records.csv")

    #
    #
    # metric_cols = ['auc_score', 'prc_score', 'sensitivity', 'specificity', 'threshold_value']
    # group_cols = ["model_type", "optimization", "threshold"]
    #
    #
    #
    # # %% Model evaluation
    #
    # def plot_threshold_summary(
    #         df_metrics: pd.DataFrame | None,
    #         df_predictions: pd.DataFrame,
    #         class_names: dict[int, str],
    #         out_path: Path | None = None,
    #         font_size_title: int = 14,
    #         font_size_big_title: int = 18,
    #         font_size_label: int = 12,
    #         font_size_legend: int = 10,
    #         font_size_cm: int = 12,
    # ):
    #     """
    #     Plot ROC curves + Confusion Matrices for each optimization × threshold_type.
    #
    #     Rows = optimization (auc / youden / sens)
    #     Cols = threshold types (standard / youden_j / custom) × (ROC + CM)
    #     """
    #
    #     # ------------------------------
    #     # helpers
    #     # ------------------------------
    #     # def _scatter_threshold_on_mean(ax, sens_val, spec_val, color):
    #     #     """Scatter point based on reported sensitivity/specificity (metrics)."""
    #     #     fpr_point = 1 - spec_val
    #     #     tpr_point = sens_val
    #     #     ax.scatter(fpr_point, tpr_point, color=color, s=70,
    #     #                edgecolors="k", zorder=3)
    #
    #     def _scatter_threshold_on_mean(ax, mean_fpr, mean_tpr, spec_val, color):
    #         """Place scatter exactly on the mean ROC curve at the FPR ~ (1 - specificity)."""
    #         fpr_point = 1 - spec_val
    #         # interpolate TPR at that FPR from the mean ROC
    #         tpr_point = np.interp(fpr_point, mean_fpr, mean_tpr)
    #         ax.scatter(fpr_point, tpr_point, color=color, s=70,
    #                    edgecolors="k", zorder=3)
    #
    #     def _draw_mean_roc(ax,
    #                        y_true,
    #                        y_score,
    #                        folds,
    #                        color,
    #                        auc_val,
    #                        auc_ci,
    #                        sens_val,
    #                        sens_ci,
    #                        spec_val,
    #                        spec_ci,
    #                        thr_val,
    #                        title,
    #                        n_bootstrap: int = 1000):
    #         mean_fpr = np.linspace(0, 1, len(y_true))
    #         tprs = []
    #         for f in np.unique(folds):
    #             mask = folds == f
    #             fpr, tpr, _ = roc_curve(y_true[mask], y_score[mask])
    #             tprs.append(np.interp(mean_fpr, fpr, tpr))
    #         mean_tpr = np.mean(tprs, axis=0)
    #         std_tpr = np.std(tprs, axis=0)
    #
    #         # --- Bootstrap AUC ---
    #         mean_auc, (low_auc, high_auc) = _bootstrap_auc(
    #             y_true, y_score, n_bootstrap=n_bootstrap
    #         )
    #         auc_text = f"AUC={mean_auc:.2f} ({low_auc:.2f}, {high_auc:.2f})"
    #
    #         # --- Sens & Spec ---
    #         sens_text = f"Se={sens_ci}" if sens_ci else f"Se={sens_val:.2f}"
    #         spec_text = f"Sp={spec_ci}" if spec_ci else f"Sp={spec_val:.2f}"
    #
    #         ax.plot(mean_fpr, mean_tpr, color=color, lw=2,
    #                 label=f"{auc_text}\n{sens_text}\n{spec_text}")
    #         ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
    #                         color=color, alpha=0.2)
    #
    #         # scatter from sensitivity/specificity values
    #         _scatter_threshold_on_mean(ax=ax, mean_fpr=mean_fpr,
    #                                    mean_tpr=mean_tpr,
    #                                    spec_val=spec_val,
    #                                    color=color)
    #
    #         ax.set_title(title, fontsize=font_size_title)
    #         ax.set_xlabel("False Positive Rate", fontsize=font_size_label)
    #         ax.set_ylabel("True Positive Rate", fontsize=font_size_label)
    #         ax.legend(fontsize=font_size_legend, loc="lower right")
    #         ax.grid(True, linestyle="--", alpha=0.5)
    #
    #     def _draw_confusion_matrix(ax, cm, color, title):
    #         cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    #         cmap = sns.light_palette(color, as_cmap=True)
    #         im = ax.imshow(cm, cmap=cmap)
    #         for r in range(cm.shape[0]):
    #             for c in range(cm.shape[1]):
    #                 val, pct = cm[r, c], cm_pct[r, c]
    #                 bg_color = im.cmap(im.norm(cm[r, c]))
    #                 brightness = colors.rgb_to_hsv(bg_color[:3])[2]
    #                 text_color = "black" if brightness > 0.5 else "white"
    #                 ax.text(c, r, f"{val}\n({pct:.1f}%)",
    #                         ha="center", va="center",
    #                         fontsize=font_size_cm, color=text_color)
    #         ax.set_xticks([0, 1])
    #         ax.set_xticklabels([class_names[0], class_names[1]])
    #         ax.set_yticks([0, 1])
    #         ax.set_yticklabels([class_names[0], class_names[1]])
    #         ax.set_xlabel("Predicted", fontsize=font_size_label)
    #         ax.set_ylabel("True", fontsize=font_size_label)
    #         ax.set_title(title, fontsize=font_size_title)
    #
    #     # ------------------------------
    #     # prepare metrics if missing
    #     # ------------------------------
    #     if df_metrics is None:
    #         df_metrics = _rebuild_metrics_from_predictions(df_predictions=df_predictions,
    #                                                        maximize='sepc')
    #
    #     max_pred_col = [col for col in df_predictions.columns if 'max' in col and 'pred' in col][0]
    #     max_metric = max_pred_col.split('y_pred')[1][1:]
    #     # --- threshold mapping ---
    #     threshold_types = {
    #         "standard": "y_pred_standard",
    #         "youden_j": "y_pred_youden",
    #         f"{max_metric}": max_pred_col  # y_pred_spec_max
    #     }
    #     optims = sorted(df_predictions["optimization"].unique())
    #     n_rows, n_cols = len(optims), len(threshold_types) * 2
    #
    #     palette = sns.color_palette("tab10", len(optims))
    #     opt_colors = {o: palette[i] for i, o in enumerate(optims)}
    #
    #     counts = None
    #     if 'fold' not in df_predictions.columns:
    #         df_predictions['fold'] = -1
    #
    #         first_opt = df_predictions['optimization'].unique()[0]
    #         subjects_first_opt = df_predictions.loc[
    #             df_predictions['optimization'] == first_opt, 'subject_id'
    #         ].unique()
    #
    #         df_same_subjects = df_predictions.loc[
    #             (df_predictions['subject_id'].isin(subjects_first_opt)) &
    #             (df_predictions['optimization'] == first_opt), 'y_true']
    #         counts = pd.Series(df_same_subjects).map(class_names).value_counts()
    #
    #     fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    #     if n_rows == 1:
    #         axes = np.expand_dims(axes, axis=0)
    #
    #     # ------------------------------
    #     # main loop
    #     # ------------------------------
    #     for i, opt in enumerate(optims):
    #         color = opt_colors[opt]
    #
    #         for j, (thr_name, pred_col) in enumerate(threshold_types.items()):
    #             ax_roc = axes[i, j * 2]
    #             ax_cm = axes[i, j * 2 + 1]
    #
    #             preds_opt = df_predictions.loc[
    #                 df_predictions["optimization"] == opt,
    #                 ["fold", "y_true", "y_pred", pred_col]
    #             ]
    #             metrics_opt = df_metrics.loc[
    #                 (df_metrics["optimization"] == opt) &
    #                 (df_metrics["threshold_type"] == thr_name)
    #                 ]
    #
    #             if metrics_opt.empty:
    #                 continue
    #
    #             y_true = preds_opt["y_true"].values
    #             if counts is None:
    #                 counts = pd.Series(y_true).map(class_names).value_counts()
    #             y_score = preds_opt["y_pred"].values
    #             folds = preds_opt["fold"].values
    #
    #             auc_val = metrics_opt["auc"].mean()
    #             sens_val = metrics_opt["sensitivity"].mean() / 100.0
    #             spec_val = metrics_opt["specificity"].mean() / 100.0
    #             thr_val = metrics_opt["threshold"].mean()
    #
    #             auc_ci = metrics_opt["auc_ci"].iloc[0] if "auc_ci" in metrics_opt else None
    #             sens_ci = metrics_opt["sensitivity_ci"].iloc[0] if "sensitivity_ci" in metrics_opt else None
    #             spec_ci = metrics_opt["specificity_ci"].iloc[0] if "specificity_ci" in metrics_opt else None
    #
    #             title = f"{opt.upper()} | {thr_name.replace('_', ' ').title()} \n(τ={thr_val:.2f})"
    #
    #             _draw_mean_roc(ax=ax_roc,
    #                            y_true=y_true,
    #                            y_score=y_score,
    #                            folds=folds,
    #                            color=color,
    #                            auc_val=auc_val,
    #                            auc_ci=auc_ci,
    #                            sens_val=sens_val,
    #                            sens_ci=sens_ci,
    #                            spec_val=spec_val,
    #                            spec_ci=spec_ci,
    #                            thr_val=thr_val,
    #                            title=title)
    #
    #             # ---- minimalist ROC export ----
    #             if out_path:
    #                 _plot_minimalist_roc(
    #                     y_true=y_true,
    #                     y_score=y_score,
    #                     folds=folds,
    #                     model=opt,
    #                     avg_type=thr_name,
    #                     color="#A9C9FF",
    #                     out_path=out_path,
    #                     auc_val=auc_val,
    #                     sens_val=sens_val,
    #                     spec_val=spec_val,
    #                     auc_ci=auc_ci,
    #                     sens_ci=sens_ci,
    #                     spec_ci=spec_ci,
    #                     thr_val=thr_val
    #                 )
    #
    #             y_pred_bin = preds_opt[pred_col].values
    #             cm = confusion_matrix(y_true, y_pred_bin, labels=[0, 1])
    #             _draw_confusion_matrix(ax_cm, cm, color, title)
    #
    #     # ------------------------------
    #     # Layout
    #     # ------------------------------
    #     big_title = "Class distribution: " + ", ".join([f"{cls}={cnt}" for cls, cnt in counts.items()])
    #     fig.suptitle(big_title, fontsize=font_size_big_title, y=0.99)
    #
    #     plt.tight_layout()
    #     if out_path:
    #         plt.savefig(out_path / f"plt_summary.png", dpi=300, bbox_inches="tight")
    #     plt.show()
    #     plt.close()
    #
    #
    # # %%
    #
    #
    # plot_threshold_summary(
    #     df_metrics=df_metrics,
    #     df_predictions=df_predictions,
    #     class_names={0: "Control", 1: "Case"},
    #     out_path=output_dir
    # )
    # %% Per subject analysis
    # results_per_subject = per_subject_evaluation(df_predictions=df_predictions,
    #                        results_dir=output_dir.joinpath('per_subject'))
    #
    # # %% Model evaluation with the subjects that will be used in the teo stage screening
    # output_dir = output_dir.joinpath('two_stage')
    # output_dir.mkdir(parents=True, exist_ok=True)
    #
    # path_quest_pred = config.get('results_path').get('results').joinpath('ml_questionnaire', 'predictions_outer_folds.csv')
    # df_pred_quest = pd.read_csv(path_quest_pred)
    # df_pred_quest_subjs = df_pred_quest.subject_id.unique()
    #
    # df_predictions = df_predictions.loc[df_predictions['subject_id'].isin(df_pred_quest_subjs)]
    #
    # plot_threshold_summary(
    #     df_metrics=df_metrics,
    #     df_predictions=df_predictions,
    #     class_names={0: "Control", 1: "Case"},
    #     out_path=output_dir
    # )
    #
    # results_per_subject = per_subject_evaluation(df_predictions=df_predictions,
    #                        results_dir=None)
    #
    # compute_feature_importance(models_dir=output_dir.joinpath('folds'),
    #                            features_names=APPROVED_FEATURES,
    #                            objective="youden",
    #                            top_n=15)


    # %% calibration analysis









