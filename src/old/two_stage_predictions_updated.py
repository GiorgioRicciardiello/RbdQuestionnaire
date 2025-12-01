"""
Two stage prediction combining the actihraphy results and the questionnare results
"""
import pandas as pd
from config.config import config
from library.two_stage.old.two_stage import compute_two_stage_permutations, select_best_two_stage
from pathlib import Path

if __name__ == "__main__":
    # %%  Select the single-night actigraphy predictions
    modality = 'avg'
    if modality == 'avg':
        file_name = r'predictions_avg_scores.csv'
    else:
        file_name = r'predictions_majority_voting.csv'

    path_acti_pred = config.get('results_path').get('results').joinpath(f'ml_actigraphy_rmv_subjects')
    path_acti_pred_subjg = path_acti_pred.joinpath('per_subject', file_name)
    df_pred_actigraphy = pd.read_csv(path_acti_pred_subjg)


    # %% Select the ML Questionnaire predictions
    path_quest_pred = config.get('results_path').get('results').joinpath('ml_questionnaire', 'predictions_outer_folds.csv')
    df_pred_quest = pd.read_csv(path_quest_pred)

    # %% Select the questionnaire alone with no ML
    df_raw_quest = pd.read_csv( config.get('data_path').get('pp_questionnaire'))
    col_quest = [col for col in df_raw_quest.columns if col.startswith('q') ]
    col_quest.extend(['subject_id', 'diagnosis'])
    df_raw_quest = df_raw_quest[col_quest]
    # consistency on the target column with the predictions
    df_raw_quest.rename(columns={'diagnosis': 'y_true'}, inplace=True)

    # %% read the predictions
    # df_metrics_actig =
    # df_metrics_quest =
    # def select_model_based_on_reported_metrics()

    # %% define output
    out_dir = Path("../../results/two_stage/rmv_subjct")
    out_dir = out_dir.mkdir(parents=True, exist_ok=True)
    # %% compute teo stage premutation
    df_two_stage, df_ts_pivot = compute_two_stage_permutations(
        df_actig=df_pred_actigraphy,
        df_quest=df_pred_quest,
        df_raw_quest=df_raw_quest,
        subject_col="subject_id",
        col_true="y_true",
        prevalence=0.015,
        output_dir=out_dir
    )

    # %% Select the best one and plot

    # Best MLQ + Actigraphy combo (with scatter plot)
    best_mlq_combo = select_best_two_stage(df_two_stage,
                                           min_sens=80,
                                           max_spec=100,
                                           make_plot=True,
                                           use_raw_q=False,
                                           output_dir=None)

    # Best RawQ + Actigraphy combo
    best_rawq_combo = select_best_two_stage(df_two_stage,
                                            min_sens=60,
                                            use_raw_q=True)

    # %%
    def compute_youden_index(df_results: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Youden Index (Se + Sp - 1) for each row in a results DataFrame.
        Expects 'sens' and 'spec' in percentage units (0-100).

        Parameters
        ----------
        df_results : pd.DataFrame
            DataFrame containing sensitivity and specificity columns.

        Returns
        -------
        df_out : pd.DataFrame
            Original df with an extra column 'youden_index',
            and a compact summary table with mean ± std by stage.
        """
        df = df_results.copy()
        # ensure numeric
        df["sens"] = pd.to_numeric(df["sens"], errors="coerce")
        df["spec"] = pd.to_numeric(df["spec"], errors="coerce")

        # compute Youden (scaled to 0–100)
        df["youden_index"] = df["sens"] + df["spec"] - 100

        # summary table
        summary = (
            df.groupby("stage")["youden_index"]
            .agg(["mean", "std", "max", "min"])
            .round(2)
            .reset_index()
        )

        print("📊 Youden Index Summary (Se + Sp - 1):")
        for _, row in summary.iterrows():
            print(f"  {row['stage']:>10}: "
                  f"mean={row['mean']:.2f}, std={row['std']:.2f}, "
                  f"min={row['min']:.2f}, max={row['max']:.2f}")

        return df, summary


    # suppose you have df_out from compute_two_stage_permutations
    df_with_youden, youden_summary = compute_youden_index(best_mlq_combo)

    # filter only the best combination
    best_mlq_combo = select_best_two_stage(df_with_youden, min_sens=81, max_spec=100)

    # look at Youden for that subset
    best_mlq_combo[["stage", "sens", "spec", "youden_index"]]

    # %% manually take the predictions
    opt = 'sens'
    thr = 'spec'
    df_pred_act_selected= df_pred_actigraphy.loc[df_pred_actigraphy["optimization"] == 'sens',
        ['subject_id', 'y_true', 'thr_spec_max', 'y_pred_spec_max',]
    ]
    assert df_pred_act_selected.subject_id.nunique() == df_pred_act_selected.shape[0]















