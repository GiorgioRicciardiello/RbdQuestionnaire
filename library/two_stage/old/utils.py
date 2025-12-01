import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
from config.config import config, config_actigraphy


def merge_prediction_sources(
        df_actig_pred: pd.DataFrame,
        df_quest: pd.DataFrame,
        df_quest_pred: pd.DataFrame, ) -> pd.DataFrame:
    """

    :param df_actig_pred: Actigraphy prediction dataframe
        ['subject_id', 'label', 'score', 'prediction_05', 'prediction_youden_innerCV']

    :param df_quest: questionnaire data sources
            ['subject_id', 'master_id', 'data_set', 'diagnosis', 'age', 'gender',
           'bmi', 'race', 'race_num', 'ethnicity', 'ethnicity_num',
           'other_neuro_sleep_diagnosis', 'q1_rbd', 'q2_smell', 'q4_constipation',
           'q5_orthostasis', 'TST', 'WASO', 'SE', 'T_avg', 'nw_night', 'actig',
           'has_quest', 'vasc_brain', 'cohort']

    :param df_quest_pred: dataframe with the predictions of the questionnaire model
        ['outer_fold', 'ID', 'y_pred_at_tau_inner', 'y_true', 'tau_inner_youden'],
    :return:
    """
    # merge the questionnaire predictions with the actigraphy
    df_actig_pred_quest = pd.merge(
        left=df_actig_pred,
        right=df_quest[
            ['subject_id', 'master_id', 'age', 'data_set', 'q1_rbd', 'q2_smell', 'q4_constipation',
             'q5_orthostasis']],
        on='subject_id',
        how='inner',
    )

    # keep only records with questionnaire data
    df_actig_pred_quest = df_actig_pred_quest.loc[~df_actig_pred_quest['q1_rbd'].isna()]
    assert df_actig_pred_quest.subject_id.nunique() == 133


    actig_pred_quest_pred = pd.merge(
        left=df_actig_pred_quest,
        right=df_quest_pred[['subject_id', 'y_pred_quest']],
        on='subject_id',
        how='left',
    )
    assert actig_pred_quest_pred.master_id.nunique() == 133
    assert actig_pred_quest_pred.subject_id.nunique() == 133
    assert actig_pred_quest_pred.subject_id.isna().sum() == 0

    return actig_pred_quest_pred


# ─────────────────────────────────────────────
# DATA LOADING & CLEANING
# ─────────────────────────────────────────────
def load_and_clean_questionnaire(path: Path) -> pd.DataFrame:
    """Load questionnaire CSV and rename/select columns."""
    df = (
        pd.read_csv(path)
        .rename(columns={
            "Subject ID": "subj_id",
            "Diagnosis": "q_truth",
            "ml classification": "q_pred"
        })
        .loc[:, ["subj_id", "q_truth", "q_pred"]]
    )
    df["subj_id"] = df["subj_id"].astype(str).str.strip().str.replace("-", "", regex=False)
    return df


def load_and_clean_actigraphy(path: Path) -> pd.DataFrame:
    """Load actigraphy CSV and rename/select columns."""
    df = (
        pd.read_csv(path)
        .rename(columns={
            "subject_id": "subj_id",
            "label": "a_truth",
            "prediction": "a_pred"
        })
        .loc[:, ["subj_id", "a_truth", "a_pred"]]
    )
    df["subj_id"] = df["subj_id"].astype(str).str.strip().str.replace("-", "", regex=False)
    return df


def merge_datasets(q_df: pd.DataFrame, a_df: pd.DataFrame) -> pd.DataFrame:
    """Merge questionnaire and actigraphy data on subj_id."""
    merged = pd.merge(q_df, a_df, on="subj_id", how="inner")
    print(f"🔗 Overlapping subjects: {len(merged)}")
    return merged

# ─────────────────────────────────────────────
# STAGING & METRICS
# ─────────────────────────────────────────────
def _assign_stages(df: pd.DataFrame,
                  col_pred_questionnaire: str = 'q_pred',
                  col_pred_actigraphy: str = 'a_pred') -> pd.DataFrame:
    """
    Two stage classification is computed in this function
    Assign screening stage labels based on predictions.
    :param df:
    :param col_pred_questionnaire:
    :param col_pred_actigraphy:
    :return:
    """
    conds = [
        (df[col_pred_questionnaire] == 1) & (df[col_pred_actigraphy] == 1),
        (df[col_pred_questionnaire] == 1) & (df[col_pred_actigraphy] == 0),
        (df[col_pred_questionnaire] == 0) & (df[col_pred_actigraphy] == 1),
        (df[col_pred_questionnaire] == 0) & (df[col_pred_actigraphy] == 0),
    ]
    labels = [
        "Confirmed iRBD",
        "Questionnaire mimic",
        "Actigraphy rescue",
        "Confirmed control"
    ]
    df["stage"] = np.select(conds, labels, default="Unknown")
    return df

#
# def compute_metrics_with_two_stages(df_merged: pd.DataFrame,
#                          col_pred_questionnaire: str = 'q_pred',
#                          col_pred_actigraphy: str = 'a_pred',
#                         col_true: str = 'label',
#                         prevalence:float=0.015,
#                          ) -> pd.DataFrame:
#     """
#     Compute the metrics of the predictions. It compares the questionnaire model results, the actigraphy model results,
#     and the two stage results (combining the questionnaire model and the actigraphy model)
#
#     :param df_merged: dataset containing the predictions of questionnaire, actigraphy and two stages alongside the
#         y_true
#     :param col_pred_questionnaire: columns  in the df_merged that contains the predictions of the questionnaire model
#     :param col_pred_actigraphy: columns in the df_merged that contains the predictions of the actigraphy model
#     :param col_true: true labels
#     :param prevalence: float, prevalence of the population
#     :return:
#     """
#     def _compute_confusion_metrics(true: pd.Series,
#                                   pred: pd.Series,
#                                    prevalence: Optional[float] = 0.015,
#                                    ) -> Dict[str, float]:
#         """Compute confusion matrix counts and metrics."""
#         TP = ((true == 1) & (pred == 1)).sum()
#         FP = ((true == 0) & (pred == 1)).sum()
#         FN = ((true == 1) & (pred == 0)).sum()
#         TN = ((true == 0) & (pred == 0)).sum()
#         sens = TP / (TP + FN) if (TP + FN) else np.nan
#         spec = TN / (TN + FP) if (TN + FP) else np.nan
#         ppv_prev = (sens * prevalence) / (sens * prevalence + (1 - spec) * (1 - prevalence))
#         npv_prev = (spec * (1 - prevalence)) / (spec * (1 - prevalence) + (1 - sens) * prevalence)
#
#         return {'TP':TP,
#                 'FP':FP,
#                 'FN': FN,
#                 'TN': TN,
#                 'sens':round(sens*100, 3),
#                 'spec':round(spec*100, 3),
#                 'ppv_adjusted':round(ppv_prev*100, 3),
#                 'npv_adjusted':round(npv_prev*100, 3),}
#
#     # assign the stages column, when stages is Confirmed iRBD then we have a 1 and 1 in both models
#     df_merged_stages = _assign_stages(df=df_merged,
#                                       col_pred_questionnaire=col_pred_questionnaire,
#                                       col_pred_actigraphy=col_pred_actigraphy)
#     # for the two stages we will only use that are confirmed iRBD for both models
#     two_stage = df_merged_stages['stage'].map({'Confirmed iRBD': 1}).fillna(0).astype(int)
#
#     # for majority voting
#     df_majority = (
#         df_merged_stages
#         .groupby("subject_id")[col_pred_actigraphy]
#         .apply(lambda x: 1 if (x.sum() > (len(x) - x.sum())) else 0)
#         .reset_index(name="subject_majority_vote")
#     )
#     df_majority = (
#         df_merged_stages
#         .drop_duplicates("subject_id")  # keep one row per subject for metadata
#         .merge(df_majority, on="subject_id", how="left")
#     )
#
#     # for proportion rule
#     proportion_rule = .20
#     df_proportion_rule = (
#         df_merged_stages
#         .groupby("subject_id")[col_pred_actigraphy]
#         .apply(lambda x: 1 if (x.mean() >= proportion_rule) else 0)  # mean = proportion of 1s
#         .reset_index(name="proportion_rule")
#     )
#     df_proportion_rule = (
#         df_merged_stages
#         .drop_duplicates("subject_id")
#         .merge(df_proportion_rule, on="subject_id", how="left")
#     )
#
#
#     rows = []
#     # Questionnaire model: one row per subject
#     df_first_night = df_merged_stages.drop_duplicates(subset="subject_id", keep="first")
#     metrics = _compute_confusion_metrics(true=df_first_night[col_true],
#                                          pred=df_first_night[col_pred_questionnaire])
#     metrics["Approach"] = "Questionnaire"
#     rows.append(metrics)
#
#     # Actigraphy model: per-night predictions
#     metrics = _compute_confusion_metrics(true=df_merged_stages[col_true],
#                                          pred=df_merged_stages[col_pred_actigraphy])
#     metrics["Approach"] = "Actigraphy"
#     rows.append(metrics)
#
#     # Actigraphy model: Majority voting predictions
#     metrics = _compute_confusion_metrics(true=df_majority[col_true],
#                                          pred=df_majority['subject_majority_vote'])
#     metrics["Approach"] = "Actigraphy Majority Vote"
#     rows.append(metrics)
#
#     # Actigraphy model: Proportion Rule
#     metrics = _compute_confusion_metrics(true=df_proportion_rule[col_true],
#                                          pred=df_proportion_rule['proportion_rule'])
#     metrics["Approach"] = f"Actigraphy Proportion Rule (>{proportion_rule*100}%)"
#     rows.append(metrics)
#
#     # Two-stage model: heuristic combination
#     metrics = _compute_confusion_metrics(true=df_merged_stages[col_true],
#                                          pred=two_stage)
#     metrics["Approach"] = "Q→A staged"
#     rows.append(metrics)
#
#     return pd.DataFrame(rows).set_index("Approach")


def compute_metrics_with_two_stages(
    df_merged: pd.DataFrame,
    col_pred_questionnaire: str = "q_pred",
    col_pred_actigraphy: str = "a_pred",
    col_question_single: str = "q_single",
    col_true: str = "label",
    prevalence: float = 0.015,
    output_dir: Path | None = None

) -> pd.DataFrame:
    """Compute metrics for questionnaire-only, actigraphy-only, and staged combos."""

    def _compute_confusion_metrics(true, pred, prevalence=0.015):
        TP = ((true == 1) & (pred == 1)).sum()
        FP = ((true == 0) & (pred == 1)).sum()
        FN = ((true == 1) & (pred == 0)).sum()
        TN = ((true == 0) & (pred == 0)).sum()
        sens = TP / (TP + FN) if (TP + FN) else np.nan
        spec = TN / (TN + FP) if (TN + FP) else np.nan
        acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) else np.nan
        ppv_prev = (sens * prevalence) / (sens * prevalence + (1 - spec) * (1 - prevalence))
        npv_prev = (spec * (1 - prevalence)) / (spec * (1 - prevalence) + (1 - sens) * prevalence)
        return {
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "sens": round(sens * 100, 3),
            "spec": round(spec * 100, 3),
            "acc": round(acc * 100, 3),
            "ppv_adjusted": round(ppv_prev * 100, 3),
            "npv_adjusted": round(npv_prev * 100, 3),
            'N':true.shape[0]
        }

    output_dir.mkdir(parents=True, exist_ok=True
                     )
    rows = []
    # --- Questionnaire: single item
    df_first = df_merged.drop_duplicates("subject_id", keep="first")
    metrics = _compute_confusion_metrics(true=df_first[col_true],
                                         pred=df_first[col_question_single],
                                         prevalence=prevalence)
    metrics["Approach"] = "Questionnaire"
    rows.append(metrics)

    # --- Questionnaire: ML
    metrics = _compute_confusion_metrics(true=df_first[col_true],
                                         pred=df_first[col_pred_questionnaire],
                                         prevalence=prevalence)
    metrics["Approach"] = "Questionnaire ML"
    rows.append(metrics)

    # --- Actigraphy: nightly
    metrics = _compute_confusion_metrics(true=df_merged[col_true],
                                         pred=df_merged[col_pred_actigraphy],
                                         prevalence=prevalence)
    metrics["Approach"] = "Actigraphy (nightly)"
    rows.append(metrics)

    # --- Actigraphy: majority vote
    df_majority = (
        df_merged.groupby("subject_id")[col_pred_actigraphy]
        .apply(lambda x: 1 if x.mean() >= 0.5 else 0)
        .reset_index(name="maj_vote")
    )
    df_majority = df_first.merge(df_majority, on="subject_id", how="left")
    metrics = _compute_confusion_metrics(true=df_majority[col_true],
                                         pred=df_majority["maj_vote"],
                                         prevalence=prevalence)
    metrics["Approach"] = "Actigraphy Majority"
    rows.append(metrics)

    # --- Staged: single question AND actigraphy
    staged_single = ((df_first[col_question_single] == 1) & (df_majority["maj_vote"] == 1)).astype(int)
    metrics = _compute_confusion_metrics(true=df_first[col_true],
                                         pred=staged_single,
                                         prevalence=prevalence)
    metrics["Approach"] = "Quest AND Actigraphy"
    rows.append(metrics)

    # --- Staged: single question OR actigraphy
    staged_single = ((df_first[col_question_single] == 1) | (df_majority["maj_vote"] == 1)).astype(int)
    metrics = _compute_confusion_metrics(true=df_first[col_true],
                                         pred=staged_single,
                                         prevalence=prevalence)
    metrics["Approach"] = "Quest OR Actigraphy"
    rows.append(metrics)


    # --- Staged: ML questionnaire AND actigraphy
    staged_ml = ((df_first[col_pred_questionnaire] == 1) & (df_majority["maj_vote"] == 1)).astype(int)
    metrics = _compute_confusion_metrics(true=df_first[col_true],
                                         pred=staged_ml,
                                         prevalence=prevalence)
    metrics["Approach"] = "MLQ AND Actigraphy"
    rows.append(metrics)

    # --- Staged: ML questionnaire OR actigraphy
    staged_ml = ((df_first[col_pred_questionnaire] == 1) | (df_majority["maj_vote"] == 1)).astype(int)
    metrics = _compute_confusion_metrics(true=df_first[col_true],
                                         pred=staged_ml,
                                         prevalence=prevalence)
    metrics["Approach"] = "MLQ OR Actigraphy"
    rows.append(metrics)

    df_row = pd.DataFrame(rows).set_index("Approach")

    # plot
    _plot_screening_results(df_row, output_dir=output_dir)

    if output_dir:
        df_row.to_csv(output_dir/"TwoStageComparison.csv", index=True)

    return pd.DataFrame(rows).set_index("Approach")

# %%
from typing import Union

def get_prediction_model(df: Union[pd.DataFrame, Path] = None,
                         model_name: str = 'youden_j',
                         scoring_strategy: str = 'auc',
                         y_true: str = 'y_true',
                         col_pred: str = 'y_pred_at_tau_inner',
                         col_tau: str = 'tau_inner_youden',
                         col_fold: str = 'outer_fold',
                         col_subject_id: str = 'subject_id',
                         ) -> pd.DataFrame:
    """
    Get the dataset across all the outer folds.
    """
    if isinstance(df, Path):
        df = pd.read_csv(df)

    mask = ((df['model_type'] == model_name) &
            (df['scoring_strategy'] == scoring_strategy))

    df_slice = df.loc[mask, [col_fold, col_subject_id, col_pred, y_true, col_tau]]
    df_slice.rename(columns={col_pred: 'y_pred_quest', }, inplace=True)

    return df_slice


# %% visualizations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def _plot_screening_results(df_row,
                           x_col="spec",
                           y_col="sens",
                           color_col="acc",
                           output_dir: Path | None = None, ):
    """
    Scatter plot of screening approaches (Sensitivity vs Specificity).
    Intuitive markers: open for single modalities, filled for combinations.
    Combined legend (Single vs Two-stage) placed outside at bottom.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Marker mapping
    marker_map = {
        "Questionnaire": ("o", "none"),
        "Questionnaire ML": ("s", "none"),
        "Actigraphy (nightly)": ("^", "none"),
        "Actigraphy Majority": ("D", "none"),
        "Quest AND Actigraphy": ("o", "black"),
        "Quest OR Actigraphy": ("s", "black"),
        "MLQ AND Actigraphy": ("^", "black"),
        "MLQ OR Actigraphy": ("D", "black"),
    }

    # Desired order
    single_order = ["Questionnaire", "Questionnaire ML",
                    "Actigraphy (nightly)", "Actigraphy Majority"]
    combo_order = ["Quest AND Actigraphy", "Quest OR Actigraphy",
                   "MLQ AND Actigraphy", "MLQ OR Actigraphy"]

    handles_map = {}
    for approach, row in df_row.iterrows():
        marker, facecolor = marker_map.get(approach, ("o", "black"))
        sc = ax.scatter(
            row[x_col], row[y_col],
            s=180,
            c=row[color_col],
            cmap="viridis",
            vmin=df_row[color_col].min(),
            vmax=df_row[color_col].max(),
            marker=marker,
            edgecolors="black",
            linewidth=1.2,
            alpha=0.9,
            facecolors=facecolor,
            label=f"{approach} (Acc={row[color_col]:.1f}%)"
        )
        handles_map[approach] = sc

    # Labels and formatting
    ax.set_xlabel("Specificity (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Sensitivity (%)", fontsize=12, fontweight="bold")
    ax.set_title("Performance of Screening Approaches", fontsize=14, fontweight="bold")
    ax.set_xlim(80, 101)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Reference diagonal
    ax.plot([80, 100], [80, 100], ls="--", color="gray", alpha=0.7)

    # Colorbar
    cbar = plt.colorbar(list(handles_map.values())[0], ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Accuracy (%)", fontsize=11)

    # Dummy handles for group titles
    single_patch = mpatches.Patch(color="none", label="Single methods", alpha=0)
    combo_patch = mpatches.Patch(color="none", label="Two-stage methods", alpha=0)

    # Full legend list
    legend_handles = [single_patch] + \
                     [handles_map[a] for a in single_order if a in handles_map] + \
                     [combo_patch] + \
                     [handles_map[a] for a in combo_order if a in handles_map]

    legend_labels = ["Single methods"] + \
                    [f"{a} (Acc={df_row.loc[a, color_col]:.1f}%)" for a in single_order if a in handles_map] + \
                    ["Two-stage methods"] + \
                    [f"{a} (Acc={df_row.loc[a, color_col]:.1f}%)" for a in combo_order if a in handles_map]

    # Legend outside bottom
    ax.legend(legend_handles, legend_labels,
              loc="upper center", bbox_to_anchor=(0.5, -0.25),
              ncol=2, fontsize=9, frameon=True)

    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir/"TwoStageScreeningResults.png", dpi=300, bbox_inches="tight")
    plt.show()

