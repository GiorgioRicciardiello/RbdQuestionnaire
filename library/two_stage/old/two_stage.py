
# screening_analysis.py
from pathlib import Path
from typing import Tuple, Dict, Optional
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

# ----------------------------------------------------
# Helper: confusion metrics with prevalence adjustment
# ----------------------------------------------------
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
        "N": true.shape[0]
    }


def _make_and_label(opt_a, col_a, model_q, opt_q, col_q) -> str:
    """
    Build a consistent identifier string for traceability:
    ACTI(<opt_a>-<col_a>) AND QUEST(<model_q>-<opt_q>-<col_q>)
    """
    return f"ACTI({opt_a}-{col_a}) AND QUEST({model_q}-{opt_q}-{col_q})"


def _extract_label(stage: str, model_q: str, opt_q: str, col_q: str,
                   opt_a: str, col_a: str) -> str:
    """
    Build a multi-line descriptive label for plots.

    Example outputs:
      Quest
        model: xgboost
        optimization: youden
        column: y_pred_at_0p5

      Actigraphy
        optimization: auc
        column: y_pred_standard

      Two-Stage
        ACTI(auc-y_pred_standard) AND QUEST(xgboost-youden-y_pred_at_0p5)
    """
    if stage == "quest":
        return (f"Quest\n"
                f"  model: {model_q}\n"
                f"  optimization: {opt_q}\n"
                f"  column: {col_q}")
    elif stage == "actig":
        return (f"Actigraphy\n"
                f"  optimization: {opt_a}\n"
                f"  column: {col_a}")
    elif stage == "two_stage":
        return _make_and_label(opt_a, col_a, model_q, opt_q, col_q)
    else:
        return stage



#
# def _plot_confusion_matrix_from_counts(ax, TP, FP, FN, TN, title, class_names=("Control", "Case")):
#     """Draw a confusion matrix on a given axis using counts."""
#     cm = np.array([[TN, FP],
#                    [FN, TP]])
#     cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
#
#     im = ax.imshow(cm, cmap="Blues")
#
#     for r in range(cm.shape[0]):
#         for c in range(cm.shape[1]):
#             val, pct = cm[r, c], cm_pct[r, c]
#             ax.text(c, r, f"{val}\n({pct:.1f}%)",
#                     ha="center", va="center", fontsize=10,
#                     color="white" if cm[r, c] > cm.max()/2 else "black")
#
#     ax.set_xticks([0, 1])
#     ax.set_xticklabels(class_names)
#     ax.set_yticks([0, 1])
#     ax.set_yticklabels(class_names)
#     ax.set_xlabel("Predicted")
#     ax.set_ylabel("True")
#     ax.set_title(title, fontsize=11)


def plot_two_stage_with_components(
    result: pd.DataFrame,
    combo_id: str,
    class_names: dict[int, str] = {0: "Control", 1: "iRBD"},
    save_plot: bool = False,
    output_dir: Path | None = None,
    use_raw_q: bool = False,
    font_size_title: int = 13,
    font_size_label: int = 12,
    font_size_cm: int = 11,
):
    """
    Create a 2x2 grid plot:
      - Top-left: Sensitivity vs Specificity scatter
      - Top-right: Questionnaire CM
      - Bottom-left: Actigraphy CM
      - Bottom-right: Two-Stage CM
    """

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    # color option 1
    palette = sns.color_palette("pastel", n_colors=8)
    stage_colors = {"quest": palette[0], "actig": palette[2], "two_stage": palette[1]}

    # color opn 2
    # Base palette (pick soft tones for quest & actig)
    # palette = sns.color_palette("pastel", n_colors=8)
    #
    # quest_color = palette[2]  # pastel teal
    # actig_color = palette[4]  # pastel green
    #
    # # pastel red for combined (lighter, less shiny)
    # combined_color = (0.95, 0.6, 0.6)  # soft pastel red
    #
    # stage_colors = {
    #     "quest": quest_color,
    #     "actig": actig_color,
    #     "two_stage": combined_color
    # }

    sns.scatterplot(
        data=result, x="spec", y="sens",
        hue="stage", style="stage",
        s=120, palette=stage_colors, edgecolor="k", ax=ax[0,0]
    )

    ax[0,0].set_xlim(max(0, result["spec"].min()-2), min(100, result["spec"].max()+2))
    ax[0,0].set_ylim(max(0, result["sens"].min()-2), min(100, result["sens"].max()+2))
    ax[0,0].set_xlabel("Specificity (%)", fontsize=font_size_label)
    ax[0,0].set_ylabel("Sensitivity (%)", fontsize=font_size_label)
    ax[0,0].set_title("Sens vs Spec", fontsize=font_size_title)
    ax[0,0].grid(True, linestyle="--", alpha=0.7)
    ax[0,0].legend(title="Stage", loc="lower left", fontsize=font_size_label-1)

    # --- Confusion Matrices ---
    def _plot_confusion_matrix_from_counts(ax, TP, FP, FN, TN, title, color):
        cm = np.array([[TN, FP], [FN, TP]])
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
        cmap = sns.light_palette(color, as_cmap=True)

        im = ax.imshow(cm, cmap=cmap)

        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                val, pct = cm[r, c], cm_pct[r, c]
                text_color = "black" if (r == c) else "dimgray"
                ax.text(c, r, f"{val}\n({pct:.1f}%)",
                        ha="center",
                        va="center", fontsize=font_size_cm,
                        # color=text_color,
                        color="black",  # force solid black
                        fontweight="bold",  # stronger for visibility
                        alpha=1.0  # fully opaque

                        )

        ax.set_xticks([0, 1])
        ax.set_xticklabels([class_names[0], class_names[1]], fontsize=font_size_label)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([class_names[0], class_names[1]], fontsize=font_size_label)
        ax.set_xlabel("Predicted", fontsize=font_size_label)
        ax.set_ylabel("True", fontsize=font_size_label)
        ax.set_title(title, fontsize=font_size_title)

    stages = ["quest", "actig", "two_stage"]
    titles = {"quest": "Questionnaire", "actig": "Actigraphy", "two_stage": "Combined"}

    for s, pos in zip(stages, [(0,1),(1,0),(1,1)]):
        row_s = result[result["stage"] == s]
        if not row_s.empty:
            row = row_s.iloc[0]
            _plot_confusion_matrix_from_counts(
                ax[pos],
                TP=row["TP"], FP=row["FP"], FN=row["FN"], TN=row["TN"],
                title=titles[s],
                color=stage_colors[s]
            )
        else:
            ax[pos].axis("off")
            ax[pos].set_title(f"{titles[s]} (no data)", fontsize=font_size_title)

    # --- Big Title with case/control counts ---
    n_ctrl = int(result["TN"].iloc[0] + result["FP"].iloc[0])
    n_case = int(result["TP"].iloc[0] + result["FN"].iloc[0])
    big_title = (f"Two-Stage vs Components\n{combo_id}\n"
                 f"{class_names[0]}={n_ctrl}, {class_names[1]}={n_case}")
    fig.suptitle(big_title, fontsize=font_size_title+2, y=0.98)

    plt.tight_layout()
    if save_plot and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = f"two_stage_grid_{'rawQ' if use_raw_q else 'mlQ'}.png"
        plt.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
        print(f"✅ Saved grid plot to {output_dir / fname}")
    plt.show()


# ----------------------------------------------------
# Compute permutation across all the predictions
# ----------------------------------------------------
def compute_two_stage_permutations(
    df_actig: pd.DataFrame,
    df_quest: pd.DataFrame,
    df_raw_quest: pd.DataFrame,
    subject_col: str = "subject_id",
    col_true: str = "y_true",
    prevalence: float = 0.015,
    output_dir: Path | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute metrics for all AND-combinations of actigraphy and questionnaire
    across models, optimizations, and thresholds. Includes single-question
    predictors combined only with actigraphy predictions.

    Output includes traceability: standalone questionnaire, actigraphy,
    and combined (two-stage) metrics, with a 'stage' column to identify.
    Adds `combo_id` for unique identification of each Acti+Quest pairing.
    Produces a pivoted comparison table as well.
    """

    rows = []

    # ----------- Identify binary columns -----------
    col_bin_actig = [
        col for col in df_actig.columns
        if ("y_pred" in col) and (set(df_actig[col].dropna().unique()) <= {0, 1})
    ]
    col_bin_quest = [
        col for col in df_quest.columns
        if ("y_pred" in col) and (set(df_quest[col].dropna().unique()) <= {0, 1})
    ]
    col_quest = [c for c in df_raw_quest.columns if c.startswith("q")]

    # ----------- Actigraphy × Questionnaire ML combos -----------
    for opt_a in df_actig["optimization"].unique():
        for col_pred_a in col_bin_actig:
            df_a = df_actig.loc[df_actig["optimization"] == opt_a,
                                [subject_col, col_true, col_pred_a]].copy()
            df_a = df_a.rename(columns={col_pred_a: "a_pred"})

            for model_q in df_quest["model_type"].unique():
                for opt_q in df_quest["optimization"].unique():
                    for col_pred_q in col_bin_quest:
                        df_q = df_quest.loc[
                            (df_quest["model_type"] == model_q) &
                            (df_quest["optimization"] == opt_q),
                            [subject_col, col_true, col_pred_q]
                        ].rename(columns={col_pred_q: "q_pred"})

                        df_merge = pd.merge(df_a, df_q, on=[subject_col, col_true], how="inner")

                        # ---- Standalone Questionnaire ----
                        m_q = _compute_confusion_metrics(df_merge[col_true], df_merge["q_pred"], prevalence)
                        m_q.update({
                            "Approach": f"Quest({model_q}-{opt_q}-{col_pred_q})",
                            "stage": "quest",
                            "opt_actig": opt_a,
                            "col_actig": col_pred_a,
                            "opt_quest": opt_q,
                            "model_quest": model_q,
                            "col_quest": col_pred_q,
                            "combo_id": _make_and_label(opt_a, col_pred_a, model_q, opt_q, col_pred_q)
                        })
                        rows.append(m_q)

                        # ---- Standalone Actigraphy ----
                        m_a = _compute_confusion_metrics(df_merge[col_true], df_merge["a_pred"], prevalence)
                        m_a.update({
                            "Approach": f"Acti({opt_a}-{col_pred_a})",
                            "stage": "actig",
                            "opt_actig": opt_a,
                            "col_actig": col_pred_a,
                            "opt_quest": opt_q,
                            "model_quest": model_q,
                            "col_quest": col_pred_q,
                            "combo_id": _make_and_label(opt_a, col_pred_a, model_q, opt_q, col_pred_q)
                        })
                        rows.append(m_a)

                        # ---- Two-stage AND ----
                        staged_and = ((df_merge["a_pred"] == 1) & (df_merge["q_pred"] == 1)).astype(int)
                        m_and = _compute_confusion_metrics(df_merge[col_true], staged_and, prevalence)
                        m_and.update({
                            "Approach": _make_and_label(opt_a, col_pred_a, model_q, opt_q, col_pred_q),
                            "stage": "two_stage",
                            "opt_actig": opt_a,
                            "col_actig": col_pred_a,
                            "opt_quest": opt_q,
                            "model_quest": model_q,
                            "col_quest": col_pred_q,
                            "combo_id": _make_and_label(opt_a, col_pred_a, model_q, opt_q, col_pred_q)
                        })
                        rows.append(m_and)

    # ----------- Single-question × Actigraphy combos -----------
    for qcol in col_quest:
        df_q1 = (
            df_raw_quest[[subject_col, col_true, qcol]]
            .copy()
            .rename(columns={qcol: "q_single"})
        )

        for opt_a in df_actig["optimization"].unique():
            for col_pred_a in col_bin_actig:
                df_a = df_actig.loc[df_actig["optimization"] == opt_a,
                                    [subject_col, col_true, col_pred_a]].copy()
                df_a = df_a.rename(columns={col_pred_a: "a_pred"})

                df_merge = pd.merge(df_q1, df_a, on=[subject_col, col_true], how="inner")

                # ---- Standalone Questionnaire ----
                m_q = _compute_confusion_metrics(df_merge[col_true], df_merge["q_single"], prevalence)
                m_q.update({
                    "Approach": f"Qcol({qcol})",
                    "stage": "quest",
                    "opt_actig": opt_a,
                    "col_actig": col_pred_a,
                    "opt_quest": "raw_q",
                    "model_quest": "single",
                    "col_quest": qcol,
                    "combo_id": _make_and_label(opt_a, col_pred_a, "single", "raw_q", qcol)
                })
                rows.append(m_q)

                # ---- Standalone Actigraphy ----
                m_a = _compute_confusion_metrics(df_merge[col_true], df_merge["a_pred"], prevalence)
                m_a.update({
                    "Approach": f"Acti({opt_a}-{col_pred_a})",
                    "stage": "actig",
                    "opt_actig": opt_a,
                    "col_actig": col_pred_a,
                    "opt_quest": "raw_q",
                    "model_quest": "single",
                    "col_quest": qcol,
                    "combo_id": _make_and_label(opt_a, col_pred_a, "single", "raw_q", qcol)
                })
                rows.append(m_a)

                # ---- Two-stage AND ----
                staged_and = ((df_merge["a_pred"] == 1) & (df_merge["q_single"] == 1)).astype(int)
                m_and = _compute_confusion_metrics(df_merge[col_true], staged_and, prevalence)
                m_and.update({
                    "Approach": _make_and_label(opt_a, col_pred_a, "single", "raw_q", qcol),
                    "stage": "two_stage",
                    "opt_actig": opt_a,
                    "col_actig": col_pred_a,
                    "opt_quest": "raw_q",
                    "model_quest": "single",
                    "col_quest": qcol,
                    "combo_id": _make_and_label(opt_a, col_pred_a, "single", "raw_q", qcol)
                })
                rows.append(m_and)

    # ----------- Build final DataFrame -----------
    df_out = pd.DataFrame(rows)

    # ----------- Pivot report -----------
    pivot_cols = ["combo_id", "opt_actig", "col_actig", "opt_quest", "model_quest", "col_quest"]
    df_pivot = (
        df_out.pivot_table(
            index=pivot_cols,
            columns="stage",
            values=["sens", "spec"]
        )
        .reset_index()
    )

    # flatten MultiIndex columns
    df_pivot.columns = [
        "_".join([c for c in col if c]).strip("_")
        for col in df_pivot.columns.values
    ]

    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(output_dir / "two_stage_permutations.csv", index=False)
        df_out[["Approach", "stage", "sens", "spec", "combo_id"]].to_csv(
            output_dir / "two_stage_sens_spec.csv", index=False
        )
        df_pivot.to_csv(output_dir / "two_stage_pivot.csv", index=False)

        print(f"✅ Saved two-stage permutation metrics to {output_dir.resolve()}")
        print(f"📊 Sens/Spec report: {output_dir / 'two_stage_sens_spec.csv'}")
        print(f"📊 Pivot report: {output_dir / 'two_stage_pivot.csv'}")

    return df_out, df_pivot


# %% Selecting the best one and visualization
def select_best_two_stage(
    df_two_stage: pd.DataFrame,
    use_raw_q: bool = False,
    min_sens: float = 70,
    max_spec: float = 100,
    make_plot: bool = True,
    output_dir: Path | None = None
) -> pd.DataFrame:
    """
    Select best two-stage combo (max spec < max_spec, sens > min_sens).
    If use_raw_q=True, restrict to raw questionnaire; else restrict to ML questionnaire.
    Returns the two_stage row + corresponding quest and actig rows.
    Optionally makes a 2x2 grid plot (scatter + confusion matrices).
    """

    # restrict dataset
    if use_raw_q:
        df_stage = df_two_stage.loc[
            (df_two_stage['stage'] == 'two_stage') &
            (df_two_stage['opt_quest'] == 'raw_q')
        ]
    else:
        df_stage = df_two_stage.loc[
            (df_two_stage['stage'] == 'two_stage') &
            (df_two_stage['opt_quest'] != 'raw_q')  # ML questionnaires only
        ]

    # apply filters
    df_stage = df_stage.loc[
        (df_stage['spec'] < max_spec) &
        (df_stage['sens'] > min_sens)
    ]

    if df_stage.empty:
        print("⚠️ No rows match the criteria.")
        return None

    # best row = max spec, then max sens
    best_two_stage = (
        df_stage
        .sort_values(by=['spec', 'sens'], ascending=[False, False])
        .head(1)
    )

    # extract combo_id
    combo_id = best_two_stage["combo_id"].iloc[0]

    # collect quest + actig + two_stage rows
    df_components = df_two_stage.loc[df_two_stage["combo_id"] == combo_id]

    # ⚡ Ensure predictions are carried along
    result = df_components.loc[df_components["stage"].isin(["quest", "actig", "two_stage"])].copy()

    # add raw preds for CM plotting
    for stage in ["quest", "actig", "two_stage"]:
        if stage in result["stage"].values:
            row = result.loc[result["stage"] == stage]
            # assume preds stored in df_two_stage as e.g. f"{stage}_preds"
            if f"{stage}_y_true" in df_two_stage.columns and f"{stage}_y_pred" in df_two_stage.columns:
                result.loc[result["stage"] == stage, "y_true"] = row[f"{stage}_y_true"].values[0]
                result.loc[result["stage"] == stage, "y_pred"] = row[f"{stage}_y_pred"].values[0]

    # ---- Grid Plot ----
    if make_plot and not result.empty:
        plot_two_stage_with_components(
            result=result,
            combo_id=combo_id,
            save_plot=True,
            output_dir=output_dir,
            use_raw_q=use_raw_q
        )

    return result

