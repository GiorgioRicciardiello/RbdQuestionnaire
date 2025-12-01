import pathlib
from pathlib import Path
from typing import Tuple, Dict, Optional, Union
import json
from sklearn.utils import resample

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix
from pathlib import Path
from matplotlib import colors
from typing import Callable, Optional, List, Tuple
from matplotlib.axes import Axes
from numpy.typing import ArrayLike
import matplotlib.patches as patches

from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    precision_recall_fscore_support, accuracy_score
)
from sklearn.calibration import calibration_curve
from library.ml_actigraphy.scoring import compute_metrics
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
    brier_score_loss
)

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve
from pathlib import Path
import matplotlib.colors as mcolors

def _plot_minimalist_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    folds: np.ndarray | None,
    model: str,
    avg_type: str,
    color: str,
    out_path: Path,
    auc_val: float,
    sens_val: float,
    spec_val: float,
    thr_val: float,
    auc_ci: str | None = None,
    sens_ci: str | None = None,
    spec_ci: str | None = None,
    n_boot: int = 1000,
    confidence: float = 0.95,
):
    """
    Generate a minimalist ROC curve without axes.
    - If folds is not None: variance from folds.
    - If folds is None (or all the same): variance from bootstrap subjects.
    Saves the figure as roc_{model}_{avg_type}_{thr:.2f}.png in out_path.
    """

    mean_fpr = np.linspace(0, 1, 200)
    tprs = []

    if folds is not None and len(np.unique(folds)) > 1:
        # ---- CV-based variance ----
        for f in np.unique(folds):
            mask = folds == f
            fpr, tpr, _ = roc_curve(y_true[mask], y_score[mask])
            tprs.append(np.interp(mean_fpr, fpr, tpr))

    else:
        # ---- Bootstrap variance ----
        rng = np.random.default_rng(42)
        n = len(y_true)
        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            y_b, s_b = y_true[idx], y_score[idx]
            if len(np.unique(y_b)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_b, s_b)
            tprs.append(np.interp(mean_fpr, fpr, tpr))

    # Aggregate curves
    tprs = np.array(tprs)
    mean_tpr = tprs.mean(axis=0)
    lower = np.percentile(tprs, (1 - confidence) / 2 * 100, axis=0)
    upper = np.percentile(tprs, (1 + confidence) / 2 * 100, axis=0)

    # --- threshold operating point ---
    fpr_all, tpr_all, thr_all = roc_curve(y_true, y_score)
    idx = np.argmin(np.abs(thr_all - thr_val))
    fpr_thr, tpr_thr = fpr_all[idx], tpr_all[idx]
    tpr_thr_interp = np.interp(fpr_thr, mean_fpr, mean_tpr)

    # --- legend text ---
    auc_text = f"AUC={auc_ci}" if auc_ci else f"AUC={auc_val:.2f}"
    sens_text = f"Se={sens_ci}" if sens_ci else f"Se={sens_val:.1f}%"
    spec_text = f"Sp={spec_ci}" if spec_ci else f"Sp={spec_val:.1f}%"
    legend_text = f"{auc_text}\n{sens_text}\n{spec_text}\nthr={thr_val:.2f}"

    # --- prepare metrics dictionary ---
    metrics_dict = {
        "model": model,
        "avg_type": avg_type,
        "thr_val": round(thr_val, 3),
        "auc": round(auc_val, 3),
        "auc_ci": auc_ci,
        "sensitivity": round(sens_val, 3),
        "sens_ci": sens_ci,
        "specificity": round(spec_val, 3),
        "spec_ci": spec_ci,
        "fpr_thr": round(fpr_thr, 3),
        "tpr_thr": round(tpr_thr_interp, 3),
    }

    # --- colors ---
    rgb = np.array(mcolors.to_rgb(color))
    light_rgb = np.clip(rgb + 0.3, 0, 1)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_facecolor("#f5f5f5")
    fig.patch.set_alpha(0)

    ax.plot(mean_fpr, mean_tpr, color=color, lw=2, label=legend_text)
    ax.fill_between(mean_fpr, lower, upper, color=light_rgb, alpha=0.5)

    # threshold marker
    ax.scatter(fpr_thr, tpr_thr_interp, color=color, s=70,
               edgecolors="k", zorder=3)

    # chance line
    ax.plot([0, 1], [0, 1], ls="--", color="gray", alpha=0.3)

    # light axes
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(axis="both", which="both", labelsize=8, labelcolor=(0, 0, 0, 0.5))
    for spine in ax.spines.values():
        spine.set_alpha(0.5)
    ax.grid(alpha=.7)

    # --- save metrics CSV ---
    df_metrics = pd.DataFrame([metrics_dict])
    existing_csvs = list(out_path.glob("roc_metrics*.csv"))
    if existing_csvs:
        df_metrics.to_csv(existing_csvs[0], mode="a", header=False, index=False)
    else:
        df_metrics.to_csv(out_path / "roc_metrics.csv", index=False)

    out_path = out_path.joinpath("roc_curves")
    out_path.mkdir(parents=True, exist_ok=True)
    fname = out_path / f"roc_{model}_{avg_type}_thr{thr_val:.2f}.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=True)
    plt.close()

    return fname


# %% plot roc curves and confusion matrices
def plot_threshold_summary(
        df_metrics: pd.DataFrame | None,
        df_predictions: pd.DataFrame,
        class_names: dict[int, str],
        out_path: Path | None = None,
        font_size_title: int = 14,
        font_size_big_title: int = 18,
        font_size_label: int = 12,
        font_size_legend: int = 10,
        font_size_cm: int = 12,
):
    """
    Plot ROC curves + Confusion Matrices for each optimization × threshold_type.

    Rows = optimization (auc / youden / sens)
    Cols = threshold types (standard / youden_j / custom) × (ROC + CM)
    """

    # ------------------------------
    # helpers
    # ------------------------------
    # def _scatter_threshold_on_mean(ax, sens_val, spec_val, color):
    #     """Scatter point based on reported sensitivity/specificity (metrics)."""
    #     fpr_point = 1 - spec_val
    #     tpr_point = sens_val
    #     ax.scatter(fpr_point, tpr_point, color=color, s=70,
    #                edgecolors="k", zorder=3)

    def _scatter_threshold_on_mean(ax, mean_fpr, mean_tpr, spec_val, color):
        """Place scatter exactly on the mean ROC curve at the FPR ~ (1 - specificity)."""
        fpr_point = 1 - spec_val
        # interpolate TPR at that FPR from the mean ROC
        tpr_point = np.interp(fpr_point, mean_fpr, mean_tpr)
        ax.scatter(fpr_point, tpr_point, color=color, s=70,
                   edgecolors="k", zorder=3)

    def _bootstrap_auc(y_true, y_score, n_bootstrap=1000, ci=0.95, random_state=42):
        """Compute bootstrap mean AUC and CI."""
        rng = np.random.RandomState(random_state)
        aucs = []
        for _ in range(n_bootstrap):
            y_res, y_score_res = resample(y_true, y_score, random_state=rng)
            try:
                aucs.append(roc_auc_score(y_res, y_score_res))
            except ValueError:
                continue
        mean_auc = np.mean(aucs)
        lower = np.percentile(aucs, ((1 - ci) / 2) * 100)
        upper = np.percentile(aucs, (1 - (1 - ci) / 2) * 100)
        return mean_auc, (lower, upper)

    def _draw_mean_roc(ax,
                       y_true,
                       y_score,
                       folds,
                       color,
                       auc_val,
                       auc_ci,
                       sens_val,
                       sens_ci,
                       spec_val,
                       spec_ci,
                       thr_val,
                       title,
                       n_bootstrap: int = 1000):
        mean_fpr = np.linspace(0, 1, len(y_true))
        tprs = []
        for f in np.unique(folds):
            mask = folds == f
            fpr, tpr, _ = roc_curve(y_true[mask], y_score[mask])
            tprs.append(np.interp(mean_fpr, fpr, tpr))
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)

        # --- Bootstrap AUC ---
        mean_auc, (low_auc, high_auc) = _bootstrap_auc(
            y_true, y_score, n_bootstrap=n_bootstrap
        )
        auc_text = f"AUC={mean_auc:.2f} ({low_auc:.2f}, {high_auc:.2f})"

        # --- Sens & Spec ---
        sens_text = f"Se={sens_ci}" if sens_ci else f"Se={sens_val:.2f}"
        spec_text = f"Sp={spec_ci}" if spec_ci else f"Sp={spec_val:.2f}"

        ax.plot(mean_fpr, mean_tpr, color=color, lw=2,
                label=f"{auc_text}\n{sens_text}\n{spec_text}")
        ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                        color=color, alpha=0.2)

        # scatter from sensitivity/specificity values
        _scatter_threshold_on_mean(ax=ax, mean_fpr=mean_fpr,
                                   mean_tpr=mean_tpr,
                                   spec_val=spec_val,
                                   color=color)

        ax.set_title(title, fontsize=font_size_title)
        ax.set_xlabel("False Positive Rate", fontsize=font_size_label)
        ax.set_ylabel("True Positive Rate", fontsize=font_size_label)
        ax.legend(fontsize=font_size_legend, loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.5)

    def _draw_confusion_matrix(ax, cm, color, title):
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
        cmap = sns.light_palette(color, as_cmap=True)
        im = ax.imshow(cm, cmap=cmap)
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                val, pct = cm[r, c], cm_pct[r, c]
                bg_color = im.cmap(im.norm(cm[r, c]))
                brightness = colors.rgb_to_hsv(bg_color[:3])[2]
                text_color = "black" if brightness > 0.5 else "white"
                ax.text(c, r, f"{val}\n({pct:.1f}%)",
                        ha="center", va="center",
                        fontsize=font_size_cm, color=text_color)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([class_names[0], class_names[1]])
        ax.set_yticks([0, 1])
        ax.set_yticklabels([class_names[0], class_names[1]])
        ax.set_xlabel("Predicted", fontsize=font_size_label)
        ax.set_ylabel("True", fontsize=font_size_label)
        ax.set_title(title, fontsize=font_size_title)

    # ------------------------------
    # prepare metrics if missing
    # ------------------------------
    if df_metrics is None:
        df_metrics = _rebuild_metrics_from_predictions(df_predictions=df_predictions,
                                                       maximize='sepc')

    max_pred_col = [col for col in df_predictions.columns if 'max' in col and 'pred' in col][0]
    max_metric = max_pred_col.split('y_pred')[1][1:]
    # --- threshold mapping ---
    threshold_types = {
        "standard": "y_pred_standard",
        "youden_j": "y_pred_youden",
        f"{max_metric}": max_pred_col  # y_pred_spec_max
    }
    optims = sorted(df_predictions["optimization"].unique())
    n_rows, n_cols = len(optims), len(threshold_types) * 2

    palette = sns.color_palette("tab10", len(optims))
    opt_colors = {o: palette[i] for i, o in enumerate(optims)}

    counts = None
    if 'fold' not in df_predictions.columns:
        df_predictions['fold'] = -1

        first_opt = df_predictions['optimization'].unique()[0]
        subjects_first_opt = df_predictions.loc[
            df_predictions['optimization'] == first_opt, 'subject_id'
        ].unique()

        df_same_subjects = df_predictions.loc[
            (df_predictions['subject_id'].isin(subjects_first_opt)) &
            (df_predictions['optimization'] == first_opt), 'y_true']
        counts = pd.Series(df_same_subjects).map(class_names).value_counts()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    # ------------------------------
    # main loop
    # ------------------------------
    for i, opt in enumerate(optims):
        color = opt_colors[opt]

        for j, (thr_name, pred_col) in enumerate(threshold_types.items()):
            ax_roc = axes[i, j * 2]
            ax_cm = axes[i, j * 2 + 1]

            preds_opt = df_predictions.loc[
                df_predictions["optimization"] == opt,
                ["fold", "y_true", "y_pred", pred_col]
            ]
            metrics_opt = df_metrics.loc[
                (df_metrics["optimization"] == opt) &
                (df_metrics["threshold_type"] == thr_name)
            ]

            if metrics_opt.empty:
                continue

            y_true = preds_opt["y_true"].values
            if counts is None:
                counts = pd.Series(y_true).map(class_names).value_counts()
            y_score = preds_opt["y_pred"].values
            folds = preds_opt["fold"].values

            auc_val = metrics_opt["auc"].mean()
            sens_val = metrics_opt["sensitivity"].mean() / 100.0
            spec_val = metrics_opt["specificity"].mean() / 100.0
            thr_val = metrics_opt["threshold"].mean()

            auc_ci = metrics_opt["auc_ci"].iloc[0] if "auc_ci" in metrics_opt else None
            sens_ci = metrics_opt["sensitivity_ci"].iloc[0] if "sensitivity_ci" in metrics_opt else None
            spec_ci = metrics_opt["specificity_ci"].iloc[0] if "specificity_ci" in metrics_opt else None

            title = f"{opt.upper()} | {thr_name.replace('_', ' ').title()} \n(τ={thr_val:.2f})"

            _draw_mean_roc(ax=ax_roc,
                           y_true=y_true,
                           y_score=y_score,
                           folds=folds,
                           color=color,
                           auc_val=auc_val,
                           auc_ci=auc_ci,
                           sens_val=sens_val,
                           sens_ci=sens_ci,
                           spec_val=spec_val,
                           spec_ci=spec_ci,
                           thr_val=thr_val,
                           title=title)


            # ---- minimalist ROC export ----
            if out_path:
                _plot_minimalist_roc(
                    y_true=y_true,
                    y_score=y_score,
                    folds=folds,
                    model=opt,
                    avg_type=thr_name,
                    color="#A9C9FF",
                    out_path=out_path,
                    auc_val=auc_val,
                    sens_val=sens_val,
                    spec_val=spec_val,
                    auc_ci=auc_ci,
                    sens_ci=sens_ci,
                    spec_ci=spec_ci,
                    thr_val=thr_val
                )

            y_pred_bin = preds_opt[pred_col].values
            cm = confusion_matrix(y_true, y_pred_bin, labels=[0, 1])
            _draw_confusion_matrix(ax_cm, cm, color, title)

    # ------------------------------
    # Layout
    # ------------------------------
    big_title = "Class distribution: " + ", ".join([f"{cls}={cnt}" for cls, cnt in counts.items()])
    fig.suptitle(big_title, fontsize=font_size_big_title, y=0.99)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path / f"plt_summary.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# %% Per subject analysis
def per_subject_evaluation(
    df_predictions: pd.DataFrame,
    subject_col: str = "subject_id",
    results_dir: Path | None = None,
    class_names: dict[int, str] = {0: "Control", 1: "Case"},
):
    """
    Aggregate predictions at the subject level using:
      1. Average of predicted scores
      2. Majority voting (average of majority-class probabilities)

    Returns
    -------
    results : dict
        {
          "avg_scores": (df_predictions_avg, df_metrics_avg),
          "majority_voting": (df_predictions_mv, df_metrics_mv)
        }
    """
    # ----------------------------------------------------
    # Helper functions
    # ----------------------------------------------------

    def _bootstrap_ci(y_true, y_score, thr: float, n_boot: int = 1000, confidence: float = 0.95) -> dict:
        """Bootstrap subjects to compute CIs for metrics in percentages."""
        rng = np.random.default_rng(42)
        sens_samples, spec_samples, auc_samples = [], [], []

        n = len(y_true)
        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            y_b, s_b = y_true[idx], y_score[idx]
            y_pred_b = (s_b >= thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_b, y_pred_b, labels=[0, 1]).ravel()

            sens_samples.append(tp / (tp + fn + 1e-12) * 100)  # percentage
            spec_samples.append(tn / (tn + fp + 1e-12) * 100)  # percentage
            if len(np.unique(y_b)) > 1:
                auc_samples.append(roc_auc_score(y_b, s_b))

        def ci_str_percent(arr):
            if len(arr) == 0:
                return np.nan
            mean = np.mean(arr)
            low, high = np.percentile(arr, [(1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100])
            return f"{mean:.1f}% ({low:.1f}%, {high:.1f}%)"


        def ci_str_decimal(arr):
            if len(arr) == 0:
                return np.nan
            mean = np.mean(arr)
            low, high = np.percentile(arr, [(1 - confidence) / 2 , (1 + confidence) / 2])
            return f"{mean:.3f} ({low:.3f}, {high:.3f})"

        return {
            "sensitivity_ci": ci_str_percent(sens_samples),
            "specificity_ci": ci_str_percent(spec_samples),
            "auc_ci": ci_str_decimal(auc_samples),
        }

    def _compute_metrics(y_true, y_score, thr: float) -> Dict[str, float]:
        """Compute metrics at given threshold; return threshold too."""
        y_pred = (y_score >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sens = tp / (tp + fn + 1e-12)
        spec = tn / (tn + fp + 1e-12)
        ppv = tp / (tp + fp + 1e-12)
        npv = tn / (tn + fn + 1e-12)
        roc_auc = roc_auc_score(y_true, y_score)
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(rec, prec)
        youden = sens + spec - 1
        brier = brier_score_loss(y_true, y_score)

        return {
            "sensitivity": round(sens * 100, 3),
            "specificity": round(spec * 100, 3),
            "auc": roc_auc,
            "prc": pr_auc,
            "ppv": ppv,
            "npv": npv,
            "youden": youden,
            "threshold": thr,
            "brier": brier
        }

    def _aggregate_avg(df_predictions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aggregate by average scores across nights per subject
        Treats probabilities directly.
        Each subject’s y_pred is the average probability across nights.

        Thresholds (0.5, Youden J, spec_max) are applied after aggregation.

        So this favors smooth probability estimates → better for ROC / AUC type analysis.
        """
        # Step 1: Create the average observation per subject base averages per subject_id × optimization
        # df_avg = (
        #     df_predictions
        #     .groupby(["subject_id", "optimization"])
        #     .agg({
        #         "y_true": "first",
        #         "y_pred": "mean",
        #     })
        # )
        df_avg = (
            df_predictions
            .groupby(["subject_id", "optimization"])
            .agg(
                y_true=("y_true", "first"),
                y_pred=("y_pred", "mean"),
            )
        )

        # Step 2: compute thresholds per optimization (average across folds)
        thr_cols: list[str] = [col for col in df_predictions if col.startswith("thr_")]
        df_thr = (
            df_predictions
            .groupby("optimization")[thr_cols]
            .mean()
        )
        # Step 3: align thresholds with df_avg (broadcast across all subjects)
        # Reindex so optimization matches df_avg’s second level
        df_thr = df_thr.reindex(df_avg.index.get_level_values("optimization").unique())
        # Add thresholds as new columns, broadcasting per optimization
        df_avg = df_avg.join(df_thr, on="optimization")
        # we have pwe subject their optimization value
        df_avg = df_avg.reset_index(drop=False)
        assert df_avg.optimization.nunique() * df_avg.subject_id.nunique() == df_avg.shape[0]
        # now each of the thresholds are unique, now we can use them to compute the binary
        # include all the thresholds even the 0.5
        thr_cols = [col for col in df_predictions if col.startswith("thr_")]
        for thr_col in thr_cols:
            pred_col = thr_col.replace("thr_", "y_pred_")
            assert df_avg[thr_col].nunique() == 1
            df_avg[pred_col] = (df_avg['y_pred'] >= df_avg[thr_col].unique()[0]).astype(int)

        # --- loop to compute metrics at subject-level ---
        all_records = []
        for opt in df_avg["optimization"].unique():
            df_opt = df_avg[df_avg["optimization"] == opt]  # one row per subject
            for thr_col in thr_cols:
                thr_val = df_opt[thr_col].iloc[0]  # unique per optimization

                # main metrics
                m = _compute_metrics(df_opt["y_true"], df_opt["y_pred"], thr_val)

                # bootstrap CI
                ci = _bootstrap_ci(df_opt["y_true"].values,
                                   df_opt["y_pred"].values,
                                   thr_val)

                # store
                m.update(ci)
                m.update({
                    "optimization": opt,
                    "threshold_type": thr_col,
                    "threshold": thr_val,
                })
                all_records.append(m)

        df_metrics_avg = pd.DataFrame(all_records)
        df_metrics_avg['threshold_type'] = df_metrics_avg['threshold_type'].str.replace('thr_', '').replace('youden', 'youden_j')

        return df_avg, df_metrics_avg

    def _aggregate_majority(df_predictions: pd.DataFrame,
                            subject_col: str = "subject_id") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aggregate predictions at the subject level using majority voting.

        - For each subject × optimization:
            - Convert nightly scores into binary using thresholds.
            - Assign the majority class (>=50% wins).
            - Continuous subject-level probability is the mean of nightly
              probabilities belonging to the majority class (so ROC curves
              still get a probability-like input).
        - Thresholds: standard (0.5), Youden J, spec_max.

        Returns
        -------
        df_mv : pd.DataFrame
            Subject-level aggregated predictions, with:
              - y_true
              - y_pred (continuous, majority-class mean prob)
              - y_pred_standard / y_pred_youden / y_pred_spec_max (binary)
        df_metrics_mv : pd.DataFrame
            Metrics (sensitivity, specificity, AUC, etc.) with bootstrap CIs.
        """
        dfs_mv = []
        for (sid, opt), g in df_predictions.groupby([subject_col, "optimization"]):
            row = {
                subject_col: sid,
                "optimization": opt,
                "y_true": g["y_true"].unique()[0],
                "thr_youden": g["thr_youden"].unique()[0],
                "thr_spec_max": g["thr_spec_max"].unique()[0],
            }

            # nightly scores
            y_scores = g["y_pred"].values
            thresholds = {
                "standard": 0.5,
                "youden_j": row["thr_youden"],
                "spec_max": row["thr_spec_max"],
            }

            # ---- continuous subject-level probability for ROC ----
            # majority class determined using baseline threshold 0.5
            y_bin_base = (y_scores >= 0.5).astype(int)
            majority_class_base = int(y_bin_base.mean() >= 0.5)
            y_majority_scores = y_scores[y_bin_base == majority_class_base]

            row["y_pred"] = (
                y_majority_scores.mean() if len(y_majority_scores) > 0 else y_scores.mean()
            )

            # ---- binary predictions for CM ----
            for tname, thr_val in thresholds.items():
                if tname == "youden_j":
                    col_name = "y_pred_youden"
                else:
                    col_name = f"y_pred_{tname}"

                y_bin = (y_scores >= thr_val).astype(int)
                majority_class = int(y_bin.mean() >= 0.5)

                row[col_name] = majority_class

            dfs_mv.append(row)

        df_mv = pd.DataFrame(dfs_mv)

        # ---------- Metrics ----------
        all_records = []
        for opt in df_mv["optimization"].unique():
            g = df_mv[df_mv["optimization"] == opt]
            thresholds = {
                "standard": 0.5,
                "youden_j": g["thr_youden"].iloc[0],
                "spec_max": g["thr_spec_max"].iloc[0],
            }
            for tname, thr_val in thresholds.items():
                # Point estimate
                m = _compute_metrics(g["y_true"], g["y_pred"], thr_val)

                # Bootstrap CI
                ci = _bootstrap_ci(g["y_true"].values, g["y_pred"].values, thr_val)

                m.update(ci)
                m.update({
                    "optimization": opt,
                    "threshold_type": tname,
                    "threshold": thr_val,
                })
                all_records.append(m)

        df_metrics_mv = pd.DataFrame(all_records)
        return df_mv, df_metrics_mv

    # ----------------------------------------------------
    # Run both aggregation strategies
    # ----------------------------------------------------
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)

    df_avg, df_metrics_avg = _aggregate_avg(df_predictions)
    df_mv, df_metrics_mv = _aggregate_majority(df_predictions)

    results = {
        "avg_scores": (df_avg, df_metrics_avg),
        "majority_voting": (df_mv, df_metrics_mv),
    }

    # compute brieri scores

    # Save + plots
    if results_dir:
        for name, (dfp, dfm) in results.items():
            dfp.to_csv(results_dir / f"predictions_{name}.csv", index=False)
            dfm.to_csv(results_dir / f"metrics_{name}.csv", index=False)

            path_plot = results_dir / name
            path_plot.mkdir(parents=True, exist_ok=True)
            print(name)

            plot_threshold_summary_per_subject(
                df_metrics=dfm,
                df_predictions=dfp,
                class_names=class_names,
                out_path=path_plot
            )

    return results


from sklearn.metrics import roc_auc_score, confusion_matrix

def _rebuild_metrics_from_predictions(
    df_predictions: pd.DataFrame,
    maximize: str = "spec"
) -> pd.DataFrame:
    """
    Recreate df_metrics from df_predictions without re-running training.
    """

    records = []

    # Map prediction columns back to threshold_type
    threshold_map = {
        "y_pred_standard": "standard",
        "y_pred_youden": "youden_j",
        f"y_pred_{maximize}_max": f"{maximize}_max"
    }
    if not 'model_type' in df_predictions:
        df_predictions['model_type'] = 'xgboost'

    for (fold, opt, model_type), group in df_predictions.groupby(["fold", "optimization", "model_type"]):
        y_true = group["y_true"].values
        y_score = group["y_pred"].values

        # Loop through threshold types
        for pred_col, thr_name in threshold_map.items():
            if pred_col not in group:
                continue

            y_pred_bin = group[pred_col].values

            # Threshold values (take mean across fold if column exists)
            thr_col = pred_col.replace("y_pred", "thr")
            thr_val = group[thr_col].mean() if thr_col in group else 0.5

            # Metrics
            auc_val = roc_auc_score(y_true, y_score)
            cm = confusion_matrix(y_true, y_pred_bin, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            sensitivity = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = 100 * tn / (tn + fp) if (tn + fp) > 0 else 0

            rec = dict(
                fold=fold,
                optimization=opt,
                model_type=model_type,
                threshold_type=thr_name,
                auc=auc_val,
                sensitivity=sensitivity,
                specificity=specificity,
                threshold=thr_val,
            )
            records.append(rec)

    df_metrics = pd.DataFrame(records)

    # Sort like original function
    cols_sens = [c for c in df_metrics.columns if c.startswith("sensitivity")]
    cols_spec = [c for c in df_metrics.columns if c.startswith("specificity")]
    cols_sens_spec = sorted(cols_sens + cols_spec)

    df_metrics.sort_values(by=["model_type", "threshold_type", "optimization", "fold"],
                           ascending=[True, True, True, True], inplace=True)

    sort_keys = ["fold", "model_type", "threshold_type", "optimization"]
    other_cols = [c for c in df_metrics.columns if c not in sort_keys + cols_sens_spec]
    df_metrics = df_metrics[sort_keys + cols_sens_spec + other_cols]

    return df_metrics



def plot_threshold_summary_per_subject(
        df_metrics: pd.DataFrame | None,
        df_predictions: pd.DataFrame,
        class_names: dict[int, str],
        out_path: Path | None = None,
        font_size_title: int = 14,
        font_size_big_title: int = 18,
        font_size_label: int = 12,
        font_size_legend: int = 10,
        font_size_cm: int = 12,
):
    """
    Plot ROC curves + Confusion Matrices for each optimization × threshold_type.
    Rows = optimization (auc / youden / spec)
    Cols = threshold types (standard / youden_j / custom) × (ROC + CM)
    """

    # ------------------------------
    # helpers
    # ------------------------------
    def _scatter_threshold(ax, y_true, y_score, thr_val, color):
        fpr, tpr, thr = roc_curve(y_true, y_score)
        idx = np.argmin(np.abs(thr - thr_val))
        ax.scatter(fpr[idx], tpr[idx], color=color, s=70,
                   edgecolors="k", zorder=3)

    def _draw_mean_roc(ax,
                       y_true,
                       y_score,
                       color,
                       auc_val,
                       auc_ci,
                       sens_val,
                       sens_ci,
                       spec_val,
                       spec_ci,
                       thr_val,
                       title,
                       n_boot: int = 1000,
                       confidence: float = 0.95):
        mean_fpr = np.linspace(0, 1, 200)
        tprs = []

        rng = np.random.default_rng(42)
        n = len(y_true)

        # --- bootstrap subjects ---
        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            y_b, s_b = y_true[idx], y_score[idx]
            if len(np.unique(y_b)) < 2:  # skip if resample has only one class
                continue
            fpr, tpr, _ = roc_curve(y_b, s_b)
            tprs.append(np.interp(mean_fpr, fpr, tpr))

        # Aggregate
        tprs = np.array(tprs)
        mean_tpr = tprs.mean(axis=0)
        lower = np.percentile(tprs, (1 - confidence) / 2 * 100, axis=0)
        upper = np.percentile(tprs, (1 + confidence) / 2 * 100, axis=0)

        # Labels
        auc_text = f"AUC={auc_ci}" if auc_ci else f"AUC={auc_val:.2f}"
        sens_text = f"Se={sens_ci}" if sens_ci else f"Se={sens_val:.2f}"
        spec_text = f"Sp={spec_ci}" if spec_ci else f"Sp={spec_val:.2f}"

        # Plot
        ax.plot(mean_fpr, mean_tpr, color=color, lw=2,
                label=f"{auc_text}\n{sens_text}\n{spec_text}")
        ax.fill_between(mean_fpr, lower, upper, color=color, alpha=0.2)

        _scatter_threshold(ax, y_true, y_score, thr_val, color)
        ax.set_title(title, fontsize=font_size_title)
        ax.set_xlabel("False Positive Rate", fontsize=font_size_label)
        ax.set_ylabel("True Positive Rate", fontsize=font_size_label)
        ax.legend(fontsize=font_size_legend, loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.5)

    def _draw_confusion_matrix(ax, cm, color, title):
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
        cmap = sns.light_palette(color, as_cmap=True)
        im = ax.imshow(cm, cmap=cmap)
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                val, pct = cm[r, c], cm_pct[r, c]
                bg_color = im.cmap(im.norm(cm[r, c]))
                brightness = colors.rgb_to_hsv(bg_color[:3])[2]
                text_color = "black" if brightness > 0.5 else "white"
                ax.text(c, r, f"{val}\n({pct:.1f}%)",
                        ha="center", va="center",
                        fontsize=font_size_cm, color=text_color)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([class_names[0], class_names[1]])
        ax.set_yticks([0, 1])
        ax.set_yticklabels([class_names[0], class_names[1]])
        ax.set_xlabel("Predicted", fontsize=font_size_label)
        ax.set_ylabel("True", fontsize=font_size_label)
        ax.set_title(title, fontsize=font_size_title)

    # ------------------------------
    # setup
    # ------------------------------
    if df_metrics is None:
        df_metrics = _rebuild_metrics_from_predictions(df_predictions=df_predictions,
                                                       maximize='sepc')

    max_pred_col = [col for col in df_predictions.columns if 'max' in col and 'pred' in col][0]
    max_metric = max_pred_col.split('y_pred')[1][1:]
    threshold_types = {
        "standard": "y_pred_standard",
        "youden_j": "y_pred_youden",
        f"{max_metric}": max_pred_col
    }
    optims = sorted(df_predictions["optimization"].unique())
    n_rows, n_cols = len(optims), len(threshold_types) * 2

    palette = sns.color_palette("tab10", len(optims))
    opt_colors = {o: palette[i] for i, o in enumerate(optims)}

    counts = None
    if 'fold' not in df_predictions.columns:
        df_predictions['fold'] = -1
        first_opt = df_predictions['optimization'].unique()[0]
        subjects_first_opt = df_predictions.loc[
            df_predictions['optimization'] == first_opt, 'subject_id'
        ].unique()
        df_same_subjects = df_predictions.loc[
            (df_predictions['subject_id'].isin(subjects_first_opt)) &
            (df_predictions['optimization'] == first_opt), 'y_true']
        counts = pd.Series(df_same_subjects).map(class_names).value_counts()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    # ------------------------------
    # main loop
    # ------------------------------
    for i, opt in enumerate(optims):
        color = opt_colors[opt]
        for j, (thr_name, pred_col) in enumerate(threshold_types.items()):
            ax_roc = axes[i, j * 2]
            ax_cm = axes[i, j * 2 + 1]

            preds_opt = df_predictions.loc[
                df_predictions["optimization"] == opt,
                ["fold", "y_true", "y_pred", pred_col]
            ]
            metrics_opt = df_metrics.loc[
                (df_metrics["optimization"] == opt) &
                (df_metrics["threshold_type"] == thr_name)
            ]

            if metrics_opt.empty:
                continue

            y_true = preds_opt["y_true"].values
            if counts is None:
                counts = pd.Series(y_true).map(class_names).value_counts()
            y_score = preds_opt["y_pred"].values

            auc_val = metrics_opt["auc"].mean()
            sens_val = metrics_opt["sensitivity"].mean()
            spec_val = metrics_opt["specificity"].mean()
            thr_val = metrics_opt["threshold"].mean()

            auc_ci = metrics_opt["auc_ci"].iloc[0] if "auc_ci" in metrics_opt else None
            sens_ci = metrics_opt["sensitivity_ci"].iloc[0] if "sensitivity_ci" in metrics_opt else None
            spec_ci = metrics_opt["specificity_ci"].iloc[0] if "specificity_ci" in metrics_opt else None

            title = (f"{opt.upper()} | {thr_name.replace('_', ' ').title()} "
                     f"\n(τ={thr_val:.2f})")

            _draw_mean_roc(ax_roc, y_true, y_score, color,
                           auc_val, auc_ci, sens_val, sens_ci,
                           spec_val, spec_ci, thr_val, title)

            # ---- minimalist ROC export ----
            if out_path:
                _plot_minimalist_roc(
                    y_true=y_true,
                    y_score=y_score,
                    folds=None,
                    model=opt,  # optimization name
                    avg_type=thr_name,  # threshold type
                    # color=color,
                    color= "#A9C9FF",
                    out_path=out_path,
                    auc_val=auc_val,
                    sens_val=sens_val,
                    spec_val=spec_val,
                    auc_ci=auc_ci,
                    sens_ci=sens_ci,
                    spec_ci=spec_ci,
                    thr_val=thr_val
                )

            y_pred_bin = preds_opt[pred_col].values
            cm = confusion_matrix(y_true, y_pred_bin, labels=[0, 1])
            _draw_confusion_matrix(ax_cm, cm, color, title)

    # ------------------------------
    # layout
    # ------------------------------
    big_title = "Class distribution: " + ", ".join([f"{cls}={cnt}" for cls, cnt in counts.items()])
    fig.suptitle(big_title, fontsize=font_size_big_title, y=0.99)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path / f"plt_summary.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()



# %% Feature importance

import pathlib
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import re


def compute_feature_importance(models_dir: pathlib.Path,
                            features_names:List[str],
                               objective: str,
                               top_n: int = 15):
    """
    Compute and plot feature importance across folds
    for XGBoost models saved as 'fold_{k}_{objective}_model.pkl'.

    Parameters
    ----------
    models_dir : pathlib.Path
        Path to directory with models.
    objective : str
        One of ['auc', 'spec', 'youden'].
    top_n : int
        Number of top features to plot (ranked by gain).
    """

    # filter model files
    model_files = sorted(models_dir.glob(f"fold*_{objective}_model.pkl"))
    if not model_files:
        raise FileNotFoundError(f"No models found for objective '{objective}' in {models_dir}")

    excel_path = models_dir.joinpath(f"feature_importance_{objective}.xlsx")
    plot_path = models_dir.joinpath(f"feature_importance_{objective}.png")

    if features_names:
        features_names = [re.sub(r'_', ' ', fet).title() for fet in features_names]

    importance_records = []
    # --- Load models and extract importance ---
    for model_file in model_files:
        with open(model_file, "rb") as f:
            model = pickle.load(f)

        # unwrap booster if needed
        if hasattr(model, "get_booster"):
            booster = model.get_booster()
        else:
            booster = model
        if booster.feature_names is None:
            booster.feature_names = features_names

        for importance_type in ["weight", "gain", "cover"]:
            imp = booster.get_score(importance_type=importance_type)
            for feat, val in imp.items():
                importance_records.append({
                    "feature": feat,
                    "importance_type": importance_type,
                    "value": val,
                    "model": model_file.stem
                })

    df = pd.DataFrame(importance_records)

    # Pivot to feature × importance_type × folds
    df_pivot = df.pivot_table(
        index=["feature", "importance_type"],
        columns="model",
        values="value"
    )

    # Compute mean ± std across folds
    df_stats = df_pivot.apply([pd.Series.mean, pd.Series.std], axis=1).reset_index()
    df_stats = df_stats.rename(columns={"mean": "mean", "std": "std"})

    # Reshape for plotting
    df_importance = df_stats.pivot(
        index="feature",
        columns="importance_type",
        values=["mean", "std"]
    )

    # Save Excel
    df_importance.to_excel(excel_path)
    print(f"Saved feature importance table to {excel_path}")

    # --- Plot ---
    importance_types = ["weight", "gain", "cover"]
    titles = {
        "weight": "Frequency (Weight)",
        "gain": "Split Gain (Gain)",
        "cover": "Coverage (Cover)"
    }

    sns.set_style("whitegrid")
    colors = sns.color_palette("pastel", n_colors=3)

    # Rank features by gain
    top_features = df_importance["mean"]["gain"].sort_values(ascending=False).head(top_n).index

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
    for idx, imp_type in enumerate(importance_types):
        ax = axes[idx]
        means = df_importance["mean"][imp_type].loc[top_features]
        stds = df_importance["std"][imp_type].loc[top_features]

        ax.barh(top_features, means, xerr=stds,
                capsize=4, color=colors[idx], edgecolor="black")

        ax.invert_yaxis()
        ax.set_title(titles[imp_type], fontsize=16, weight="bold")
        ax.set_xlabel("Importance (mean ± std)", fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.grid(True, linestyle="--", alpha=0.6, axis="x")

    fig.suptitle(f"XGBoost Feature Importance ({objective.capitalize()} Optimization)",
                 fontsize=18, weight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.savefig(plot_path, dpi=300)
    plt.show()

    plt.close()
    print(f"Saved feature importance plot to {plot_path}")





