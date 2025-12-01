"""
Compute the table with hypothesis testing
"""
import pathlib
import pandas as pd
from config.config import config
from statsmodels.stats.power import GofChisquarePower
from statsmodels.stats.gof import chisquare_effectsize
from scipy.stats import chi2_contingency


def plan_sample_size(effect_sizes,
                     alpha=0.05,
                     power_target=0.80,
                     n_bins=3):
    """
    Compute required sample sizes for a chi-square test
    given effect sizes, alpha, power, and number of bins.

    Parameters
    ----------
    effect_sizes : dict
        Dictionary of labels -> Cohen's w values
        e.g., {"small": 0.1, "medium": 0.3, "large": 0.5}
    alpha : float
        Significance level (default=0.05).
    power_target : float
        Desired power (default=0.80).
    n_bins : int
        Number of response categories (bins).

    Returns
    -------
    pd.DataFrame
        Table of required sample sizes for each effect size.
    """
    analysis = GofChisquarePower()
    results = []

    for label, w in effect_sizes.items():
        n_required = analysis.solve_power(effect_size=w,
                                          nobs=None,
                                          alpha=alpha,
                                          power=power_target,
                                          n_bins=n_bins)
        results.append({
            "Effect size label": label,
            "Cohen w": w,
            "Required N (total sample)": round(n_required)
        })

    return pd.DataFrame(results)



# %% Main
if __name__ == "__main__":
    # %% Read data
    df_data = pd.read_csv(config.get('data_path').get('pp_questionnaire'))
    strata_diagnosis = 'diagnosis'

    # %% columns of interest for power size
    # Answer the question if we collected enought sample to differentiate between cases and controls
    col_qusst = [col for col in df_data.columns if col.startswith("q")]
    df_data = df_data[col_qusst + [strata_diagnosis]]

    # %% Power Size: Planning Stage

    # Example 1: Using Cohen's benchmarks
    effect_sizes = {"small": 0.1, "medium": 0.3, "large": 0.5}
    results_df = plan_sample_size(effect_sizes)
    print(results_df)

    # Example 2: Using your observed effect sizes (pilot data)
    observed_effects = {
        "q1_rbd": 0.97,
        "q2_smell": 0.36,
        "q4_constipation": 0.21,
        "q5_orthostasis": 0.10
    }
    observed_df = plan_sample_size(observed_effects)
    print(observed_df)


    # %% Post collection power size calculation
    # Calculate sample size
    analysis = GofChisquarePower()
    alpha = 0.05
    results = []

    for q in col_qusst:
        # contingency table: diagnosis x responses
        table = pd.crosstab(df_data['diagnosis'], df_data[q])

        # chi-square test
        chi2, p, dof, expected = chi2_contingency(table)

        # effect size
        obs = (table / table.values.sum()).values.flatten()
        exp = (expected / expected.sum()).flatten()
        w = chisquare_effectsize(obs, exp)

        # achieved power
        nobs = table.values.sum()
        power = analysis.solve_power(effect_size=w,
                                     nobs=nobs,
                                     alpha=alpha,
                                     power=None,
                                     n_bins=table.size)

        results.append({
            "question": q,
            "nobs": nobs,
            "chi2": chi2,
            "p_value": p,
            "effect_size_w": w,
            "achieved_power": power
        })

    results_df = pd.DataFrame(results)
