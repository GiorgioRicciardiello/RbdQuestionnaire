"""
Use the preprocess dataset and generate the table one.
"""
from config.config import config, config_actigraphy
import numpy as np
import pandas as pd
from typing import List, Optional
from scipy.stats import fisher_exact, mannwhitneyu, shapiro, ttest_ind, chi2_contingency
from tabulate import tabulate
from statsmodels.stats.multitest import multipletests


def stats_test_binary_symptoms(
    data: pd.DataFrame,
    columns: List[str],
    strata_col: str = 'NT1',
    SHOW: Optional[bool] = False
) -> pd.DataFrame:
    """
    Perform Chi-Square or Fisher's Exact Test for binary (0/1) variables between two groups.

    Returns
    -------
    pd.DataFrame with:
    - 'Variable'
    - percent and count "yes" in each group
    - p-values
    - Odds Ratio (with 95% CI)
    - Statistical test used
    """
    unique_strata = data[strata_col].dropna().unique()
    if len(unique_strata) != 2:
        raise ValueError(f"Strata column '{strata_col}' must have exactly 2 unique values.")
    group0, group1 = np.sort(unique_strata)

    results = []

    for col in columns:
        if col == strata_col:
            continue

        # Only keep rows with non-missing values for both col and strata_col
        df = data[[col, strata_col]].dropna()

        if not set(df[col].unique()).issubset({0, 1}):
            continue  # Skip non-binary

        # Get group-specific counts *only for non-NaN rows*
        group0_df = df[df[strata_col] == group0]
        group1_df = df[df[strata_col] == group1]

        n0 = len(group0_df)
        n1 = len(group1_df)
        b = group0_df[col].sum()  # yes in group0
        d = group1_df[col].sum()  # yes in group1
        a = n0 - b                # no in group0
        c = n1 - d                # no in group1

        # Skip if any cell in 2x2 table is missing
        if n0 == 0 or n1 == 0:
            results.append({
                'Variable': col,
                f'{strata_col}={group1}': f'NA',
                f'{strata_col}={group0}': f'NA',
                'p-value': np.nan,
                'p-value formatted': np.nan,
                'Effect Size (Odds Ratio 95% CI)': np.nan,
                'Stat Method': 'Insufficient data'
            })
            continue

        table = [[a, b], [c, d]]

        if SHOW:
            print(tabulate(table, headers=[f"{col} = 0", f"{col} = 1"],
                           showindex=[f'{strata_col}={group0}', f'{strata_col}={group1}'],
                           tablefmt="grid"))

        # Choose Fisher if expected count < 5
        chi2_stat, p_chi2, _, expected = chi2_contingency(table)
        use_fisher = expected.min() < 5

        if use_fisher:
            odds_ratio, p_value = fisher_exact(table)
            method = "Fisher's Exact Test"
        else:
            p_value = p_chi2
            odds_ratio = (a * d) / (b * c) if b * c != 0 else np.nan
            method = "Chi-Square Test"

        # Confidence interval for OR if all cells > 0
        if all(x > 0 for x in [a, b, c, d]):
            se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
            log_or = np.log(odds_ratio)
            ci_low = np.exp(log_or - 1.96 * se)
            ci_high = np.exp(log_or + 1.96 * se)
        else:
            ci_low = ci_high = np.nan

        pct0 = 100 * b / n0 if n0 > 0 else np.nan
        pct1 = 100 * d / n1 if n1 > 0 else np.nan

        p_fmt = f"{p_value:.4f}" if p_value >= 1e-4 else f"{p_value:.2e}"

        results.append({
            'Variable': col,
            f'{strata_col}={group0}': f'{pct0:.1f}% ({int(b)}/{n0})',
            f'{strata_col}={group1}': f'{pct1:.1f}% ({int(d)}/{n1})',
            'p-value': p_value,
            'p-value formatted': p_fmt,
            'Effect Size (Odds Ratio 95% CI)': f'{round(odds_ratio, 3)} [{ci_low:.2f}, {ci_high:.2f}]' if not np.isnan(odds_ratio) else np.nan,
            'Stat Method': method
        })

    return pd.DataFrame(results)


def stats_test_continuous(
    data: pd.DataFrame,
    columns: List[str],
    strata_col: str = 'NT1',
    SHOW: Optional[bool] = False
) -> pd.DataFrame:
    """
    Perform statistical tests for continuous variables between two groups.
    - First, test each group's values for normality via Shapiro-Wilk.
    - If both groups are approximately normal, run an independent t-test (unequal variance)
      and compute Cohen's d.
    - Otherwise, run a two-sided Mann-Whitney U test and compute rank-biserial r.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing continuous variables and a grouping column.
    columns : List[str]
        List of column names to test.
    strata_col : str, default 'NT1'
        Name of the binary grouping column (must have exactly two unique, non-NaN values).
    SHOW : bool, default False
        If True, print summary statistics, p-values, and effect sizes for each variable.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row corresponds to one tested column and contains:
          - 'Variable'
          - '{strata_col}={group0} Mean (SD)'
          - '{strata_col}={group1} Mean (SD)'
          - 'n {strata_col}={group0}' (sample size in group0)
          - 'n {strata_col}={group1}' (sample size in group1)
          - 'p-value' (numeric)
          - 'p-value formatted' (string with 4 decimals or scientific if < 1e-4)
          - 'Effect Size' (Cohen's d or rank-biserial r)
          - 'Stat Method' ('Independent t-test' or 'Mann-Whitney U test')
    """

    def _rank_biserial(group0, group1):
        # Perform U‐test (two‐sided)
        U, p = mannwhitneyu(group0, group1, alternative='two-sided')
        n0, n1 = len(group0), len(group1)
        # Convert U to Z
        mu_U = n0 * n1 / 2
        sigma_U = np.sqrt(n0 * n1 * (n0 + n1 + 1) / 12)
        Z = (U - mu_U) / sigma_U
        # r_rb = Z / sqrt(N)
        return Z / np.sqrt(n0 + n1), p

    def _cliffs_delta(group0, group1):
        n0, n1 = len(group0), len(group1)
        gt = 0
        lt = 0
        for x in group0:
            for y in group1:
                if x > y:
                    gt += 1
                elif x < y:
                    lt += 1
        return (gt - lt) / (n0 * n1)


    results = []

    # Identify and sort the two unique strata values
    unique_groups = data[strata_col].dropna().unique()
    if len(unique_groups) != 2:
        raise ValueError(f"Strata column '{strata_col}' must have exactly 2 unique values.")
    group0, group1 = np.sort(unique_groups)

    for col in columns:
        if col == strata_col:
            continue

        # Keep only rows where both col and strata_col are non-missing
        df = data[[col, strata_col]].dropna().copy()

        # Extract values for each group
        group0_vals = df.loc[df[strata_col] == group0, col].to_numpy()
        group1_vals = df.loc[df[strata_col] == group1, col].to_numpy()

        # Count observations
        n0, n1 = len(group0_vals), len(group1_vals)

        # Descriptive statistics (use ddof=1 for sample SD)
        mean0, std0 = (np.nan, np.nan) if n0 == 0 else (group0_vals.mean(), group0_vals.std(ddof=1))
        mean1, std1 = (np.nan, np.nan) if n1 == 0 else (group1_vals.mean(), group1_vals.std(ddof=1))

        # If either group has fewer than 10 observations, report NaNs and continue
        if n0 < 10 or n1 < 10:
            results.append({
                'Variable': col,
                f'{strata_col}={group0} Mean (SD)': f'{mean0:.2f}\u00B1{std0:.2f} ({n0})',
                f'{strata_col}={group1} Mean (SD)': f'{mean1:.2f}\u00B1{std1:.2f} ({n1})',
                # 'n ' + f'{strata_col}={group0}': n0,
                # 'n ' + f'{strata_col}={group1}': n1,
                'p-value': np.nan,
                'p-value formatted': np.nan,
                'Effect Size': np.nan,
                'Stat Method': np.nan
            })
            continue

        # Normality check (Shapiro-Wilk). If Shapiro fails or raises error, treat as non-normal.
        try:
            normal0 = shapiro(group0_vals).pvalue > 0.05
        except:
            normal0 = False
        try:
            normal1 = shapiro(group1_vals).pvalue > 0.05
        except:
            normal1 = False

        # Initialize placeholders
        p_value = np.nan
        effect_size = np.nan
        stat_method = ''

        if normal0 and normal1:
            # Independent (Welch's) t-test
            stat_method = 'Independent t-test'
            t_stat, p_value = ttest_ind(group0_vals, group1_vals, equal_var=False)

            # Cohen's d (using pooled standard deviation)
            s_pooled = np.sqrt(
                (((n0 - 1) * std0**2) + ((n1 - 1) * std1**2)) / (n0 + n1 - 2)
            )
            cohen_d = (mean1 - mean0) / s_pooled if s_pooled > 0 else np.nan
            effect_size = round(cohen_d, 3)

        else:
            # Mann-Whitney U test
            stat_method = 'Mann-Whitney U test'
            r_rb, p_value = _rank_biserial(group0_vals, group1_vals)
            effect_size = round(r_rb, 3)

        # Format p-value string
        p_fmt = f"{p_value:.4f}" if p_value >= 0.0001 else f"{p_value:.4e}"

        results.append({
            'Variable': col,
            f'{strata_col}={group0} Mean (SD)': f'{mean0:.2f}\u00B1{std0:.2f} ({n0})',
            f'{strata_col}={group1} Mean (SD)': f'{mean1:.2f}\u00B1{std1:.2f} ({n1})',
            # 'n ' + f'{strata_col}={group0}': n0,
            # 'n ' + f'{strata_col}={group1}': n1,
            'p-value': p_value,
            'p-value formatted': p_fmt,
            'Effect Size': effect_size,
            'Stat Method': stat_method
        })

        if SHOW:
            print(
                f"{col}: {stat_method} | "
                f"{strata_col}={group0} → {mean0:.2f}\u00B1{std0:.2f} (n={n0}); "
                f"{strata_col}={group1} → {mean1:.2f}\u00B1{std1:.2f} (n={n1}); "
                f"p = {p_fmt}; ES = {effect_size}"
            )

    return pd.DataFrame(results)


def stats_test_ordinal_symptoms(
    data: pd.DataFrame,
    columns: List[str],
    strata_col: str = 'NT1',
    SHOW: Optional[bool] = False
) -> pd.DataFrame:
    """
    Perform Mann-Whitney U tests on ordinal categorical symptoms (e.g., 0, 0.5, 1) between two groups.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing ordinal symptom variables and a binary group column.
    columns : List[str]
        List of ordinal columns to test.
    strata_col : str, default 'NT1'
        Column indicating the grouping variable (must have exactly 2 unique values).
    SHOW : bool, default False
        If True, prints summaries and test results.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing:
        - 'Variable'
        - Mean ± SD for each group
        - p-value (raw and formatted)
        - Mann-Whitney U test statistic
        - Rank-biserial effect size
    """
    # Identify the two strata
    groups = data[strata_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError("Strata column must contain exactly two unique values.")
    group0, group1 = sorted(groups)

    results = []

    for col in columns:
        if col == strata_col:
            continue

        df = data[[col, strata_col]].dropna()
        vals0 = df[df[strata_col] == group0][col]
        vals1 = df[df[strata_col] == group1][col]

        n0, n1 = len(vals0), len(vals1)

        if n0 < 10 or n1 < 10:
            results.append({
                'Variable': col,
                f'{strata_col}={group0} Mean (SD)': f'{vals0.mean():.2f}±{vals0.std():.2f} ({n0})',
                f'{strata_col}={group1} Mean (SD)': f'{vals1.mean():.2f}±{vals1.std():.2f} ({n1})',
                'p-value': np.nan,
                'p-value formatted': np.nan,
                'U statistic': np.nan,
                'Effect Size (r_rb)': np.nan,
                'Stat Method': 'Mann-Whitney U (n<10: skipped)'
            })
            continue

        # Mann-Whitney U test
        U, p = mannwhitneyu(vals0, vals1, alternative='two-sided')

        # Rank-biserial correlation (Z / sqrt(N))
        mu_U = n0 * n1 / 2
        sigma_U = np.sqrt(n0 * n1 * (n0 + n1 + 1) / 12)
        Z = (U - mu_U) / sigma_U
        r_rb = Z / np.sqrt(n0 + n1)

        p_fmt = f"{p:.4f}" if p >= 0.0001 else f"{p:.4e}"

        results.append({
            'Variable': col,
            f'{strata_col}={group0} Mean (SD)': f'{vals0.mean():.2f}±{vals0.std():.2f} ({n0})',
            f'{strata_col}={group1} Mean (SD)': f'{vals1.mean():.2f}±{vals1.std():.2f} ({n1})',
            'p-value': p,
            'p-value formatted': p_fmt,
            'U statistic': int(U),
            'Effect Size (r_rb)': round(r_rb, 3),
            'Stat Method': 'Mann-Whitney U test'
        })

        if SHOW:
            print(f"{col}: U = {U}, p = {p_fmt}, r_rb = {r_rb:.3f}")

    return pd.DataFrame(results)


if __name__ == '__main__':
    # %% Read data
    df_data = pd.read_csv(config.get('data_path').get('pp_questionnaire'))
    df_actig = pd.read_csv(config_actigraphy.get('pp_actig'))
    # %% output paths
    base_path = config.get('results_path').get('results').joinpath('table_one')

    if not base_path.exists():
        base_path.mkdir(parents=True, exist_ok=True)

    # %% define variables
    target = 'diagnosis'

    ordinal_var = [col for col in df_data.columns if col.startswith('q')]
    binary_var = ['gender']
    continuous_var = ['age']

    columns = list(set(ordinal_var + binary_var + continuous_var + [target]))

    df_data.reset_index(drop=True, inplace=True)
    df_data = df_data.reindex(sorted(df_data.columns), axis=1)
    print(f'Dataset dimension: {df_data.shape}')

    continuous_var = ['age']
    continuous_var = [var for var in continuous_var if var in df_data]

    variable_definitions = {
        'q1_rbd': 'Q1 RBD',
        'age': 'Age',
        'q4_constipation': 'Q4 Constipation',
        'q2_smell': 'Q2 Smell',
        'q5_orthostasis': 'Q5 Orthostasis',
        'gender': 'Gender',
    }
    # %% compute statistics of significance different groups
    categorical_var = [col for col in df_data.columns if col.startswith('q')]
    categorical_var.append('gender')

    df_stats_bin = stats_test_binary_symptoms(data=df_data,
                               columns=categorical_var,
                               strata_col=target,
                                SHOW=True)


    df_stats_cont = stats_test_continuous(
                data=df_data,
                columns=list(continuous_var),
                strata_col=target,
                SHOW=True)

    df_stats_ord = stats_test_ordinal_symptoms(
                data=df_data,
                columns=list(ordinal_var),
                strata_col=target,
                SHOW=True)


    df_tab_one = pd.concat([df_stats_bin, df_stats_cont, df_stats_ord], axis=0)

    df_tab_one['Variable'] = df_tab_one['Variable'].map(variable_definitions)
    df_tab_one.reset_index(drop=True, inplace=True)

    # %% correct for multiple comparisosn test

    # Apply Benjamini-Hochberg FDR correction
    p_val_idx = df_tab_one[~df_tab_one['p-value'].isna()].index
    raw_p_array = np.array(df_tab_one.loc[p_val_idx,'p-value'] , dtype=float)
    _, p_fdr_array, _, _ = multipletests(raw_p_array, alpha=0.05, method='fdr_bh')

    # Insert FDR-adjusted p-values back into df_results
    df_tab_one['p-value FDR'] = '-'
    df_tab_one.loc[p_val_idx, 'p-value FDR'] = p_fdr_array
    # Format the FDR-adjusted p-values
    formatted_fdr = [
        (f"{pv:.4f}" if pv >= 0.0001 else f"{pv:.4e}") for pv in p_fdr_array
    ]
    df_tab_one.loc[p_val_idx, 'p-value FDR formatted'] = formatted_fdr


    # %% Inlcude the annotation
    def _annotate_name(row):
        base = row["Variable"]
        if row["Stat Method"] == "Mann-Whitney U test":
            return f"{base}^Δ"
        elif row["Stat Method"] == "Chi-Square Test":
            return f"{base}^‡"
        else:
            return f"{base}"


    df_tab_one["Variable"] = df_tab_one.apply(_annotate_name, axis=1)

    # df_tab_one = df_tab_one.sort_values(by='Variable', ascending=True).reset_index(drop=True)

    # %% save the tables
    df_tab_one.to_csv(base_path.joinpath('table_one_reduced.csv'), index=False,)

