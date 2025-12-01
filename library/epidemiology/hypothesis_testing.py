"""
Hypothesis testing for between group comparisons. Useful to perform the stat test for table one
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy.stats import fisher_exact, kruskal, mannwhitneyu, shapiro, ttest_ind, chi2_contingency
import warnings

# ---------- Shared helpers (align with your ordinal helpers) ----------
def _or_ci_woolf(a, b, c, d, haldane: bool = True) -> Tuple[float, float, float]:
    """
    Odds ratio and 95% CI using Woolf's method.
    If any cell is 0 and haldane=True, applies Haldane–Anscombe correction (+0.5 to all cells).
    Table layout:
        [[a, b],
         [c, d]]
        rows = group0, group1 ; columns = no(0), yes(1)
    """
    aa, bb, cc, dd = a, b, c, d
    if haldane and (aa == 0 or bb == 0 or cc == 0 or dd == 0):
        aa += 0.5; bb += 0.5; cc += 0.5; dd += 0.5

    # OR = (a*d)/(b*c)
    if bb * cc == 0:
        return (np.nan, np.nan, np.nan)
    or_val = (aa * dd) / (bb * cc)

    # SE(log OR) = sqrt(1/a + 1/b + 1/c + 1/d)
    se = np.sqrt(1/aa + 1/bb + 1/cc + 1/dd)
    log_or = np.log(or_val)
    ci_low = np.exp(log_or - 1.96 * se)
    ci_high = np.exp(log_or + 1.96 * se)
    return (or_val, ci_low, ci_high)

def _is_strict_binary(x: pd.Series) -> bool:
    vals = pd.Series(x.dropna().unique())
    try:
        return set(vals.astype(int)) == {0, 1} and vals.size <= 2
    except Exception:
        return False

def _rank_biserial_from_u(u: float, n0: int, n1: int) -> float:
    """
     r_rb = 1 - 2U/(n0*n1)  (equivalent to 2A - 1, where A is Vargha–Delaney A)
     Because we compute the two-sided this method is directionless
    :param u:
    :param n0:
    :param n1:
    :return:
    """
    return 1.0 - 2.0 * (u / (n0 * n1))


def rank_biserial_from_samples(x, y):
    # U for "x > y"

    Ux = mannwhitneyu(x, y, alternative='greater').statistic
    n0, n1 = len(x), len(y)
    return (2 * Ux / (n0 * n1)) - 1  # positive when x > y


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    # O(N*M); fine for typical sample sizes
    # computed as P(x>y) − P(x<y) (directional, positive when x>y).
    gt = 0
    lt = 0
    for a in x:
        lt += np.sum(a < y)
        gt += np.sum(a > y)
    n0, n1 = len(x), len(y)
    return (gt - lt) / (n0 * n1)

def _holms_correction(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = [0.0] * m
    prev = 0.0
    for k, idx in enumerate(order):
        adj = (m - k) * pvals[idx]
        adj = max(adj, prev)  # ensure monotonicity
        adjusted[idx] = min(adj, 1.0)
        prev = adjusted[idx]
    return adjusted

def _hedges_g(mean0, mean1, sd0, sd1, n0, n1) -> float:
    # Cohen's d (pooled) with small-sample correction J -> Hedges' g
    df = n0 + n1 - 2
    if df <= 0:
        return np.nan
    s_pooled_sq = (((n0 - 1) * (sd0 ** 2)) + ((n1 - 1) * (sd1 ** 2))) / df
    s_pooled = np.sqrt(s_pooled_sq) if s_pooled_sq > 0 else np.nan
    d = (mean1 - mean0) / s_pooled if (s_pooled is not np.nan and s_pooled > 0) else np.nan
    # J correction
    J = 1.0 - (3.0 / (4.0 * df - 1.0)) if df > 1 else 1.0
    return d * J if not np.isnan(d) else np.nan

def _normal_shapiro(x: np.ndarray, alpha: float = 0.05) -> Optional[bool]:
    # Shapiro is defined for 3<=n<=5000; outside that range, skip and return None
    n = len(x)
    if n < 3 or n > 5000:
        return None
    try:
        return shapiro(x).pvalue > alpha
    except Exception:
        return None

def _iqr(series: np.ndarray) -> Tuple[float, float, float]:
    q1, q2, q3 = np.percentile(series, [25, 50, 75])
    return q2, q1, q3  # median, Q1, Q3

def _mean_sd(x: np.ndarray) -> Tuple[float, float]:
    return (np.nanmean(x), np.nanstd(x, ddof=1)) if len(x) > 0 else (np.nan, np.nan)


def _pformat(p: float) -> str:
    if p is None or (isinstance(p, float) and (np.isnan(p) or np.isinf(p))):
        return "nan"
    if p >= 1e-4:
        return f"{p:.4f}"
    else:
        # option 1
        f"{p:.4e}"
        # option 2
        # Format in LaTeX: a × 10^{b} without e-notation
        a, b = f"{p:.1e}".split("e")
        return f"{float(a):.0f} × 10^{{{int(b)}}}"

# %% Hypothesis test Binary
def stats_test_binary_symptoms(
    data: pd.DataFrame,
    columns: List[str],
    strata_col: str = 'NT1',
    positive_value: int = 1,
    SHOW: Optional[bool] = False
) -> pd.DataFrame:
    """
    Binary comparisons between two groups (2x2):
      - Chooses Fisher's Exact Test if any expected cell < 5, else Chi-square.
      - Reports % yes with counts in each group.
      - Reports Odds Ratio with 95% CI (Woolf; Haldane–Anscombe for zeros).

    Parameters
    ----------
    data : DataFrame with binary columns and a grouping column.
    columns : Variables to test (binary).
    strata_col : Grouping column (must have exactly two unique non-NaN values).
    positive_value : Value treated as “yes” (default 1).
    SHOW : If True, prints the 2x2 table per variable.

    Returns
    -------
    DataFrame with:
      - Variable
      - {strata_col}={group0} (% yes with counts)
      - {strata_col}={group1} (% yes with counts)
      - p-value, p-value formatted
      - Effect Size (Odds Ratio 95% CI)
      - Stat Method
    """
    # Validate groups
    unique_strata = pd.Series(data[strata_col]).dropna().unique()
    if len(unique_strata) != 2:
        raise ValueError(f"Strata column '{strata_col}' must have exactly 2 unique values.")
    group0, group1 = np.sort(unique_strata)

    out = []
    for col in columns:
        if col == strata_col:
            continue

        df = data[[col, strata_col]].dropna().copy()

        # Enforce strict binary (0/1); if not, try to coerce equal to positive_value
        if _is_strict_binary(df[col]):
            bin_col = df[col].astype(int)
        else:
            # Coerce: mark positive_value as 1, everything else as 0
            bin_col = (df[col] == positive_value).astype(int)
            warnings.warn(f"Coercing positive values, Binary array not strictly binary {col}", UserWarning)

        df = df.assign(__y__=bin_col)

        # Ns per group
        n0 = (df[strata_col] == group0).sum()
        n1 = (df[strata_col] == group1).sum()

        if n0 == 0 or n1 == 0:
            out.append({
                'Variable': str(col),
                f'{strata_col}={group0}': f'0/0',
                f'{strata_col}={group1}': f'0/0',
                'p-value': np.nan,
                'p-value formatted': 'nan',
                'Effect Size (Odds Ratio 95% CI)': np.nan,
                'Stat Method': 'None (empty group)'
            })
            continue

        # Counts of "yes"
        b = df.loc[df[strata_col] == group0, '__y__'].sum()  # controls yes
        d = df.loc[df[strata_col] == group1, '__y__'].sum()  # cases yes
        a = n0 - b  # controls no
        c = n1 - d  # cases no

        table = np.array([[a, b],
                          [c, d]], dtype=float)

        if SHOW:
            print(
                f"\n{col} | {strata_col}={group0} vs {strata_col}={group1}\n"
                f"[[no, yes],[no, yes]] = {table.tolist()}"
            )

        # Decide test based on expected counts
        # Decide test robustly
        use_fisher = False
        try:
            chi2, p_chi2, dof, expected = chi2_contingency(table, correction=False)
            if expected.min() < 5:
                use_fisher = True
        except ValueError:
            # chi2_contingency failed due to zero expected counts → fallback to Fisher
            use_fisher = True

        if use_fisher:
            or_est, p_val = fisher_exact(table, alternative='two-sided')
            method = "Fisher's Exact Test"
            # CI from Woolf (with Haldane) for reporting consistency
            or_w, ci_l, ci_h = _or_ci_woolf(a, b, c, d, haldane=True)
            # Prefer Fisher's OR if finite; use Woolf/Haldane for CI display
            or_final = or_est if np.isfinite(or_est) else or_w
        else:
            p_val = p_chi2
            method = "Chi-Square Test"
            or_final, ci_l, ci_h = _or_ci_woolf(a, b, c, d, haldane=True)

        # Percents
        pct0 = 100.0 * b / n0 if n0 > 0 else np.nan
        pct1 = 100.0 * d / n1 if n1 > 0 else np.nan

        out.append({
            'Variable': str(col),
            f'{strata_col}={group0}': f'{pct0:.1f}% ({b}/{n0})',
            f'{strata_col}={group1}': f'{pct1:.1f}% ({d}/{n1})',
            'p-value': p_val,
            'p-value formatted': _pformat(p_val),
            'Effect Size (Odds Ratio 95% CI)': (
                f"{(np.round(or_final, 3) if pd.notna(or_final) else np.nan)} "
                f"[{(np.round(ci_l, 2) if pd.notna(ci_l) else np.nan)}, "
                f"{(np.round(ci_h, 2) if pd.notna(ci_h) else np.nan)}]"
            ),
            'Stat Method': method
        })

    return pd.DataFrame(out)

# %% Hypothesis test continous
def stats_test_continuous(
    data: pd.DataFrame,
    columns: List[str],
    strata_col: str = 'NT1',
    SHOW: Optional[bool] = False
) -> pd.DataFrame:
    """
    Continuous-variable tests between two groups:
      - If both groups approximately normal (Shapiro where valid): Welch's t-test + Hedges' g
      - Otherwise: Mann–Whitney U + rank-biserial r (and Cliff's delta)

    Output includes Mean±SD AND Median [IQR] for each group, p-values (raw & formatted),
    effect size (method-dependent), and chosen statistical method.
    """
    # Validate groups
    unique_groups = pd.Series(data[strata_col]).dropna().unique()
    if len(unique_groups) != 2:
        raise ValueError(f"Strata column '{strata_col}' must have exactly 2 unique values.")
    group0, group1 = np.sort(unique_groups)

    rows = []
    for col in columns:
        if col == strata_col:
            continue

        df = data[[col, strata_col]].dropna()
        x = df.loc[df[strata_col] == group0, col].to_numpy()
        y = df.loc[df[strata_col] == group1, col].to_numpy()
        n0, n1 = len(x), len(y)

        # Descriptives
        mean0, sd0 = _mean_sd(x)
        mean1, sd1 = _mean_sd(y)
        med0, q10, q30 = _iqr(x) if n0 > 0 else (np.nan, np.nan, np.nan)
        med1, q11, q31 = _iqr(y) if n1 > 0 else (np.nan, np.nan, np.nan)

        # Early exit for very small samples
        if n0 < 3 or n1 < 3:
            rows.append({
                'Variable': col,
                f'{strata_col}={group0} Mean (SD)': f'{mean0:.2f}±{sd0:.2f} (n={n0})',
                f'{strata_col}={group1} Mean (SD)': f'{mean1:.2f}±{sd1:.2f} (n={n1})',
                f'{strata_col}={group0} Median [IQR]': f'{med0:.2f} [{q10:.2f}, {q30:.2f}]',
                f'{strata_col}={group1} Median [IQR]': f'{med1:.2f} [{q11:.2f}, {q31:.2f}]',
                'p-value': np.nan,
                'p-value formatted': 'nan',
                'Effect Size': np.nan,
                'Effect Size (name)': np.nan,
                'Stat Method': 'Insufficient n (<3 in a group)'
            })
            if SHOW:
                print(f"{col}: insufficient n (n0={n0}, n1={n1})")
            continue

        # Normality checks (may be None if out-of-range)
        norm0 = _normal_shapiro(x)
        norm1 = _normal_shapiro(y)

        # Decide method:
        # If both checks are True where evaluated (or both None but we allow Welch as robust),
        # use Welch; else use Mann–Whitney.
        use_welch = False
        if norm0 is None and norm1 is None:
            # Shapiro not applicable (very large samples). Welch is robust -> go with Welch.
            use_welch = True
        elif (norm0 is True) and (norm1 is True):
            use_welch = True
        else:
            use_welch = False

        if use_welch:
            # Welch's t-test (unequal variances)
            t_stat, p_val = ttest_ind(x, y, equal_var=False)
            g = _hedges_g(mean0, mean1, sd0, sd1, n0, n1)
            eff_name = "Hedges' g"
            eff = None if np.isnan(g) else round(float(g), 3)
            method = "Welch's t-test"
        else:
            # Mann–Whitney U
            U, p_val = mannwhitneyu(x, y, alternative='two-sided')
            # r_rb = _rank_biserial_from_u(U, n0, n1)
            r_rb = rank_biserial_from_samples(x, y)
            cd = _cliffs_delta(x, y)
            eff_name = "rank-biserial r (Cliff's δ)"
            # Report r, include δ in parentheses
            eff = f"{round(float(r_rb), 3)} ({round(float(cd), 3)})" if not (np.isnan(r_rb) or np.isnan(cd)) else (
                  (round(float(r_rb), 3) if not np.isnan(r_rb) else np.nan)
            )
            method = "Mann-Whitney U"

        p_fmt = _pformat(p_val)

        rows.append({
            'Variable': col,
            f'{strata_col}={group0} Mean (SD)': f'{mean0:.2f}±{sd0:.2f} (n={n0})',
            f'{strata_col}={group1} Mean (SD)': f'{mean1:.2f}±{sd1:.2f} (n={n1})',
            f'{strata_col}={group0} Median [IQR]': f'{med0:.2f} [{q10:.2f}, {q30:.2f}]',
            f'{strata_col}={group1} Median [IQR]': f'{med1:.2f} [{q11:.2f}, {q31:.2f}]',
            'p-value': p_val,
            'p-value formatted': p_fmt,
            'Effect Size': eff,
            'Effect Size (name)': eff_name,
            'Stat Method': method,
            'Normality (Shapiro)': f"{strata_col}={group0}:{norm0} | {strata_col}={group1}:{norm1}"
        })

        if SHOW:
            print(
                f"{col}: {method} | "
                f"{strata_col}={group0} → {mean0:.2f}±{sd0:.2f} (n={n0}); "
                f"{strata_col}={group1} → {mean1:.2f}±{sd1:.2f} (n={n1}); "
                f"p={p_fmt}; ES[{eff_name}]={eff}"
            )

    return pd.DataFrame(rows)

# %% Hypothesis test ordinal
def stats_test_ordinal(
    data: pd.DataFrame,
    columns: List[str],
    strata_col: str,
    order_map: Optional[Dict[str, List]] = None,
    pairwise: bool = False,
    SHOW: bool = False
) -> pd.DataFrame:
    """
    Hypothesis testing for ORDINAL outcomes across groups.
      - If exactly 2 groups: Mann–Whitney U (two-sided), with rank-biserial r and Cliff's delta.
      - If >2 groups: Kruskal–Wallis H test. Optionally pairwise Mann–Whitney with Holm correction.

    Parameters
    ----------
    data : DataFrame containing ordinal columns and a grouping column.
    columns : List of ordinal columns to test.
    strata_col : Grouping column (>=2 unique values).
    order_map : Optional dict {col: ordered_levels_list} to coerce proper ordering for non-numeric ordinals.
    pairwise : If True and groups>2, performs pairwise MWU with Holm correction (wide result).
    SHOW : Print per-variable summaries.

    Returns
    -------
    DataFrame with per-variable results. For 2 groups: medians (IQR) per group, p, effect sizes.
    For >2 groups: Kruskal–Wallis p; if pairwise=True, includes Holm-adjusted pairwise p-values.
    """
    groups = [g for g in pd.Series(data[strata_col]).dropna().unique()]
    ng = len(groups)
    if ng < 2:
        raise ValueError(f"'{strata_col}' must have at least 2 groups.")

    # stable ordering for group labels
    groups = np.sort(groups)

    rows = []
    for col in columns:
        if col == strata_col:
            continue

        df = data[[col, strata_col]].dropna().copy()

        # Coerce ordinal values to numeric codes (while preserving order)
        s = df[col]
        if isinstance(s, pd.CategoricalDtype):
            if s.cat.ordered:
                vals = s.cat.codes.to_numpy()
                lvl_map = dict(zip(s.cat.categories, range(len(s.cat.categories))))
            else:
                # unordered categorical: impose order from order_map or alphabetical
                levels = order_map.get(col) if order_map and col in order_map else sorted(s.cat.categories)
                cat = pd.Categorical(s.astype(str), categories=levels, ordered=True)
                vals = cat.codes.to_numpy()
                lvl_map = dict(zip(levels, range(len(levels))))
        elif np.issubdtype(s.dtype, np.number):
            # assume numeric encodes order already
            vals = s.to_numpy()
            lvl_map = None
        else:
            # strings: use order_map or sorted unique
            levels = order_map.get(col) if order_map and col in order_map else sorted(s.astype(str).unique())
            cat = pd.Categorical(s.astype(str), categories=levels, ordered=True)
            # vals = cat.codes # .to_numpy()
            # lvl_map = dict(zip(levels, range(len(levels))))

        df = df.assign(__ord__=vals)

        # Split by group
        grp_vals = [df.loc[df[strata_col] == g, "__ord__"].to_numpy() for g in groups]
        ns = [len(v) for v in grp_vals]

        # Summaries (median [IQR]) per group
        summaries = {}
        for g, v in zip(groups, grp_vals):
            if len(v) > 0:
                med, q1, q3 = _iqr(v)
                summaries[g] = f"{med:.2f} [{q1:.2f}, {q3:.2f}] (n={len(v)})"
            else:
                summaries[g] = "nan [nan, nan] (n=0)"

        if ng == 2:
            x, y = grp_vals[0], grp_vals[1]
            if len(x) == 0 or len(y) == 0:
                p = np.nan
                r_rb = np.nan
                cd = np.nan
            else:
                U, p = mannwhitneyu(x, y, alternative='two-sided')
                # r_rb = _rank_biserial_from_u(U, len(x), len(y))
                r_rb = rank_biserial_from_samples(x,y)

                cd = _cliffs_delta(x, y)
            # p_fmt = f"{p:.4f}" if pd.notna(p) and p >= 1e-4 else (f"{p:.4e}" if pd.notna(p) else "nan")
            p_fmt = _pformat(p)
            row = {
                "Variable": col,
                f"{strata_col}={groups[0]} Median [IQR]": summaries[groups[0]],
                f"{strata_col}={groups[1]} Median [IQR]": summaries[groups[1]],
                "p-value": p,
                "p-value formatted": p_fmt,
                "Effect Size (rank-biserial r)": np.round(r_rb, 3) if pd.notna(r_rb) else np.nan,
                "Effect Size (Cliffs delta)": np.round(cd, 3) if pd.notna(cd) else np.nan,
                "Stat Method": "Mann-Whitney U (ordinal, two-sided)",
            }
            rows.append(row)

            if SHOW:
                print(f"{col}: MWU p={p_fmt}; r_rb={row['Effect Size (rank-biserial r)']}; "
                      f"Cliff's δ={row['Effect Size (Cliffs delta)']}")

            else:
                # Kruskal–Wallis across >2 groups
                if any(n == 0 for n in ns):
                    H, p = np.nan, np.nan
                else:
                    H, p = kruskal(*grp_vals, nan_policy='omit')
                # p_fmt = f"{p:.4f}" if pd.notna(p) and p >= 1e-4 else (f"{p:.4e}" if pd.notna(p) else "nan")
                p_fmt = _pformat(p)
                row = {
                    "Variable": col,
                    **{f"{strata_col}={g} Median [IQR]": summaries[g] for g in groups},
                    "p-value": p,
                    "p-value formatted": p_fmt,
                    "Effect Size (rank-biserial r)": np.nan,
                    "Effect Size (Cliff's delta)": np.nan,
                    "Stat Method": "Kruskal-Wallis (ordinal)"
                }

            # Optional pairwise with Holm correction
            if pairwise and all(len(v) > 0 for v in grp_vals):
                pairs = []
                pvals = []
                for i in range(ng):
                    for j in range(i+1, ng):
                        Ui, pij = mannwhitneyu(grp_vals[i], grp_vals[j], alternative='two-sided')
                        pairs.append((groups[i], groups[j]))
                        pvals.append(pij)
                p_adj = _holms_correction(pvals)
                for (g0, g1), pa in zip(pairs, p_adj):
                    key = f"Pairwise MWU p (Holm): {g0} vs {g1}"
                    row[key] = pa
            rows.append(row)

            if SHOW:
                print(f"{col}: Kruskal–Wallis p={p_fmt}" + (" with pairwise Holm" if pairwise else ""))

    return pd.DataFrame(rows)

# %% callable function
def stats_test(data:pd.DataFrame,
               columns:List[str],
               var_type:str,
               strata_col:str='NT1', **kwargs):
    """
    Dispatch hypothesis testing based on variable type.
    var_type: 'binary' | 'continuous' | 'ordinal'
    kwargs are forwarded (e.g., SHOW, order_map, pairwise, positive_value).
    """
    vt = str(var_type).lower()
    if vt == 'binary':
        return stats_test_binary_symptoms(data=data, columns=columns, strata_col=strata_col, **kwargs)
    elif vt == 'continuous':
        return stats_test_continuous(data=data, columns=columns, strata_col=strata_col, **kwargs)
    elif vt == 'ordinal':
        return stats_test_ordinal(data=data, columns=columns, strata_col=strata_col, **kwargs)
    else:
        raise ValueError("var_type must be one of: 'binary', 'continuous', 'ordinal'")

