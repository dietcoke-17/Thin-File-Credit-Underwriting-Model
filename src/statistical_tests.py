"""Phase 2: Statistical validation.

Before a feature is trusted in a scorecard it needs to demonstrate it
actually separates good customers from bad ones. Chi-square tests
categorical/binned features for independence from the target; the
Kolmogorov-Smirnov test measures distributional separation for
continuous features. validate_features runs both across every
candidate feature at once, this is the step that actually isolates
which alternate-data metrics correlate with repayment, rather than
just demonstrating the test mechanics on one feature.
"""

from typing import List

import pandas as pd
from scipy import stats


def chi_square_test(data: pd.DataFrame, cat_col: str, target_col: str = "loan_default"):
    """Chi-square test of independence between a binned/categorical
    feature and the default flag.

    Returns (chi2_statistic, p_value, degrees_of_freedom).
    """
    contingency = pd.crosstab(data[cat_col], data[target_col])
    chi2, p_value, dof, _expected = stats.chi2_contingency(contingency)
    return chi2, p_value, dof


def ks_test(data: pd.DataFrame, cont_col: str, target_col: str = "loan_default"):
    """KS test of separation between the Good and Bad distributions of a
    continuous feature.

    Returns (ks_statistic, p_value).
    """
    good = data.loc[data[target_col] == 0, cont_col]
    bad = data.loc[data[target_col] == 1, cont_col]
    return stats.ks_2samp(good, bad)


def validate_features(
    data: pd.DataFrame,
    candidate_features: List[str],
    target: str = "loan_default",
    bins: int = 10,
) -> pd.DataFrame:
    """Run chi-square (on quantile-binned values) and KS (on raw values)
    for every candidate feature in one pass.

    This is the actual feature-isolation step: a feature with a
    significant chi-square p-value and a nontrivial KS statistic is
    demonstrably linked to repayment, one with neither is a candidate
    to drop regardless of what the IV score alone says.

    Returns a DataFrame with one row per feature: chi2, chi2_p, ks_stat, ks_p.
    """
    rows = []
    for feature in candidate_features:
        tmp = data[[feature, target]].copy()
        if pd.api.types.is_numeric_dtype(tmp[feature]) and tmp[feature].nunique() > bins:
            tmp["_bin"] = pd.qcut(tmp[feature], q=bins, duplicates="drop")
        else:
            tmp["_bin"] = tmp[feature]

        chi2, chi2_p, _dof = chi_square_test(tmp, "_bin", target_col=target)
        ks_stat, ks_p = ks_test(data, feature, target_col=target)

        rows.append(
            {
                "feature": feature,
                "chi2": chi2,
                "chi2_p": chi2_p,
                "ks_stat": ks_stat,
                "ks_p": ks_p,
                "significant": bool(chi2_p < 0.05 and ks_p < 0.05),
            }
        )

    return pd.DataFrame(rows).sort_values("ks_stat", ascending=False).reset_index(drop=True)

