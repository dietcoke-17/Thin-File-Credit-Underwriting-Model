"""Phase 2: Statistical validation.

Before a feature is trusted in a scorecard it needs to demonstrate it
actually separates good customers from bad ones. Chi-square tests
categorical/binned features for independence from the target; the
Kolmogorov-Smirnov test measures distributional separation for
continuous features.
"""

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
