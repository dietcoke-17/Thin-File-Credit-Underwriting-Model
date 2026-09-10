"""Phase 3: Weight of Evidence (WoE) binning and Information Value (IV).

WoE binning is the industry-standard way to turn a raw feature into
something a logistic regression can use monotonically and interpretably.
IV then summarizes how predictive the whole feature is, features below
IV_THRESHOLD are dropped before modeling.

IV interpretation (standard credit-risk convention):
    < 0.02          not useful
    0.02 - 0.1      weak
    0.1  - 0.3      medium
    0.3  - 0.5      strong
    > 0.5           suspiciously strong, check for leakage
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config import IV_THRESHOLD, WOE_BINS


def calculate_woe_iv(
    data: pd.DataFrame, feature: str, target: str = "loan_default", bins: int = WOE_BINS
) -> Tuple[pd.DataFrame, float]:
    """Bin a feature (qcut for numeric, as-is for categorical) and compute
    WoE/IV per bin.

    Returns (woe_table, total_iv).
    """
    tmp = data[[feature, target]].copy()
    if pd.api.types.is_numeric_dtype(tmp[feature]) and tmp[feature].nunique() > bins:
        tmp["bin"] = pd.qcut(tmp[feature], q=bins, duplicates="drop")
    else:
        tmp["bin"] = tmp[feature]

    grouped = tmp.groupby("bin", observed=True)[target].agg(["count", "sum"])
    grouped.columns = ["total", "bad"]
    grouped["good"] = grouped["total"] - grouped["bad"]

    total_good = grouped["good"].sum()
    total_bad = grouped["bad"].sum()

    # Small floor to avoid log(0) in sparse bins.
    grouped["good_pct"] = (grouped["good"] / total_good).replace(0, 0.0001)
    grouped["bad_pct"] = (grouped["bad"] / total_bad).replace(0, 0.0001)
    grouped["woe"] = np.log(grouped["good_pct"] / grouped["bad_pct"])
    grouped["iv"] = (grouped["good_pct"] - grouped["bad_pct"]) * grouped["woe"]

    return grouped, grouped["iv"].sum()


def transform_to_woe(
    data: pd.DataFrame, feature: str, woe_table: pd.DataFrame, bins: int = WOE_BINS
) -> pd.Series:
    """Map each row's raw feature value to the WoE of the bin it falls into."""
    tmp = data[[feature]].copy()
    if pd.api.types.is_numeric_dtype(tmp[feature]) and tmp[feature].nunique() > bins:
        tmp["bin"] = pd.qcut(tmp[feature], q=bins, duplicates="drop")
    else:
        tmp["bin"] = tmp[feature]
    return tmp["bin"].map(woe_table["woe"].to_dict())


def select_features(
    data: pd.DataFrame, candidate_features: List[str], target: str = "loan_default"
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], List[str]]:
    """Compute IV for every candidate feature and keep those at or above
    IV_THRESHOLD.

    Returns (iv_summary_df, woe_tables_by_feature, selected_feature_names).
    """
    iv_summary, woe_tables = {}, {}
    for feature in candidate_features:
        woe_table, iv = calculate_woe_iv(data, feature, target=target)
        iv_summary[feature] = iv
        woe_tables[feature] = woe_table

    iv_df = pd.DataFrame(list(iv_summary.items()), columns=["feature", "IV"]).sort_values(
        "IV", ascending=False
    )
    selected = iv_df.loc[iv_df["IV"] >= IV_THRESHOLD, "feature"].tolist()
    return iv_df.reset_index(drop=True), woe_tables, selected


def build_woe_dataset(
    data: pd.DataFrame,
    selected_features: List[str],
    woe_tables: Dict[str, pd.DataFrame],
    target: str = "loan_default",
) -> pd.DataFrame:
    """Assemble the modeling dataset: customer_id, one *_woe column per
    selected feature, and the target.
    """
    woe_df = pd.DataFrame({"customer_id": data["customer_id"]})
    for feature in selected_features:
        woe_df[f"{feature}_woe"] = transform_to_woe(data, feature, woe_tables[feature])
    woe_df[target] = data[target].values
    return woe_df.dropna().reset_index(drop=True)
