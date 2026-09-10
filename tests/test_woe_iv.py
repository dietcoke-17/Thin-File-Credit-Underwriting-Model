import numpy as np
import pandas as pd
import pytest

from src.woe_iv import build_woe_dataset, calculate_woe_iv, select_features, transform_to_woe


def _predictive_df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    strong_feature = rng.uniform(0, 1, size=n)
    # default probability rises sharply with strong_feature -> should get high IV
    default = rng.binomial(1, np.clip(strong_feature, 0.02, 0.98))
    noise_feature = rng.uniform(0, 1, size=n)  # unrelated to default -> low IV
    return pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "strong_feature": strong_feature,
            "noise_feature": noise_feature,
            "loan_default": default,
        }
    )


def test_iv_is_nonnegative():
    df = _predictive_df()
    _, iv = calculate_woe_iv(df, "strong_feature")
    assert iv >= 0


def test_predictive_feature_has_higher_iv_than_noise():
    df = _predictive_df()
    _, iv_strong = calculate_woe_iv(df, "strong_feature")
    _, iv_noise = calculate_woe_iv(df, "noise_feature")
    assert iv_strong > iv_noise


def test_select_features_drops_low_iv():
    df = _predictive_df()
    iv_df, woe_tables, selected = select_features(df, ["strong_feature", "noise_feature"])
    assert "strong_feature" in selected
    assert set(iv_df["feature"]) == {"strong_feature", "noise_feature"}


def test_transform_to_woe_returns_series_same_length():
    df = _predictive_df()
    woe_table, _ = calculate_woe_iv(df, "strong_feature")
    woe_series = transform_to_woe(df, "strong_feature", woe_table)
    assert len(woe_series) == len(df)


def test_build_woe_dataset_has_expected_columns():
    df = _predictive_df()
    iv_df, woe_tables, selected = select_features(df, ["strong_feature", "noise_feature"])
    woe_df = build_woe_dataset(df, selected, woe_tables)
    assert "loan_default" in woe_df.columns
    for f in selected:
        assert f"{f}_woe" in woe_df.columns
