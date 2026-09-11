import pandas as pd

from src.statistical_tests import chi_square_test, ks_test, validate_features


def _toy_df():
    # Constructed so the bucket clearly correlates with default and the
    # continuous feature clearly separates Good from Bad.
    return pd.DataFrame(
        {
            "bucket": ["low"] * 40 + ["high"] * 40,
            "continuous": list(range(0, 40)) + list(range(60, 100)),
            "loan_default": [0] * 35 + [1] * 5 + [0] * 5 + [1] * 35,
        }
    )


def test_chi_square_detects_dependence():
    df = _toy_df()
    chi2, p_value, dof = chi_square_test(df, "bucket")
    assert chi2 > 0
    assert p_value < 0.05
    assert dof == 1


def test_ks_test_detects_separation():
    df = _toy_df()
    ks_stat, p_value = ks_test(df, "continuous")
    assert 0 < ks_stat <= 1
    assert p_value < 0.05


def test_ks_test_near_zero_for_identical_distributions():
    df = pd.DataFrame(
        {
            "continuous": list(range(100)),
            "loan_default": [0, 1] * 50,
        }
    )
    ks_stat, _ = ks_test(df, "continuous")
    assert ks_stat < 0.2


def test_validate_features_covers_every_candidate():
    df = _toy_df().rename(columns={"bucket": "cat_feature"})
    df["extra_continuous"] = df["continuous"] + 5
    result = validate_features(df, ["continuous", "extra_continuous"])
    assert set(result["feature"]) == {"continuous", "extra_continuous"}
    assert {"chi2", "chi2_p", "ks_stat", "ks_p", "significant"}.issubset(result.columns)


def test_validate_features_flags_predictive_feature_as_significant():
    df = _toy_df()
    result = validate_features(df, ["continuous"])
    assert bool(result.loc[0, "significant"]) is True
