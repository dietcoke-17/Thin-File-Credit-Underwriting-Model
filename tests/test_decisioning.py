import numpy as np
import pandas as pd
import pytest

from src.decisioning import build_cutoff_table, compute_gini, find_optimal_cutoff


def test_gini_matches_auc_formula():
    assert compute_gini(0.5) == pytest.approx(0.0)
    assert compute_gini(1.0) == pytest.approx(1.0)
    assert compute_gini(0.75) == pytest.approx(0.5)


def _toy_scorecard(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    # Higher score should mean lower default probability, by construction.
    scores = rng.integers(300, 900, size=n)
    prob_default = 1 - (scores - 300) / 600
    loan_default = rng.binomial(1, np.clip(prob_default, 0.01, 0.99))
    return pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "Final_Credit_Score": scores,
            "loan_default": loan_default,
        }
    )


def test_cutoff_table_approval_rate_increases_monotonically():
    scorecard_df = _toy_scorecard()
    table = build_cutoff_table(scorecard_df, n_bins=10)
    assert list(table["approval_rate"]) == sorted(table["approval_rate"])


def test_cutoff_table_bad_rate_rises_as_approval_rate_rises():
    # Approving more people (going further down the score list) should
    # never make the approved-book bad rate go down, since scores are
    # constructed to be monotonically related to risk.
    scorecard_df = _toy_scorecard()
    table = build_cutoff_table(scorecard_df, n_bins=10)
    assert table["bad_rate_approved"].iloc[-1] >= table["bad_rate_approved"].iloc[0]


def test_find_optimal_cutoff_returns_max_ks_row():
    scorecard_df = _toy_scorecard()
    table = build_cutoff_table(scorecard_df, n_bins=10)
    optimal = find_optimal_cutoff(table)
    assert optimal["ks"] == table["ks"].max()
