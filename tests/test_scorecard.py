from src.scorecard import compute_scaling_constants, probability_to_score
from src.config import BASE_SCORE, BASE_ODDS


def test_score_at_base_odds_equals_base_score():
    factor, offset = compute_scaling_constants()
    p_default_at_base_odds = 1 / (1 + BASE_ODDS)  # odds_good = BASE_ODDS at this PD
    score = probability_to_score(p_default_at_base_odds, factor, offset)
    assert abs(score - BASE_SCORE) <= 1  # integer rounding tolerance


def test_score_increases_as_pd_decreases():
    factor, offset = compute_scaling_constants()
    low_risk_score = probability_to_score(0.01, factor, offset)
    high_risk_score = probability_to_score(0.5, factor, offset)
    assert low_risk_score > high_risk_score


def test_score_is_clipped_to_bounds():
    factor, offset = compute_scaling_constants()
    assert probability_to_score(0.999999, factor, offset) >= 300
    assert probability_to_score(0.000001, factor, offset) <= 900


def test_doubling_pdo_doubles_odds_at_one_pdo_step():
    # Moving exactly one PDO above the base score should double the odds.
    factor, offset = compute_scaling_constants()
    from src.config import PDO
    import numpy as np

    base_ln_odds = np.log(BASE_ODDS)
    one_pdo_ln_odds = base_ln_odds + np.log(2)
    score_at_double_odds = offset + factor * one_pdo_ln_odds
    assert abs(score_at_double_odds - (BASE_SCORE + PDO)) < 1e-6
