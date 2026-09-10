"""Phase 5: Scorecard generation, 300-900 scale.

Converts a model's probability of default into a familiar three-digit
score using the Points-to-Double-the-Odds (PDO) method:

    Score = Offset + Factor * ln(Odds)

where Factor = PDO / ln(2) and Offset = Base_Score - Factor * ln(Base_Odds).
"""

import numpy as np
import pandas as pd

from src.config import BASE_ODDS, BASE_SCORE, PDO, SCORE_CAP, SCORE_FLOOR


def compute_scaling_constants(
    pdo: float = PDO, base_score: float = BASE_SCORE, base_odds: float = BASE_ODDS
):
    """Return (factor, offset) for the PDO scaling formula."""
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return factor, offset


def probability_to_score(
    p_default: float,
    factor: float,
    offset: float,
    floor: int = SCORE_FLOOR,
    cap: int = SCORE_CAP,
) -> int:
    """Convert a single probability of default into a clipped integer score."""
    p_default = np.clip(p_default, 1e-6, 1 - 1e-6)
    odds_good = (1 - p_default) / p_default
    score = offset + factor * np.log(odds_good)
    return int(np.clip(round(score), floor, cap))


def build_scorecard(woe_df: pd.DataFrame, pred_prob: pd.Series) -> pd.DataFrame:
    """Attach Probability_of_Default and Final_Credit_Score columns and
    return the customer-level scorecard output.
    """
    factor, offset = compute_scaling_constants()

    out = woe_df.copy()
    out["Probability_of_Default"] = pred_prob.values
    out["Final_Credit_Score"] = out["Probability_of_Default"].apply(
        lambda p: probability_to_score(p, factor, offset)
    )
    return out[["customer_id", "Probability_of_Default", "Final_Credit_Score"]]
