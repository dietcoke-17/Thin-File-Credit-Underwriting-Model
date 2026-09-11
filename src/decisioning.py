"""Phase 6: Risk decisioning, Gini coefficient and credit cutoff analysis.

ROC-AUC tells you the model discriminates. It does not tell a lender
where to draw the approve/decline line. This module reports Gini
alongside AUC (the metric credit risk teams actually quote day to day)
and builds a cutoff table across the score range showing the trade-off
every scorecard eventually has to make: approve more people and take on
more bad debt, or hold the line and shrink the approved book.

The KS-optimal cutoff (the score where cumulative Good% and cumulative
Bad% separate the most) is reported as the statistically defensible
starting point. The cutoff a lender actually uses in production also
depends on risk appetite and capital position, which is a business
decision this module deliberately does not make for them.
"""

import numpy as np
import pandas as pd


def compute_gini(auc: float) -> float:
    """Gini coefficient from ROC-AUC: Gini = 2 * AUC - 1."""
    return 2 * auc - 1


def build_cutoff_table(scorecard_df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Approval-rate vs bad-rate trade-off across score deciles, highest
    scores (safest customers) approved first.

    scorecard_df must have Final_Credit_Score and loan_default columns.

    For each cutoff:
        approval_rate:          share of the population that would be approved
        bad_rate_approved:      default rate among approved customers
        cum_good_pct_captured:  share of all Good customers captured at this cutoff
        cum_bad_pct_captured:   share of all Bad customers captured at this cutoff
        ks:                     |cum_good_pct_captured - cum_bad_pct_captured|
    """
    df = scorecard_df.sort_values("Final_Credit_Score", ascending=False).reset_index(drop=True)
    n = len(df)
    total_good = (df["loan_default"] == 0).sum()
    total_bad = (df["loan_default"] == 1).sum()

    bin_edges = np.linspace(0, n, n_bins + 1).astype(int)[1:]
    rows = []
    for edge in bin_edges:
        approved = df.iloc[:edge]
        cum_good_pct = (approved["loan_default"] == 0).sum() / total_good
        cum_bad_pct = (approved["loan_default"] == 1).sum() / total_bad
        rows.append(
            {
                "cutoff_score": int(approved["Final_Credit_Score"].min()),
                "approval_rate": round(edge / n, 4),
                "bad_rate_approved": round((approved["loan_default"] == 1).mean(), 4),
                "cum_good_pct_captured": round(cum_good_pct, 4),
                "cum_bad_pct_captured": round(cum_bad_pct, 4),
                "ks": round(abs(cum_good_pct - cum_bad_pct), 4),
            }
        )
    return pd.DataFrame(rows)


def find_optimal_cutoff(cutoff_table: pd.DataFrame) -> pd.Series:
    """Return the cutoff-table row with the highest KS separation, the
    standard statistically-derived starting point for a decision cutoff.
    """
    return cutoff_table.loc[cutoff_table["ks"].idxmax()]
