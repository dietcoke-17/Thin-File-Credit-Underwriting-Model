"""Phase 4: Logistic regression modeling on WoE-transformed features.

statsmodels is used deliberately instead of sklearn's LogisticRegression:
it reports p-values and confidence intervals, which is what a model
validation review or credit committee will ask for.
"""

from typing import Any, List, Tuple

import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score


def fit_logistic_model(
    woe_df: pd.DataFrame, feature_cols: List[str], target: str = "loan_default"
) -> Tuple[Any, pd.Series]:
    """Fit a logistic regression on WoE-transformed features.

    Returns (fitted_result, predicted_probabilities).
    """
    X = sm.add_constant(woe_df[feature_cols])
    y = woe_df[target]

    result = sm.Logit(y, X).fit(disp=0)
    pred_prob = result.predict(X)
    return result, pred_prob


def evaluate_auc(y_true: pd.Series, pred_prob: pd.Series) -> float:
    """ROC-AUC of predicted default probabilities against actual outcomes."""
    return roc_auc_score(y_true, pred_prob)
