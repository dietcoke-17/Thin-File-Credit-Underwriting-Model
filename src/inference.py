"""Scores a single new customer with an already-fitted model bundle.

A "bundle" is everything Phase 3-6 produced that's needed to score one raw
customer outside the batch pipeline: the WoE tables for the selected
features (to turn a raw value into a WoE), the fitted logistic regression
coefficients, the PDO scaling constants, and the KS-optimal cutoff score.
`build_bundle` assembles one from `pipeline.run()`'s return value;
`score_customer` is the only thing the live API needs to call.
"""

from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from src.scorecard import compute_scaling_constants, probability_to_score


def build_bundle(pipeline_results: dict) -> Dict[str, Any]:
    result = pipeline_results["model_result"]
    factor, offset = compute_scaling_constants()
    return {
        "selected_features": pipeline_results["selected_features"],
        "woe_tables": {
            f: pipeline_results["woe_tables"][f] for f in pipeline_results["selected_features"]
        },
        "params": result.params.to_dict(),
        "factor": factor,
        "offset": offset,
        "cutoff_score": int(pipeline_results["optimal_cutoff"]["cutoff_score"]),
    }


def save_bundle(bundle: dict, path: str) -> None:
    joblib.dump(bundle, path)


def load_bundle(path: str) -> Dict[str, Any]:
    return joblib.load(path)


def _woe_for_value(value: float, woe_table: pd.DataFrame) -> float:
    """Map a raw feature value to the WoE of the training-time bin it falls
    into, clipping to the nearest edge bin for values outside the training
    range rather than raising.
    """
    intervals = sorted(
        (idx for idx in woe_table.index if isinstance(idx, pd.Interval)),
        key=lambda iv: iv.left,
    )
    if not intervals:
        return float(woe_table["woe"].iloc[0])
    if value <= intervals[0].right:
        return float(woe_table.loc[intervals[0], "woe"])
    for iv in intervals:
        if value <= iv.right:
            return float(woe_table.loc[iv, "woe"])
    return float(woe_table.loc[intervals[-1], "woe"])


def score_customer(raw_features: Dict[str, float], bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Score one customer's raw alternate-data feature values.

    raw_features must contain a value for every feature in
    bundle["selected_features"]. Returns probability of default, the
    300-900 credit score, an approve/decline decision against the
    KS-optimal cutoff, and the per-feature WoE used.
    """
    params = bundle["params"]
    log_odds = params.get("const", 0.0)
    woe_values = {}
    for feature in bundle["selected_features"]:
        woe = _woe_for_value(raw_features[feature], bundle["woe_tables"][feature])
        woe_values[feature] = woe
        log_odds += params[f"{feature}_woe"] * woe

    p_default = 1 / (1 + np.exp(-log_odds))
    score = probability_to_score(p_default, bundle["factor"], bundle["offset"])
    decision = "Approve" if score >= bundle["cutoff_score"] else "Decline"

    return {
        "probability_of_default": round(float(p_default), 4),
        "credit_score": score,
        "cutoff_score": bundle["cutoff_score"],
        "decision": decision,
        "woe_values": {k: round(v, 4) for k, v in woe_values.items()},
    }
