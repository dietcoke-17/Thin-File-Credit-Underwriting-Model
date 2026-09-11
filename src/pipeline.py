"""End-to-end pipeline: wires together all six phases and produces the
final scored customer table plus the diagnostic plots used in the README.
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")  # headless-safe backend for CI / Docker
import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, RESULTS_DIR
from src.data_generation import load_dataset
from src.decisioning import build_cutoff_table, compute_gini, find_optimal_cutoff
from src.modeling import evaluate_auc, fit_logistic_model
from src.scorecard import build_scorecard
from src.statistical_tests import chi_square_test, ks_test, validate_features
from src.woe_iv import build_woe_dataset, select_features

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CANDIDATE_FEATURES = [
    "age",
    "mobile_recharge_freq_30d",
    "utility_days_past_due",
    "telecom_data_usage_gb",
    "agri_yield_stability_index",
    "wallet_transaction_count",
]


def run(save_outputs: bool = True) -> dict:
    """Run all five phases in order and return the key artifacts.

    Returns a dict with: df, iv_df, selected_features, woe_df, model_result,
    auc, scorecard_df.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ---- Phase 1: data ----
    df = load_dataset()
    logger.info("Phase 1 | rows=%d default_rate=%.2f%%", len(df), 100 * df["loan_default"].mean())

    # ---- Phase 2: statistical validation ----
    df["utility_dpd_bin"] = pd.cut(
        df["utility_days_past_due"], bins=[-1, 0, 5, 15, 1000], labels=["0", "1-5", "6-15", "15+"]
    )
    chi2, chi_p, dof = chi_square_test(df, "utility_dpd_bin")
    ks_stat, ks_p = ks_test(df, "agri_yield_stability_index")
    logger.info("Phase 2 | example chi2=%.2f (p=%.2e, dof=%d)", chi2, chi_p, dof)
    logger.info("Phase 2 | example KS=%.4f (p=%.2e)", ks_stat, ks_p)

    feature_validation_df = validate_features(df, CANDIDATE_FEATURES)
    n_significant = int(feature_validation_df["significant"].sum())
    logger.info(
        "Phase 2 | %d/%d candidate features significant on both Chi2 and KS (p<0.05)",
        n_significant,
        len(CANDIDATE_FEATURES),
    )

    # ---- Phase 3: WoE / IV ----
    iv_df, woe_tables, selected_features = select_features(df, CANDIDATE_FEATURES)
    logger.info("Phase 3 | selected features (IV >= threshold): %s", selected_features)
    woe_df = build_woe_dataset(df, selected_features, woe_tables)

    # ---- Phase 4: logistic regression ----
    feature_cols = [f"{f}_woe" for f in selected_features]
    result, pred_prob = fit_logistic_model(woe_df, feature_cols)
    auc = evaluate_auc(woe_df["loan_default"], pred_prob)
    logger.info("Phase 4 | ROC-AUC=%.4f", auc)

    # ---- Phase 5: scorecard ----
    scorecard_df = build_scorecard(woe_df, pred_prob)
    scorecard_df["loan_default"] = woe_df["loan_default"].values
    logger.info(
        "Phase 5 | score range %d-%d",
        scorecard_df["Final_Credit_Score"].min(),
        scorecard_df["Final_Credit_Score"].max(),
    )

    # ---- Phase 6: risk decisioning (Gini + cutoff analysis) ----
    gini = compute_gini(auc)
    cutoff_table = build_cutoff_table(scorecard_df)
    optimal_cutoff = find_optimal_cutoff(cutoff_table)
    logger.info("Phase 6 | Gini=%.4f", gini)
    logger.info(
        "Phase 6 | KS-optimal cutoff: score>=%d | approval_rate=%.1f%% | bad_rate_approved=%.2f%%",
        optimal_cutoff["cutoff_score"],
        100 * optimal_cutoff["approval_rate"],
        100 * optimal_cutoff["bad_rate_approved"],
    )

    if save_outputs:
        scorecard_df.to_csv(os.path.join(RESULTS_DIR, "scored_customers.csv"), index=False)
        iv_df.to_csv(os.path.join(RESULTS_DIR, "iv_summary.csv"), index=False)
        feature_validation_df.to_csv(
            os.path.join(RESULTS_DIR, "feature_validation.csv"), index=False
        )
        cutoff_table.to_csv(os.path.join(RESULTS_DIR, "cutoff_analysis.csv"), index=False)
        with open(os.path.join(RESULTS_DIR, "model_summary.txt"), "w") as f:
            f.write(str(result.summary()))
            f.write(f"\n\nROC-AUC: {auc:.4f}\n")
            f.write(f"Gini coefficient: {gini:.4f}\n")
            f.write(
                f"KS-optimal cutoff: score >= {optimal_cutoff['cutoff_score']} "
                f"(approval rate {optimal_cutoff['approval_rate']:.1%}, "
                f"bad rate among approved {optimal_cutoff['bad_rate_approved']:.2%}, "
                f"KS {optimal_cutoff['ks']:.4f})\n"
            )
        _save_figures(df, scorecard_df, pred_prob, woe_df["loan_default"], cutoff_table, optimal_cutoff)

    return {
        "df": df,
        "feature_validation_df": feature_validation_df,
        "iv_df": iv_df,
        "selected_features": selected_features,
        "woe_tables": woe_tables,
        "woe_df": woe_df,
        "model_result": result,
        "auc": auc,
        "gini": gini,
        "scorecard_df": scorecard_df,
        "cutoff_table": cutoff_table,
        "optimal_cutoff": optimal_cutoff,
    }


def _save_figures(df, scorecard_df, pred_prob, y_true, cutoff_table, optimal_cutoff):
    from sklearn.metrics import roc_curve

    # Default rate by utility delinquency bucket
    fig, ax = plt.subplots(figsize=(6, 4))
    df.groupby("utility_dpd_bin", observed=True)["loan_default"].mean().plot(
        kind="bar", ax=ax, color="#4C72B0"
    )
    ax.set_title("Default rate by utility delinquency bucket")
    ax.set_ylabel("Default rate")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "default_rate_by_dpd_bucket.png"), dpi=120)
    plt.close(fig)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, pred_prob)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(fpr, tpr, color="#4C72B0", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve: Probability of Default Model")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "roc_curve.png"), dpi=120)
    plt.close(fig)

    # Score distribution by outcome
    fig, ax = plt.subplots(figsize=(6, 4.5))
    good = scorecard_df.loc[scorecard_df["loan_default"] == 0, "Final_Credit_Score"]
    bad = scorecard_df.loc[scorecard_df["loan_default"] == 1, "Final_Credit_Score"]
    ax.hist(good, bins=30, alpha=0.6, label="Good", density=True, color="#55A868")
    ax.hist(bad, bins=30, alpha=0.6, label="Default", density=True, color="#C44E52")
    ax.set_xlabel("Final Credit Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution by Actual Outcome")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "score_distribution.png"), dpi=120)
    plt.close(fig)

    # Gains / cutoff chart: cumulative Good% and Bad% captured across the
    # score range, with the KS-optimal cutoff marked. This is the chart a
    # credit risk team actually uses to pick an approve/decline line.
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = cutoff_table["approval_rate"] * 100
    ax.plot(x, cutoff_table["cum_good_pct_captured"] * 100, color="#55A868", label="Cumulative Good % captured")
    ax.plot(x, cutoff_table["cum_bad_pct_captured"] * 100, color="#C44E52", label="Cumulative Bad % captured")
    ax.axvline(optimal_cutoff["approval_rate"] * 100, color="gray", linestyle="--",
               label=f"KS-optimal cutoff (score \u2265 {int(optimal_cutoff['cutoff_score'])})")
    ax.set_xlabel("Approval rate (% of population, best scores approved first)")
    ax.set_ylabel("Cumulative % captured")
    ax.set_title("Cutoff Analysis: Approval Rate vs Risk Capture")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "cutoff_analysis.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    run()
