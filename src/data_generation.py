"""Phase 1: Synthetic alternate-data generation and SQL integration.

Simulates a thin-file lending portfolio: customers with no bureau history,
scored instead on telecom, utility, and agricultural proxy signals. A SQL
join is included to mirror how demographic data and vendor alternate-data
would actually be joined in production, they typically live in separate
systems.
"""

import sqlite3

import numpy as np
import pandas as pd

from src.config import N_CUSTOMERS, RANDOM_SEED


def generate_raw_data(n: int = N_CUSTOMERS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate a synthetic thin-file customer dataset.

    Returns a DataFrame with demographic fields, five alternate-data
    features, and a binary loan_default target.
    """
    rng = np.random.default_rng(seed)

    customer_id = np.arange(1, n + 1)
    age = rng.integers(21, 65, size=n)
    region = rng.choice(["North", "South", "East", "West"], size=n)
    occupation = rng.choice(
        ["Farmer", "Laborer", "SmallBusiness", "Salaried"],
        size=n,
        p=[0.35, 0.25, 0.2, 0.2],
    )

    mobile_recharge_freq_30d = rng.poisson(4, size=n)
    utility_days_past_due = rng.exponential(scale=6, size=n).astype(int)
    telecom_data_usage_gb = np.round(rng.gamma(shape=2, scale=2, size=n), 2)
    agri_yield_stability_index = np.round(rng.beta(5, 2, size=n), 3)
    wallet_transaction_count = rng.poisson(15, size=n)

    # Weighted risk score -> probability of default -> binary outcome.
    # Weights favor utility delinquency and agri stability, which is
    # roughly what tends to hold in real alternate-data portfolios.
    risk_score = (
        -0.18 * mobile_recharge_freq_30d
        + 0.22 * utility_days_past_due
        - 3.20 * agri_yield_stability_index
        - 0.06 * wallet_transaction_count
        - 0.015 * telecom_data_usage_gb
        + rng.normal(0, 1.8, size=n)
    )
    z = (risk_score - risk_score.mean()) / risk_score.std()
    prob_default = 1 / (1 + np.exp(-(1.15 * z - 1.85)))
    loan_default = rng.binomial(1, prob_default)

    return pd.DataFrame(
        {
            "customer_id": customer_id,
            "age": age,
            "region": region,
            "occupation": occupation,
            "mobile_recharge_freq_30d": mobile_recharge_freq_30d,
            "utility_days_past_due": utility_days_past_due,
            "telecom_data_usage_gb": telecom_data_usage_gb,
            "agri_yield_stability_index": agri_yield_stability_index,
            "wallet_transaction_count": wallet_transaction_count,
            "loan_default": loan_default,
        }
    )


def join_demographics_and_vendor_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Split the raw dataset into a demographics table and a vendor
    alternate-data table, then re-join them with SQL, the way this would
    actually happen when the two sources live in separate systems.
    """
    conn = sqlite3.connect(":memory:")
    try:
        raw_df[["customer_id", "age", "region", "occupation"]].to_sql(
            "demographics", conn, index=False
        )
        raw_df[
            [
                "customer_id",
                "mobile_recharge_freq_30d",
                "utility_days_past_due",
                "telecom_data_usage_gb",
                "agri_yield_stability_index",
                "wallet_transaction_count",
                "loan_default",
            ]
        ].to_sql("vendor_data", conn, index=False)

        query = """
            SELECT d.customer_id, d.age, d.region, d.occupation,
                   v.mobile_recharge_freq_30d, v.utility_days_past_due,
                   v.telecom_data_usage_gb, v.agri_yield_stability_index,
                   v.wallet_transaction_count, v.loan_default
            FROM demographics d
            JOIN vendor_data v ON d.customer_id = v.customer_id
        """
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


def load_dataset(n: int = N_CUSTOMERS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Convenience entry point: generate and join in one call."""
    raw_df = generate_raw_data(n=n, seed=seed)
    return join_demographics_and_vendor_data(raw_df)
