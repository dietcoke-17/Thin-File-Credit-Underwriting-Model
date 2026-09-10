import pandas as pd

from src.data_generation import generate_raw_data, join_demographics_and_vendor_data, load_dataset


def test_generate_raw_data_shape_and_columns():
    df = generate_raw_data(n=500, seed=1)
    assert len(df) == 500
    expected_cols = {
        "customer_id", "age", "region", "occupation",
        "mobile_recharge_freq_30d", "utility_days_past_due",
        "telecom_data_usage_gb", "agri_yield_stability_index",
        "wallet_transaction_count", "loan_default",
    }
    assert expected_cols.issubset(set(df.columns))


def test_loan_default_is_binary():
    df = generate_raw_data(n=500, seed=1)
    assert set(df["loan_default"].unique()).issubset({0, 1})


def test_reproducible_with_same_seed():
    df1 = generate_raw_data(n=200, seed=7)
    df2 = generate_raw_data(n=200, seed=7)
    pd.testing.assert_frame_equal(df1, df2)


def test_sql_join_preserves_row_count_and_columns():
    raw_df = generate_raw_data(n=300, seed=3)
    joined = join_demographics_and_vendor_data(raw_df)
    assert len(joined) == len(raw_df)
    assert set(joined["customer_id"]) == set(raw_df["customer_id"])


def test_load_dataset_end_to_end():
    df = load_dataset(n=300, seed=3)
    assert len(df) == 300
    assert "loan_default" in df.columns
