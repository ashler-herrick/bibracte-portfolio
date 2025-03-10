import pytest
import polars as pl

from bet_edge.indicators.ohlcv_time_series import calc_rsi


@pytest.fixture
def sample_data():
    return pl.DataFrame({"price": [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]})


def test_calc_rsi(sample_data):
    # Define parameters for the RSI calculation
    column_name = "price"
    window_size = 3

    # Call the function
    result_df = calc_rsi(sample_data, column_name, window_size)

    # Extract the RSI column for validation
    rsi_values = result_df.select("RSI").to_series()

    # Assert the shape of the output (same length as input)
    assert len(rsi_values) == len(sample_data), "Output RSI column length mismatch with input."

    # Validate known RSI values (manually calculate or verify a few values)
    # For simplicity, we check if the RSI values are within a plausible range
    assert all(0 <= rsi <= 100 for rsi in rsi_values.drop_nulls()), "RSI values are out of expected range."

    # Optionally check for nulls in the RSI column for edge cases (e.g., first few rows)
    assert result_df["RSI"].null_count() > 0, "Expected null values in RSI for initial rows."
