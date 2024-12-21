import pytest
import polars as pl
from polars.testing import assert_frame_equal


from bet_edge.dataframes import time_series as ts
from bet_edge.dataframes.dataframe_manager import DataFrameManager


@pytest.fixture
def sample_dfm1():
    df = pl.DataFrame(
        {
            "id": [1, 1, 1, 1, 2, 2, 2, 2],
            "season": [2022, 2022, 2021, 2021, 2022, 2022, 2021, 2021],
            "week": [1, 2, 1, 2, 1, 2, 1, 2],
            "stat": [10, 20, 30, 40, 50, 60, 70, 80],
        }
    )
    return DataFrameManager(df, ["id", "season", "week"])


@pytest.fixture
def sample_dfm2():
    df = pl.DataFrame(
        {
            "id": [1, 1, 1, 1, 1, 1],
            "season": [2022, 2022, 2022, 2021, 2021, 2021],
            "week": [1, 2, 3, 1, 2, 3],
            "stat1": [10, 20, 30, 40, 50, 60],
            "stat2": [10, 20, 30, 40, 50, 60],
        }
    )
    return DataFrameManager(df, ["id", "season", "week"])


def test_genr_agg_dfm(sample_dfm2):
    test = ts.genr_agg_dfm(sample_dfm2, ["id"], ["stat1", "stat2"])
    expected = {
        "id": [1],
        "stat1": 210,
        "stat2": 210,
    }
    expected_df = pl.DataFrame(expected, schema={"id": pl.Int64, "stat1": pl.Int64, "stat2": pl.Int64})
    assert_frame_equal(test.dataframe, expected_df)


def test_calc_summary_stats(sample_dfm2):
    result_dfm = ts.calc_summary_stats(sample_dfm2, ["id"], ["stat1"], "test_attr")

    expected = {
        "id": [1],
        "test_attr_stat1_mean": 35,
        "test_attr_stat1_std": 18.708286933869708,
        "test_attr_count_n": 6,
    }
    expected_df = pl.DataFrame(
        expected,
        schema={
            "id": pl.Int64,
            "test_attr_stat1_mean": pl.Float64,
            "test_attr_stat1_std": pl.Float64,
            "test_attr_count_n": pl.UInt32,
        },
    )
    assert_frame_equal(result_dfm.dataframe, expected_df)


def test_calc_shifted_dfm(sample_dfm2):
    result_df = ts.calc_shifted_dfm(sample_dfm2, ["id", "season"], ["stat1"], n=1)
    expected_df = pl.DataFrame(
        {
            "id": [1, 1, 1, 1, 1, 1],
            "season": [2022, 2022, 2022, 2021, 2021, 2021],
            "week": [1, 2, 3, 1, 2, 3],
            "stat1": [10, 20, 30, 40, 50, 60],
            "stat2": [10, 20, 30, 40, 50, 60],
            "shifted_stat1_1": [None, 10, 20, None, 40, 50],
        }
    )
    assert_frame_equal(result_df.dataframe, expected_df)


def test_calc_cuml_stats(sample_dfm2):
    result = ts.calc_cuml_stats(sample_dfm2, ["id", "season"], ["stat1"], "test_attr")
    expected = {
        "id": [1, 1, 1, 1, 1, 1],
        "season": [2021, 2021, 2021, 2022, 2022, 2022],
        "week": [1, 2, 3, 1, 2, 3],
        "test_attr_stat1_cuml_mean": [40.0, 45.0, 50.0, 10.0, 15.0, 20.0],
        "test_attr_stat1_cuml_std": [None, 7.0710678118654755, 10.0, None, 7.0710678118654755, 10.0],
        "test_attr_count_cuml_n": [1, 2, 3, 1, 2, 3],
    }
    expected_df = pl.DataFrame(
        expected,
        schema={
            "id": pl.Int64,
            "season": pl.Int64,
            "week": pl.Int64,
            "test_attr_stat1_cuml_mean": pl.Float64,
            "test_attr_stat1_cuml_std": pl.Float64,
            "test_attr_count_cuml_n": pl.UInt32,
        },
    )
    assert_frame_equal(result.dataframe, expected_df)


def test_calc_offset_summary_stats(sample_dfm2):
    result = ts.calc_offset_summary_stats(sample_dfm2, ["id", "season"], ["stat1"], "test_attr")
    expected = {
        "id": [1, 1, 1, 1, 1, 1],
        "season": [2021, 2021, 2021, 2022, 2022, 2022],
        "week": [1, 2, 3, 1, 2, 3],
        "test_attr_shifted_stat1_1_cuml_mean": [None, 40.0, 45.0, None, 10.0, 15.0],
        "test_attr_shifted_stat1_1_cuml_std": [None, None, 7.0710678118654755, None, None, 7.0710678118654755],
        "test_attr_count_cuml_n": [0, 1, 2, 0, 1, 2],
    }
    expected_df = pl.DataFrame(
        expected,
        schema={
            "id": pl.Int64,
            "season": pl.Int64,
            "week": pl.Int64,
            "test_attr_shifted_stat1_1_cuml_mean": pl.Float64,
            "test_attr_shifted_stat1_1_cuml_std": pl.Float64,
            "test_attr_count_cuml_n": pl.UInt32,
        },
    )
    assert_frame_equal(result.dataframe, expected_df)

def test_add_stats():
    # Mock base DataFrameManager
    base_df = pl.DataFrame({
        "id": [1, 2, 3],
        "season": [2021, 2021, 2021],
        "team": ["A", "B", "C"]
    })
    base_dm = DataFrameManager(base_df, primary_key=["id"], dimensions=["id","season","team"])

    # Mock stats DataFrameManager
    stats_df = pl.DataFrame({
        "id": [1, 2, 4, 1],
        "season": [2021, 2021, 2021, 2020],
        "team": ["A", "B", "C", "A"],
        "stat": [10, 20, 30, 40]
    })
    stats_dm = DataFrameManager(stats_df, primary_key=["id", "season"], dimensions=["id","season","team"])

    # Test without prev_ssn
    result_dm = ts.add_stats(base_dm, stats_dm)
    expected_df = pl.DataFrame({
        "id": [1, 2, 3],
        "season": [2021, 2021, 2021],
        "team": ["A", "B", "C"],
        "stat": [10, 20, None]
    })
    assert_frame_equal(result_dm.dataframe,expected_df)

    # Test with prev_ssn
    result_dm_prev_ssn = ts.add_stats(base_dm, stats_dm, prev_ssn=True)
    expected_df_prev_ssn = pl.DataFrame({
        "id": [1, 2, 3],
        "season": [2021, 2021, 2021],
        "team": ["A", "B", "C"],
        "stat": [40, None, None]  # Stats shouldn't match due to season adjustment
    })
    assert_frame_equal(result_dm_prev_ssn.dataframe,expected_df_prev_ssn)




