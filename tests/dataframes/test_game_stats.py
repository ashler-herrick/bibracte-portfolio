import pytest
import polars as pl
from polars.testing import assert_frame_equal


from bet_edge.dataframes import game_stats as gs
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
    test = gs.genr_agg_dfm(sample_dfm2, ["id"], ["stat1", "stat2"])
    expected = {
        "id": [1],
        "stat1": 210,
        "stat2": 210,
    }
    expected_df = pl.DataFrame(expected, schema={"id": pl.Int64, "stat1": pl.Int64, "stat2": pl.Int64})
    assert_frame_equal(test.dataframe, expected_df)


def test_calc_summary_stats(sample_dfm2):
    result_dfm = gs.calc_summary_stats(sample_dfm2, ["id"], ["stat1"], "test_attr")

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
    result_df = gs.calc_shifted_dfm(sample_dfm2, ["id", "season"], ["stat1"], n=1)
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
    result = gs.calc_cuml_stats(sample_dfm2, ["id", "season"], ["stat1"], "test_attr")
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
    result = gs.calc_offset_summary_stats(sample_dfm2, ["id", "season"], ["stat1"], "test_attr")
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
