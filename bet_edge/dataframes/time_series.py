"""
This module provides utilities for statistical calculations and transformations on
DataFrameManager objects, including aggregation, rolling statistics, and summary statistics.
"""

import logging
from typing import List

import polars as pl

from bet_edge.dataframes.managers import DataFrameManager
from bet_edge.dataframes.column_helpers import sym_diff


logger = logging.getLogger(__name__)


def genr_agg_dfm(dfm: DataFrameManager, group_cols: List[str], sum_cols: List[str]) -> DataFrameManager:
    """
    Generates an aggregated DataFrameManager by grouping on specified columns and summing others.

    Args:
        dfm (DataFrameManager): The DataFrameManager object to aggregate.
        group_cols (List[str]): The columns to group by.
        sum_cols (List[str]): The columns to sum.

    Returns:
        DataFrameManager: A new DataFrameManager containing the aggregated data.
    """
    agg_df = (
        dfm.dataframe.lazy()
        .group_by(group_cols)
        .agg([pl.sum(col).alias(col) for col in sum_cols])
        .sort(group_cols)
        .collect()
    )
    dfm = DataFrameManager(agg_df, primary_key=group_cols, dimensions=group_cols)
    return dfm


def _format_col_name(attr: str, affix: str, col: str, stat: str) -> str:
    """
    Formats column names based on attributes, affix, and statistical measure.
    """
    return f"{attr}_{affix}_{col}_{stat}" if affix else f"{attr}_{col}_{stat}"


def calc_summary_stats(
    dfm: DataFrameManager,
    group_cols: List[str],
    cols: List[str],
    attr: str,
    affix: str = "",
) -> DataFrameManager:
    """
    Calculates summary statistics (mean, std, count) for specified columns.

    Args:
        dfm (DataFrameManager): The DataFrameManager to calculate stats for.
        group_cols (List[str]): Columns to group by.
        cols (List[str]): Columns for which stats are calculated.
        attr (str): Attribute prefix for naming stats.
        affix (str): Optional affix to append to attribute names.

    Returns:
        DataFrameManager: A DataFrameManager containing summary statistics.
    """
    df = dfm.dataframe.lazy().group_by(group_cols).agg(
        [pl.mean(f"{col}").alias(_format_col_name(attr, affix, col, "mean")) for col in cols]
        + [pl.std(f"{col}").alias(_format_col_name(attr, affix, col, "std")) for col in cols]
        + [pl.count(f"{cols[0]}").alias(_format_col_name(attr, affix, "count", "n"))]
    ).collect()
    return DataFrameManager(df, primary_key=group_cols, dimensions=group_cols)


def calc_cuml_stats(
    dfm: DataFrameManager, group_cols: List[str], cols: List[str], attr: str, affix: str = ""
) -> DataFrameManager:
    """
    Calculates cumulative mean, standard deviation and count for specified columns.

    Args:
        dfm (DataFrameManager): The DataFrame to calculate cumulative stats on.
        group_cols (List[str]): Columns to group by.
        cols (List[str]): Columns for which cumulative stats are calculated.
        dropped_cols (List[str]): Columns dropped during the group by operation.
        attr (str): Attribute prefix for naming stats.
        affix (str): Optional affix to append to attribute names.

    Returns:
        DataFrameManager: DataFrame containing cumulative statistics.
    """
    dropped_cols = sym_diff(dfm.primary_key, group_cols)
    explode_cols = (
        dropped_cols
        + [f"cuml_mean_{col}" for col in cols]
        + [f"cuml_sum_sq_{col}" for col in cols]
        + [f"cuml_sum_{col}" for col in cols]
        + ["cuml_n"]
    )
    agg = (
        dfm.dataframe.lazy().group_by(group_cols)
        .agg(
            [pl.col(col) for col in dropped_cols]
            + [(pl.col(col).cum_sum() / pl.col(col).cum_count()).alias(f"cuml_mean_{col}") for col in cols]
            + [(pl.col(col) ** 2).cum_sum().alias(f"cuml_sum_sq_{col}") for col in cols]
            + [(pl.col(col)).cum_sum().alias(f"cuml_sum_{col}") for col in cols]
            + [pl.col(cols[0]).cum_count().alias("cuml_n")]
        )
        .explode(explode_cols)
    ).sort(group_cols).collect()

    stats = agg.lazy().select(
        dfm.primary_key
        + [pl.col(f"cuml_mean_{col}").alias(_format_col_name(attr, affix, col, "cuml_mean")) for col in cols]
        + [
            (
                (pl.col(f"cuml_sum_sq_{col}") - (pl.col(f"cuml_sum_{col}") ** 2 / pl.col("cuml_n")))
                / (pl.col("cuml_n") - 1)
            )
            .sqrt()
            .alias(_format_col_name(attr, affix, col, "cuml_std"))
            .fill_nan(None)
            for col in cols
        ]
        + [pl.col("cuml_n").alias(_format_col_name(attr, affix, "count", "cuml_n"))]
    ).collect()

    return DataFrameManager(stats, primary_key=dfm.primary_key, dimensions=dfm.primary_key)


def calc_shifted_dfm(dfm: DataFrameManager, group_cols: List[str], cols: List[str], n: int = 1) -> DataFrameManager:
    """
    Calculates shifted values for specified columns.

    Args:
        dfm (DataFrameManager): The DataFrameManager to shift values in.
        group_cols (List[str]): Columns to group by when shifting.
        cols (List[str]): Columns to shift.
        n (int): Number of steps to shift.

    Returns:
        DataFrameManager : DataFrameManager containing shifted columns.
    """
    shifted = dfm.dataframe.with_columns(
        [pl.col(col).shift(i).over(group_cols).alias(f"shifted_{col}_{n}") for col in cols for i in range(1, n + 1)]
    )
    return DataFrameManager(shifted, primary_key=dfm.primary_key, dimensions=dfm.dimensions)


def calc_offset_summary_stats(
    dfm: DataFrameManager,
    group_cols: List[str],
    cols: List[str],
    attr: str,
    affix: str = "",
) -> DataFrameManager:
    """
    Calculates offset summary statistics for the specified columns.

    Args:
        dfm (DataFrameManager): The DataFrameManager containing the data.
        group_cols (List[str]): Columns to group by.
        cols (List[str]): Columns for which offset summary stats are calculated.
        attr (str): Attribute prefix for naming stats.
        affix (str): Optional affix to append to attribute names.

    Returns:
        DataFrameManager: A DataFrameManager containing offset summary statistics.
    """
    shifted = calc_shifted_dfm(dfm, group_cols, cols)
    stats = calc_cuml_stats(
        shifted,
        group_cols,
        dfm.get_col_diff(shifted),
        attr,
        affix,
    )

    return stats


def add_stats(base: DataFrameManager, stats: DataFrameManager, prev_ssn: bool = False) -> DataFrameManager:
    """
    Adds a table of stats by joining it with the base DataFrameManager on the foreign key.

    Args:
        base (DataFrameManager): The base DataFrameManager.
        stats (DataFrameManager): The stats DataFrameManager to join.
        prev_ssn (bool): Whether to adjust the season column for a prior season join.

    Returns:
        DataFrameManager: A DataFrameManager with stats added.
    """
    if prev_ssn:
        stats = DataFrameManager(
            stats.dataframe.with_columns((pl.col("season") + 1).alias("season")), 
            primary_key=stats.primary_key,
            dimensions=stats.dimensions,
        )
    foreign_key = base.get_foreign_key(stats)
    stats_cols = base.get_col_diff(stats)
    new_df = base.dataframe.join(stats.dataframe, on=foreign_key, how="left").select(
        base.dataframe.columns + stats_cols
    )
    new = DataFrameManager(new_df, primary_key=base.primary_key, dimensions=base.dimensions)
    new.assert_valid_pk()
    return new


def fill_null_curr_season_stats(dfm: DataFrameManager, cols: List[str], attr: str) -> DataFrameManager:
    """
    Fills null values in current season stats using prior season stats.

    Args:
        dfm (DataFrameManager): The DataFrameManager to update.
        cols (List[str]): Columns for which stats are calculated.
        attr (str): Attribute prefix for naming stats.

    Returns:
        DataFrameManager: A DataFrameManager with null values filled.
    """
    base_cols = [_format_col_name(attr, "curr_ssn", col, func) for col in cols for func in ["mean", "std"]]
    fill_cols = [_format_col_name(attr, "prev_ssn", col, func) for col in cols for func in ["mean", "std"]]
    dfm.validate_columns_exist(base_cols + fill_cols)

    df = dfm.dataframe.with_columns(
        [
            pl.coalesce([pl.col(base_col), pl.col(fill_col)]).alias(base_col)
            for base_col, fill_col in zip(base_cols, fill_cols)
        ]
    )

    coal = DataFrameManager(df, primary_key=dfm.primary_key, dimensions=dfm.dimensions)
    return coal


def add_rolling_stats(
    dfm: DataFrameManager,
    over_cols: List[str],
    cols: List[str],
    attr: str,
    window: int,
) -> DataFrameManager:
    """
    Adds rolling mean and standard deviation over a specified window for selected columns.

    Args:
        dfm (DataFrameManager): The DataFrameManager containing the data.
        over_cols (List[str]): Columns to group by for rolling statistics.
        cols (List[str]): Columns for which rolling stats are calculated.
        attr (str): Attribute prefix for naming stats.
        window (int): Rolling window size.

    Returns:
        DataFrameManager: A DataFrameManager with rolling statistics added.
    """
    df = dfm.dataframe.sort(dfm.primary_key)
    for col in cols:
        df = df.lazy().with_columns(
            [
                pl.col(col)
                .shift(1)
                .rolling_mean(window)
                .over(over_cols)
                .alias(_format_col_name(attr, f"rolling_{window}", col, "mean")),
                pl.col(col)
                .shift(1)
                .rolling_std(window)
                .over(over_cols)
                .alias(_format_col_name(attr, f"rolling_{window}", col, "std")),
            ]
        ).collect()
    return DataFrameManager(df, primary_key=dfm.primary_key, dimensions=dfm.dimensions)


def calc_grouped_rolling_stats(
    dfm: DataFrameManager,
    over_cols: List[str],
    group_cols: List[str],
    cols: List[str],
    attr: str,
    window: int,
) -> DataFrameManager:
    """
    Calculates grouped rolling statistics (mean and std) over a specified window for selected columns.

    Args:
        dfm (DataFrameManager): The DataFrameManager containing the data.
        over_cols (List[str]): Columns to group by for rolling statistics.
        group_cols (List[str]): Columns to group by for final aggregation.
        cols (List[str]): Columns for which grouped rolling stats are calculated.
        attr (str): Attribute prefix for naming stats.
        window (int): Rolling window size.

    Returns:
        DataFrameManager: A DataFrameManager with grouped rolling statistics.
    """
    attr_means = [_format_col_name(attr, f"rolling_{window}", col, "mean") for col in cols]
    attr_stds = [_format_col_name(attr, f"rolling_{window}", col, "std") for col in cols]
    try:
        dfm.validate_columns_exist([*attr_means, *attr_stds])
    except KeyError:
        dfm = add_rolling_stats(dfm, over_cols, cols, attr, window)

    grouped = dfm.dataframe.lazy().group_by(group_cols).agg(
        [pl.mean(mean_col).alias("group_" + mean_col) for mean_col in attr_means]
        + [
            (pl.col(std_col) ** 2 / pl.col(std_col).count()).mean().sqrt().alias("group_" + std_col)
            for std_col in attr_stds
        ]
    ).collect()
    grouped_dfm = DataFrameManager(grouped, primary_key=group_cols, dimensions=group_cols)
    return grouped_dfm
