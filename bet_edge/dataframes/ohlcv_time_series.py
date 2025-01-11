from typing import Union, List

import polars as pl


def calc_rsi(df: pl.DataFrame, col: str, window: int) -> pl.DataFrame:
    tdf = (
        df.lazy().with_columns(
            [
                pl.col(col).diff().alias("delta"),
                pl.when(pl.col(col).diff() > 0)
                .then(pl.col(col).diff())
                .otherwise(0.0)
                .alias("gain"),
                pl.when(pl.col(col).diff() < 0)
                .then(-pl.col(col).diff())
                .otherwise(0.0)
                .alias("loss"),
            ]
        )
        .with_columns(
            [
                pl.col("gain").rolling_mean(window_size=window).alias("avg_gain"),
                pl.col("loss").rolling_mean(window_size=window).alias("avg_loss"),
            ]
        )
        .with_columns(
            [
                (pl.col("avg_gain") / pl.col("avg_loss")).alias("RS"),
                (100 - (100 / (1 + (pl.col("avg_gain") / pl.col("avg_loss"))))).alias("RSI"),
            ]
        )
    ).collect()

    return tdf


def calc_macd(df: pl.DataFrame, col: str, fast_period: int, slow_period: int, signal_period: int) -> pl.DataFrame:
    tdf = df.lazy().with_columns([
        pl.col(col).ewm_mean(alpha=2/(fast_period + 1)).alias("ema_fast"),
        pl.col(col).ewm_mean(alpha=2/(slow_period + 1)).alias("ema_slow")
    ]).with_columns(
        (pl.col("ema_fast") - pl.col("ema_slow")).alias('MACD')
    ).with_columns(
        pl.col('MACD').ewm_mean(alpha=2 / (signal_period + 1)).alias("macd_signal")
    ).collect()
    return tdf


def calc_crossover(df: pl.DataFrame, col: str, compare: Union[float, str], signal_name: str) -> pl.DataFrame:
    if isinstance(compare, str):
        cmpr = pl.col(compare)
    elif isinstance(compare, float):
        cmpr = compare
    else:
        raise ValueError("Variable 'compare' must of either type str or float.")
    return (
    df.lazy()
    .with_columns((pl.col(col) > cmpr).alias("above"))  # Check if short > long
    .with_columns(pl.col("above").shift(1).alias("above_prev"))  # Previous state
    .with_columns(
        pl.when(pl.col("above") & ~pl.col("above_prev")).then(1)
        .when(~pl.col("above") & pl.col("above_prev")).then(-1)
        .otherwise(None)
        .alias(signal_name)  # Add signal
    )
    .drop(["above", "above_prev"])  # Clean up intermediate columns
    ).collect()

