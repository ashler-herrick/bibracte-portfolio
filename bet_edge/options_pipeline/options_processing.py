import logging
from typing import Dict


import polars as pl
import numpy as np
import py_vollib_vectorized as pv

logger = logging.getLogger(__name__)

def process_option_tickers(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Processes a DataFrame containing OPRA-style option tickers.
    Expects a 'ticker' column in the input DataFrame with tickers in the format:
      O:<underlying><exp_date><opt_type><strike>
    where:
      - underlying: variable length (letters and digits) after "O:" and before the last 15 characters
      - exp_date: 6 digits (YYMMDD)
      - opt_type: 'C' or 'P'
      - strike: 8 digits (strike price multiplied by 1000)
    
    Returns a new DataFrame with additional columns:
      - underlying
      - expiration_date (parsed as a Date)
      - option_type (lowercase)
      - strike_price (float)
    """
    # Use a single with_columns to compute all substrings by using str.len_chars() inline.
    lf = lf.with_columns([
        # Underlying: from index 2 to (len - 15) -> equivalent to ticker[2: len-15]
        pl.col("ticker")
          .str.slice(2, pl.col("ticker").str.len_chars() - 17)
          .alias("underlying"),
        # exp_date: 6 characters starting at (len - 15)
        pl.col("ticker")
          .str.slice(pl.col("ticker").str.len_chars() - 15, 6)
          .alias("exp_date"),
        # Option type: 1 character at (len - 9)
        pl.col("ticker")
          .str.slice(pl.col("ticker").str.len_chars() - 9, 1)
          .alias("opt_type"),
        # Strike string: last 8 characters
        pl.col("ticker")
          .str.slice(-8, 8)
          .alias("strike_str")
    ])
    
    # Convert the extracted strings to proper types in one go.
    lf = lf.with_columns([
        (
            "20"
            + pl.col("exp_date").str.slice(0, 2)
            + "-"
            + pl.col("exp_date").str.slice(2, 2)
            + "-"
            + pl.col("exp_date").str.slice(4, 2)
        ).str.strptime(pl.Date, format="%Y-%m-%d").alias("expiration_date"),
        (pl.col("strike_str").cast(pl.Int64) / 1000).alias("strike_price"),
        pl.col("opt_type").str.to_lowercase().alias("opt_type")
    ]).drop(["exp_date", "strike_str"])
    
    return lf


def process_datetime_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Processes datetime columns in the DataFrame.
    Expects:
      - 'window_start' as a nanosecond timestamp (int)
      - 'expiration_date' as a Date (or timestamp) column.
    
    Adds:
      - window_start_dt: as datetime
      - expiration_dt: as datetime
      - DTE: days to expiration
      - t: time to expiration in years (DTE / 365)
    """
    ns_in_day = 86400 * 10**9
    # Combine datetime casts and DTE computation to reduce intermediate copies.
    lf = lf.with_columns([
        pl.col("window_start").cast(pl.Datetime("ns")).alias("window_start_dt"),
        pl.col("expiration_date").cast(pl.Datetime("ns")).alias("expiration_dt"),
        ((pl.col("expiration_date").cast(pl.Datetime("ns")) - pl.col("window_start").cast(pl.Datetime("ns")))
            .cast(pl.Int64) / ns_in_day).alias("DTE")
    ])
    # Compute t from DTE.
    lf = lf.with_columns((pl.col("DTE") / 365.0).alias("t"))
    return lf


def join_stocks(stock_lf: pl.LazyFrame, option_lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Joins stock data with options data based on the underlying ticker and window start time.
    Performs a left join between `options_df` and `stock_df` (using lazy API) and renames stock columns.
    """
    # Use lazy join to defer computation until collect() is called.
    lf = option_lf.lazy().join(
        stock_lf.lazy(),
        how="left",
        left_on=["underlying", "window_start"],
        right_on=["ticker", "window_start"]
    ).rename({
        "volume_right": "stock_volume",
        "open_right": "stock_open",
        "close_right": "stock_close",
        "high_right": "stock_high",
        "low_right": "stock_low",
        "transactions_right": "stock_transactions",
    })
    return lf


def compute_vectorized_option_metrics(df: pl.DataFrame, risk_free_rate: float = 0.01) -> pl.DataFrame:
    """
    Computes option implied volatility and Greeks using py_vollib_vectorized.
    Expects in df:
      - stock_close, strike_price, close (option price), t, and opt_type.
    Returns the DataFrame with new columns:
      - implied_vol, delta, gamma, theta, vega
    """
    # Convert necessary columns to numpy arrays
    S = df["stock_close"].to_numpy()  # Underlying stock price
    K = df["strike_price"].to_numpy()   # Strike price
    option_price = df["close"].to_numpy() # Option market price
    t_array = df["t"].to_numpy()          # Time to expiration in years
    r_array = np.array(risk_free_rate)
    flag = df["opt_type"].to_numpy()

    # Compute implied volatility and Greeks using vectorized functions.
    iv = pv.implied_volatility.vectorized_implied_volatility(
        price=option_price, S=S, K=K, t=t_array, r=r_array, flag=flag, return_as="numpy"
    )
    greeks = pv.api.get_all_greeks(S=S, K=K, t=t_array, r=r_array, sigma=iv, flag=flag, return_as="dict")

    # Add the computed arrays back into the DataFrame.
    df = df.with_columns([
        pl.Series("implied_vol", iv),
        pl.Series("delta", greeks["delta"]),
        pl.Series("gamma", greeks["gamma"]),
        pl.Series("theta", greeks["theta"]),
        pl.Series("vega", greeks["vega"])
    ])
    return df


def compute_vectorized_option_metrics_chunked(df: pl.DataFrame, risk_free_rate: float = 0.01, chunk_size: int = 10_000_000) -> pl.DataFrame:
    chunks = []
    n = df.height
    for i in range(0, n, chunk_size):
        chunk = df[i : i + chunk_size]
        chunk = compute_vectorized_option_metrics(chunk, risk_free_rate)
        chunks.append(chunk)
    return pl.concat(chunks)