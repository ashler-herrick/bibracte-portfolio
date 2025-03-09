import polars as pl
import numpy as np
import py_vollib_vectorized as pv  # Ensure you have installed py_vollib_vectorized

def process_option_tickers(df: pl.DataFrame) -> pl.DataFrame:
    """
    Processes a DataFrame containing OPRA-style option tickers.
    
    Expects a 'ticker' column in the input DataFrame with tickers in the format:
      O:<underlying><exp_date><opt_type><strike>
    where:
      - underlying: variable length (letters and digits) after "O:" and before the last 15 characters
      - exp_date: 6 digits (YYMMDD)
      - opt_type: 'C' or 'P'
      - strike: 8 digits (strike price multiplied by 1000)
    
    Returns a new DataFrame with the following additional columns:
      - underlying: the underlying asset extracted from the ticker
      - expiration_date: the expiration date as a Date type (assumed to be 20YY-MM-DD)
      - option_type: 'Call' or 'Put'
      - strike_price: the strike price as a float (strike / 1000)
    """
    # Calculate the length of each ticker
    df = df.with_columns(pl.col("ticker").str.len_chars().alias("ticker_length"))
    
    # The last 15 characters always represent exp_date (6), option type (1) and strike (8).
    # Underlying is what remains after the "O:" prefix (2 characters) and before the last 15 characters.
    df = df.with_columns(
        pl.col("ticker").str.slice(2, pl.col("ticker_length") - 2 - 15).alias("underlying"),
        pl.col("ticker").str.slice(pl.col("ticker_length") - 15, 6).alias("exp_date"),
        pl.col("ticker").str.slice(pl.col("ticker_length") - 9, 1).alias("opt_type"),
        pl.col("ticker").str.slice(-8, 8).alias("strike_str")
    )
    
    # Remove the temporary ticker_length column
    df = df.drop("ticker_length")
    
    # Convert the expiration date from YYMMDD to a proper date.
    # Assumes years are in 2000's.
    df = df.with_columns(
        (
            "20" + pl.col("exp_date").str.slice(0, 2) + "-" +
            pl.col("exp_date").str.slice(2, 2) + "-" +
            pl.col("exp_date").str.slice(4, 2)
        ).str.strptime(pl.Date, format="%Y-%m-%d").alias("expiration_date"),
        (pl.col("strike_str").cast(pl.Int64) / 1000).alias("strike_price"),
        pl.col("opt_type").str.to_lowercase().alias("opt_type")
    )

    
    # Optionally drop intermediate columns
    df = df.drop(["exp_date", "strike_str"])
    
    return df

def process_datetime_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Processes datetime columns in the DataFrame.
    Expects:
      - 'window_start' as a nanosecond timestamp (int)
      - 'expiration_date' as a Date (or timestamp) column.
      
    Adds the following columns:
      - window_start_dt: window_start as a datetime
      - expiration_dt: expiration_date as a datetime
      - DTE: days to expiration (difference in days)
      - t: time to expiration in years (DTE / 365)
    """
    df = df.with_columns([
        pl.col("window_start").cast(pl.Datetime("ns")).alias("window_start_dt"),
        pl.col("expiration_date").cast(pl.Datetime("ns")).alias("expiration_dt")
    ])
    
    ns_in_day = 86400 * 10**9
    df = df.with_columns([
        ((pl.col("expiration_dt") - pl.col("window_start_dt"))
            .cast(pl.Int64) / ns_in_day).alias("DTE"),
        (((pl.col("expiration_dt") - pl.col("window_start_dt"))
            .cast(pl.Int64) / ns_in_day) / 365.0).alias("t")
    ])

    return df

def compute_vectorized_option_metrics(df: pl.DataFrame, risk_free_rate: float = 0.01) -> pl.DataFrame:
    """
    Computes option implied volatility and Greeks using the py_vollib_vectorized library.
    
    Expects the following columns in df:
      - stock_close: underlying stock price
      - strike_price: strike price of the option
      - close: option market price (using the 'close' price here)
      - t: time to expiration in years (computed earlier)
      - option_type: either "c" or "p"
    
    Returns the DataFrame with the following new columns:
      - implied_vol
      - delta
      - gamma
      - theta
      - vega
    """
    # Extract necessary arrays from the DataFrame
    S = df["stock_close"].to_numpy()         # Underlying stock price
    K = df["strike_price"].to_numpy()          # Strike price
    option_price = df["close"].to_numpy()      # Option market price
    t_array = df["t"].to_numpy()               # Time to expiration in years
    
    # Create an array for risk-free rate (same value for all rows)
    r_array = np.full_like(S, risk_free_rate, dtype=np.float64)
    flag = df["opt_type"].to_numpy()
    
    # Compute implied volatility using vectorized function
    iv = pv.implied_volatility.vectorized_implied_volatility(price=option_price,S=S,K=K,t=t_array,r=r_array,flag=flag, return_as="numpy")
    
    # Compute Greeks using the computed implied volatility
    delta = pv.greeks.delta(S=S, K=K, t=t_array, r=r_array, sigma=iv, flag=flag, return_as="numpy")
    gamma = pv.greeks.gamma(S=S, K=K, t=t_array, r=r_array, sigma=iv, flag=flag, return_as="numpy")
    theta = pv.greeks.theta(S=S, K=K, t=t_array, r=r_array, sigma=iv, flag=flag, return_as="numpy")
    vega  = pv.greeks.vega(S=S, K=K, t=t_array, r=r_array, sigma=iv, flag=flag, return_as="numpy")
    
    # Add the computed values back into the DataFrame
    df = df.with_columns([
        pl.Series("implied_vol", iv),
        pl.Series("delta", delta),
        pl.Series("gamma", gamma),
        pl.Series("theta", theta),
        pl.Series("vega", vega)
    ])
    return df

def process_options_data(df: pl.DataFrame, risk_free_rate: float = 0.01) -> pl.DataFrame:
    """
    Processes the options data by performing:
      1. Ticker processing to extract underlying, expiration_date, option_type, and strike_price.
      2. Datetime conversion for window_start and expiration_date (calculating DTE and t).
      3. Computation of implied volatility and Greeks using py_vollib_vectorized.
    
    Returns the fully processed DataFrame.
    """
    df = process_option_tickers(df)
    df = process_datetime_columns(df)
    df = compute_vectorized_option_metrics(df, risk_free_rate)
    return df

# Example usage
if __name__ == "__main__":
    # Example DataFrame; replace with your actual data source.
    df = pl.DataFrame({
        "ticker": ["O:AAPL230617C00150000", "O:AAPL230617P00150000"],
        "volume": [100, 200],
        "open": [10.0, 11.0],
        "close": [10.5, 11.2],
        "high": [10.8, 11.5],
        "low": [9.8, 10.9],
        "window_start": [1681099200000000000, 1681099200000000000],
        "transactions": [50, 60],
        # A dummy underlying column; note that process_option_tickers will extract the underlying from the ticker
        "underlying": ["AAPL", "AAPL"],
        "opt_type": ["C", "P"],
        # The expiration_date column will be overwritten by process_option_tickers (parsed from the ticker)
        "expiration_date": [230617, 230617],
        # Dummy strike_price column; will be replaced by ticker-derived strike_price
        "strike_price": [150, 150],
        "stock_volume": [10000, 12000],
        "stock_open": [148.0, 148.5],
        "stock_close": [149.5, 149.7],
        "stock_high": [150.0, 150.2],
        "stock_low": [147.5, 147.8],
        "stock_transactions": [500, 600]
    })
    
    processed_df = process_options_data(df, risk_free_rate=0.01)
    print(processed_df)
