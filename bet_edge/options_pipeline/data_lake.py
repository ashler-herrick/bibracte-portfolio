import os
import logging
from datetime import datetime

import polars as pl

DATA_LAKE_DIR = r"c:\Users\Ashle\OneDrive\Documents\data_lake"
dividend_cache = {}

logger = logging.getLogger(__name__)

def load_parquet_file(data_lake_path: str, category: str, date_str: str) -> pl.DataFrame:
    """
    Loads a Parquet file from the data lake for the given category and date.
    
    Args:
        data_lake_path (str): Base directory of your data lake.
        category (str): The data category (e.g., 'us_stocks_sip' or 'us_options_opra').
        date_str (str): Date string in the format 'YYYY-MM-DD'.
        
    Returns:
        pl.DataFrame or None: The loaded Polars DataFrame, or None if the file is not found.
    """
    # Split date into components
    year, month, day = date_str.split("-")
    file_path = os.path.join(data_lake_path, category, "day_aggs_v1", year, month, f"{date_str}.parquet")

    if os.path.exists(file_path):
        logging.info(f"Loading file: {file_path}")
        return pl.read_parquet(file_path)
    else:
        raise FileNotFoundError(f"File not found: {file_path}")

def load_data_for_day(date_str: str, base_dir: str = r"c:\Users\Ashle\OneDrive\Documents\data_lake"):
    """
    Loads stock and options data for the given date.
    
    Args:
        date_str (str): Date in 'YYYY-MM-DD' format.
        base_dir (str): Base directory for the data lake.
    
    Returns:
        tuple: (stock_df, options_df) where each is a Polars DataFrame or None.
    """
    stock_category = "us_stocks_sip"
    options_category = "us_options_opra"

    logging.info(f"Loading data for date: {date_str}")

    stock_df = load_parquet_file(base_dir, stock_category, date_str)
    options_df = load_parquet_file(base_dir, options_category, date_str)

    return stock_df, options_df

def get_dividend_data(ticker: str, year: int, base_dir: str = DATA_LAKE_DIR) -> pl.DataFrame:
    """
    Loads dividend data for a given ticker and year from the data lake.
    Uses a cache to avoid multiple reads.
    """
    key = (ticker, year)
    if key in dividend_cache:
        return dividend_cache[key]
    
    # Construct file path: e.g., dividends/year=2023/AAPL.parquet
    file_path = os.path.join(base_dir,'dividends', f"{year}", f"{ticker}.parquet")
    if os.path.exists(file_path):
        df = pl.read_parquet(file_path)
        # Ensure ex_dividend_date is a datetime column
        if df.schema.get("ex_dividend_date") not in {pl.Datetime("ns"), pl.Datetime("ms")}:
            df = df.with_columns(
                pl.col("ex_dividend_date").str.strptime(pl.Datetime, format="%Y-%m-%d")
            )
        dividend_cache[key] = df
        return df
    else:
        dividend_cache[key] = None
        return None

if __name__ == "__main__":
    date_str = "2023-04-10"  # Set the date for which you want to load data.
    stock_df, options_df = load_data_for_day(date_str)

    if stock_df is not None:
        print("Stock DataFrame loaded successfully.")
        print(f"Stock DataFrame Preview:\n{stock_df.head()}")
        print(f"Stock DataFrame Schema:\n{stock_df.schema}")

    if options_df is not None:
        print("Options DataFrame loaded successfully.")
        print(f"Options DataFrame Preview:\n{options_df.head()}")
        print(f"Options DataFrame Schema:\n{options_df.schema}")
