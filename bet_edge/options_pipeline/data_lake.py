import os
import logging
import gzip
import shutil
from typing import List, Union
from datetime import datetime, timedelta

import polars as pl
import boto3
from botocore.config import Config

from bet_edge.data_io.env_cred_provider import EnvCredProvider

DATA_LAKE_DIR = r"c:\Users\Ashle\OneDrive\Documents\data_lake"
dividend_cache = {}
earnings_cache = {}

logger = logging.getLogger(__name__)

# Initialize AWS S3 session
ecp = EnvCredProvider()
creds = ecp.get_credentials()

session = boto3.Session(
    aws_access_key_id=creds["polygon_access_key_id"],
    aws_secret_access_key=creds["polygon_secret_access_key"],
)

s3 = session.client(
    "s3",
    endpoint_url="https://files.polygon.io",
    config=Config(signature_version="s3v4"),
)


def load_parquet_file(category: str, date_str: str, agg_key: str = "day_aggs_v1") -> pl.DataFrame:
    """
    Load a Parquet file from the data lake for the given category and date.

    Args:
        data_lake_path (str): Base directory of the data lake.
        category (str): The data category (e.g., 'us_stocks_sip' or 'us_options_opra').
        date_str (str): Date string in the format 'YYYY-MM-DD'.
        agg_key (str, optional): Aggregation key for the file structure. Defaults to "day_aggs_v1".

    Returns:
        pl.DataFrame: The loaded Polars DataFrame.
    """
    year, month, day = date_str.split("-")
    file_path = os.path.join(DATA_LAKE_DIR, category, agg_key, year, month, f"{date_str}.parquet")

    if os.path.exists(file_path):
        logger.info(f"Loading file: {file_path}")
        return pl.read_parquet(file_path)
    else:
        raise FileNotFoundError(f"File not found: {file_path}")
    

def scan_parquet_file(category: str, date_str: str, agg_key: str = "day_aggs_v1") -> pl.LazyFrame:
    """
    Load a Parquet file from the data lake for the given category and date.

    Args:
        data_lake_path (str): Base directory of the data lake.
        category (str): The data category (e.g., 'us_stocks_sip' or 'us_options_opra').
        date_str (str): Date string in the format 'YYYY-MM-DD'.
        agg_key (str, optional): Aggregation key for the file structure. Defaults to "day_aggs_v1".

    Returns:
        pl.DataFrame: The loaded Polars DataFrame.
    """
    year, month, day = date_str.split("-")
    file_path = os.path.join(DATA_LAKE_DIR, category, agg_key, year, month, f"{date_str}.parquet")

    if os.path.exists(file_path):
        logger.info(f"Loading file: {file_path}")
        return pl.scan_parquet(file_path)
    else:
        raise FileNotFoundError(f"File not found: {file_path}")


def download_and_process_s3_data(bucket_name: str, prefix: str):
    """
    Download, decompress, and convert CSV files from S3 into Parquet format.

    Args:
        bucket_name (str): The name of the S3 bucket.
        prefix (str): The prefix of the files to download from S3.
        base_local_dir (str, optional): The base directory for local storage. Defaults to DATA_LAKE_DIR.
    """
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            object_key = obj["Key"]
            if not object_key.endswith(".csv.gz"):
                continue

            try:
                local_file_path = os.path.join(DATA_LAKE_DIR, *object_key.split("/"))
                decompressed_file_path = local_file_path.replace(".gz", "")
                parquet_file_path = decompressed_file_path.replace(".csv", ".parquet")

                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                logger.info(f"Downloading: {object_key}")
                s3.download_file(bucket_name, object_key, local_file_path)

                logger.info(f"Decompressing: {local_file_path}")
                with gzip.open(local_file_path, "rb") as f_in:
                    with open(decompressed_file_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

                logger.info(f"Converting to Parquet: {parquet_file_path}")
                df = pl.read_csv(decompressed_file_path)
                df.write_parquet(parquet_file_path)

                os.remove(local_file_path)
                os.remove(decompressed_file_path)

                logger.info(f"Stored as Parquet: {parquet_file_path}")

            except Exception as e:
                logger.error(f"Error processing {object_key}: {e}")


def load_stock_and_option_data(start_date: str, end_date: str, stock_category: str = "us_stocks_sip", options_category: str = "us_options_opra"):
    """
    Load stock and options data for a range of dates.

    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        base_dir (str, optional): Base directory for the data lake. Defaults to DATA_LAKE_DIR.

    Returns:
        tuple: (stock_df, options_df) where each is a Polars DataFrame containing 
               data concatenated over the date range.
    """
    stock_dfs = []
    options_dfs = []

    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")

    while current_date <= end_date_obj:
        date_str = current_date.strftime("%Y-%m-%d")
        logger.info(f"Loading data for date: {date_str}")

        try:
            stock_df = load_parquet_file(stock_category, date_str)
            if stock_df is not None:
                stock_dfs.append(stock_df)
        except Exception as e:
            logger.error(f"Error loading stock data for {date_str}: {e}")

        try:
            options_df = load_parquet_file(options_category, date_str)
            if options_df is not None:
                options_dfs.append(options_df)
        except Exception as e:
            logger.error(f"Error loading options data for {date_str}: {e}")

        current_date += timedelta(days=1)

    # Concatenate the DataFrames if any data was loaded.
    if stock_dfs:
        stock_data = pl.concat(stock_dfs)
    else:
        stock_data = pl.DataFrame()

    if options_dfs:
        options_data = pl.concat(options_dfs)
    else:
        options_data = pl.DataFrame()

    return stock_data, options_data


def load_dividend_data(tickers: Union[List[str], str]) -> pl.DataFrame:
    """
    Load dividend data for one or more tickers from the data lake.
    Uses a cache to avoid multiple reads.
    
    Args:
        tickers (List[str] | str): List of stock ticker symbols or a single ticker symbol.
        base_dir (str, optional): Base directory for the data lake. Defaults to "data_lake".

    Returns:
        pl.DataFrame: The loaded dividend data for all requested tickers.
    """
    if isinstance(tickers, str):
        tickers = [tickers]  # Convert single ticker to list
    
    dfs = []
    for ticker in tickers:
        if ticker in dividend_cache:
            df = dividend_cache[ticker]
        else:
            file_path = os.path.join(DATA_LAKE_DIR, "dividends_by_ticker", f"{ticker}.parquet")
            if os.path.exists(file_path):
                df = pl.read_parquet(file_path)
                # Ensure 'ex_dividend_date' is in datetime format
                if df.schema.get("ex_dividend_date") not in {pl.Datetime("ns"), pl.Datetime("ms")}:
                    df = df.with_columns(
                        pl.col("ex_dividend_date").str.strptime(pl.Datetime, format="%Y-%m-%d")
                    )
                dividend_cache[ticker] = df
            else:
                dividend_cache[ticker] = pl.DataFrame()
                df = pl.DataFrame()

        if df.height > 0:
            # Add a ticker column to identify data source
            df = df.with_columns(pl.lit(ticker).alias("ticker"))
            dfs.append(df)

    # Return concatenated DataFrame if there are multiple tickers, else return single DataFrame
    return pl.concat(dfs) if dfs else pl.DataFrame()


def load_earnings_data(tickers: Union[List[str], str]) -> pl.DataFrame:
    """
    Load earnings data for one or more tickers from the data lake.
    Uses a cache to avoid multiple reads.
    
    Args:
        tickers (List[str] | str): List of stock ticker symbols or a single ticker symbol.
        base_dir (str, optional): Base directory for the data lake. Defaults to "data_lake".

    Returns:
        pl.DataFrame: The loaded earnings data for all requested tickers.
    """
    if isinstance(tickers, str):
        tickers = [tickers]  # Convert single ticker to list
    
    dfs = []
    for ticker in tickers:
        if ticker in earnings_cache:
            df = earnings_cache[ticker]
        else:
            file_path = os.path.join(DATA_LAKE_DIR, "earnings_by_ticker", f"{ticker}.parquet")
            if os.path.exists(file_path):
                df = pl.read_parquet(file_path)
                # Ensure 'Earnings Date' is in datetime format
                #if df.schema.get("Earnings Date") not in {pl.Datetime("ns"), pl.Datetime("ms")}:
                #    df = df.with_columns(
                #        pl.col("Earnings Date").str.strptime(pl.Datetime, format="%Y-%m-%d")
                #    )
                earnings_cache[ticker] = df
            else:
                earnings_cache[ticker] = pl.DataFrame()
                df = pl.DataFrame()

        if df.height > 0:
            # Add a ticker column to identify data source
            df = df.with_columns(pl.lit(ticker).alias("ticker"))
            dfs.append(df)

    # Return concatenated DataFrame if multiple tickers, else return an empty DataFrame
    return pl.concat(dfs) if dfs else pl.DataFrame()