import os
import logging
import gzip
import shutil
import polars as pl
import boto3
from botocore.config import Config
from bet_edge.data_io.env_cred_provider import EnvCredProvider

DATA_LAKE_DIR = r"c:\Users\Ashle\OneDrive\Documents\data_lake"
dividend_cache = {}
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


def load_parquet_file(data_lake_path: str, category: str, date_str: str, agg_key: str = "day_aggs_v1") -> pl.DataFrame:
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
    file_path = os.path.join(data_lake_path, category, agg_key, year, month, f"{date_str}.parquet")

    if os.path.exists(file_path):
        logger.info(f"Loading file: {file_path}")
        return pl.read_parquet(file_path)
    else:
        raise FileNotFoundError(f"File not found: {file_path}")


def download_and_process_s3_data(bucket_name: str, prefix: str, base_local_dir: str = DATA_LAKE_DIR):
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
                local_file_path = os.path.join(base_local_dir, *object_key.split("/"))
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


def load_stock_and_option_data(date_str: str, base_dir: str = DATA_LAKE_DIR):
    """
    Load stock and options data for the given date.

    Args:
        date_str (str): Date in 'YYYY-MM-DD' format.
        base_dir (str, optional): Base directory for the data lake. Defaults to DATA_LAKE_DIR.

    Returns:
        tuple: (stock_df, options_df) where each is a Polars DataFrame.
    """
    stock_category = "us_stocks_sip"
    options_category = "us_options_opra"

    logger.info(f"Loading data for date: {date_str}")

    stock_df = load_parquet_file(base_dir, stock_category, date_str)
    options_df = load_parquet_file(base_dir, options_category, date_str)

    return stock_df, options_df


def load_dividend_data(ticker: str, year: int, base_dir: str = DATA_LAKE_DIR) -> pl.DataFrame:
    """
    Load dividend data for a given ticker and year from the data lake.
    Uses a cache to avoid multiple reads.

    Args:
        ticker (str): The stock ticker symbol.
        year (int): The year for which dividend data is needed.
        base_dir (str, optional): Base directory for the data lake. Defaults to DATA_LAKE_DIR.

    Returns:
        pl.DataFrame: The loaded dividend data.
    """
    key = (ticker, year)
    if key in dividend_cache:
        return dividend_cache[key]

    file_path = os.path.join(base_dir, "dividends", f"{year}", f"{ticker}.parquet")
    if os.path.exists(file_path):
        df = pl.read_parquet(file_path)
        if df.schema.get("ex_dividend_date") not in {pl.Datetime("ns"), pl.Datetime("ms")}:
            df = df.with_columns(pl.col("ex_dividend_date").str.strptime(pl.Datetime, format="%Y-%m-%d"))
        dividend_cache[key] = df
        return df
    else:
        dividend_cache[key] = None
        return pl.DataFrame()
