import os
import logging
from typing import Optional

import s3fs
import polars as pl

from bet_edge.data_io.interfaces import IDataStorage
from bet_edge.data_io.storage.local_data import LocalStorage
from bet_edge.data_io.storage.postgres_data import PostgresStorage
from bet_edge.data_io.storage.s3_data import S3Storage
from bet_edge.data_io.env_cred_provider import EnvCredProvider

CRED_PROVIDER = EnvCredProvider()
logger = logging.getLogger()


def infer_format_from_extension(identifier: str, default_format: str = "binary") -> str:
    """
    Infers the data format based on the file extension of the given identifier.

    Args:
        identifier (str): The file path or S3 URI.
        default_format (str, optional): The default format to return if the extension is unrecognized. Defaults to "binary".

    Returns:
        str: The inferred data format ("csv", "parquet", "binary", etc.).
    """
    _, ext = os.path.splitext(identifier.lower())
    format_mapping = {
        '.csv': 'csv',
        '.parquet': 'parquet',
        '.json': 'json',
        '.txt': 'text',
        '.bin': 'binary',
        # Add more mappings as needed
    }
    return format_mapping.get(ext, default_format)


def transfer_data(
    source_storage: "IDataStorage",
    source_id: str,
    dest_storage: "IDataStorage",
    dest_id: str,
    format: Optional[str] = None
):
    """
    Transfers data from a source storage to a destination storage.

    Args:
        source_storage (IDataStorage): The storage object from which data will be read.
        source_id (str): The identifier (file path, S3 URI, or table name) of the data in the source storage.
        dest_storage (IDataStorage): The storage object to which data will be written.
        dest_id (str): The identifier (file path, S3 URI, or table name) for the data in the destination storage.
        format (Optional[str]): The format of the data ("binary", "csv", etc.). If None, inferred from the source_identifier.

    Returns:
        None
    """
    if format is None:
        format = infer_format_from_extension(source_id)
        logger.info(f"Inferred format '{format}' from source identifier '{source_id}'.")
    else:
        logger.info(f"Using provided format '{format}' for transfer.")

    data = source_storage.read(source_id, format=format)
    dest_storage.write(data, dest_id, format=format)
    logger.info(f"Data transferred from '{source_id}' ({type(source_storage).__name__}) "
          f"to '{dest_id}' ({type(dest_storage).__name__}) in '{format}' format.")


def transfer_local_to_s3(
    source_file_path: str,
    dest_s3_uri: str,
    format: Optional[str] = None,
):
    """
    Transfer data from a local file to an S3 URI.

    Args:
        source_file_path (str): Path to the local file (e.g., "/path/to/source.csv").
        dest_s3_uri (str): The S3 URI (e.g., "s3://bucket/key.csv").
        format (str, optional): The data format ("binary", "csv", etc.). Defaults to "binary".
    """
    local_storage = LocalStorage()
    s3_storage = S3Storage(CRED_PROVIDER)

    transfer_data(
        source_storage=local_storage,
        source_id=source_file_path,
        dest_storage=s3_storage,
        dest_id=dest_s3_uri,
        format=format
    )


def transfer_s3_to_local(
    source_s3_uri: str,
    destination_file_path: str,
    format: Optional[str] = None,

):
    """
    Transfer data from an S3 URI to a local file.

    Args:
        source_s3_uri (str): The S3 URI (e.g., "s3://bucket/key.csv").
        destination_file_path (str): Path to the local file (e.g., "/path/to/dest.csv").
        format (Optional[str], optional): The data format ("binary", "csv", etc.). Defaults to None (inferred from source_s3_uri).
    """

    s3_storage = S3Storage(CRED_PROVIDER)
    local_storage = LocalStorage()

    transfer_data(
        source_storage=s3_storage,
        source_id=source_s3_uri,
        dest_storage=local_storage,
        dest_id=destination_file_path,
        format=format
    )

def transfer_local_to_postgres(
    source_file_path: str,
    destination_table: str,
    format: Optional[str] = None,
):
    """
    Transfer data from a local file to a PostgreSQL table.

    Args:
        source_file_path (str): Path to the local file (e.g., "/path/to/source.csv").
        destination_table (str): PostgreSQL table name (e.g., "public.my_table").
        format (Optional[str], optional): The data format ("csv" usually). Defaults to None (inferred from source_file_path).
    """

    local_storage = LocalStorage()
    postgres_storage = PostgresStorage(CRED_PROVIDER)

    transfer_data(
        source_storage=local_storage,
        source_id=source_file_path,
        dest_storage=postgres_storage,
        dest_id=destination_table,
        format=format
    )

def transfer_postgres_to_local(
    source_table: str,
    dest_file_path: str,
    format: Optional[str] = None,
):
    """
    Transfer data from a PostgreSQL table to a local file.

    Args:
        source_table (str): PostgreSQL table name (e.g., "public.my_table").
        dest_file_path (str): Path to the local file (e.g., "/path/to/dest.csv").
        format (Optional[str], optional): The data format ("csv" usually). Defaults to None (inferred from destination_file_path).
    """
    local_storage = LocalStorage()
    postgres_storage = PostgresStorage(CRED_PROVIDER)

    transfer_data(
        source_storage=postgres_storage,
        source_id=source_table,
        dest_storage=local_storage,
        dest_id=dest_file_path,
        format=format
    )


def transfer_polars_to_s3(
        source_df: pl.DataFrame,
        s3_uri: str,   
):
    """
    Write a polars dataframe to S3.

    Args:
        source_df (pl.DataFrame): Source dataframe to write to S3.
        s3_uri (str): The S3 URI (e.g., "s3://bucket/key.csv").
    """
    ext = infer_format_from_extension(s3_uri)
    # Validate supported formats
    supported_formats = ['csv', 'parquet']
    if ext not in supported_formats:
        raise ValueError(f"Unsupported format '{ext}'. Supported formats are: {supported_formats}")
    
    fs = s3fs.S3FileSystem()
    with fs.open(s3_uri, mode='wb') as f:
        if ext == 'csv':
            source_df.write_csv(f)  #type: ignore
        elif ext == 'parquet':
            source_df.write_parquet(f) #type: ignore


def transfer_s3_to_polars(
        s3_uri: str,   
):
    """
    Load an S3 object to a polars dataframe.

    Args:
        s3_uri (str): The S3 URI (e.g., "s3://bucket/key.csv").
    """
    creds = CRED_PROVIDER.get_credentials()
    storage_options = {"key" : creds["aws_access_key_id"], "secret" : creds["aws_secret_access_key"], "region_name" : creds["aws_default_region"]}
    ext = infer_format_from_extension(s3_uri)
    # Validate supported formats
    supported_formats = ['csv', 'parquet']
    if ext not in supported_formats:
        raise ValueError(f"Unsupported format '{ext}'. Supported formats are: {supported_formats}")
    
    if ext == 'csv':
        return pl.read_csv(s3_uri, storage_options=storage_options)
    elif ext == 'parquet':
        return pl.read_parquet(s3_uri, storage_options=storage_options)