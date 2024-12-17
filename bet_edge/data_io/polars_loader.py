"""
This module provides the PolarsLoader class, which integrates Polars DataFrame operations
with a credential provider and file handler. The loader enables uploading and downloading
Polars DataFrames to/from a specified storage backend (e.g., AWS S3) through an IFileHandler
implementation.
"""

import os
import logging
import polars as pl
from bet_edge.data_io.interfaces import ICredentialProvider, IFileHandler
from io import BytesIO

logger = logging.getLogger(__name__)


class PolarsLoader:
    """
    A loader for handling Polars DataFrames, providing methods to upload them
    to and download them from a storage backend.

    The class depends on:
    - An `ICredentialProvider` for managing credentials (optional, based on the file handler).
    - An `IFileHandler` implementation for file operations, allowing flexibility in choosing
      the storage backend (e.g., AWS S3, local filesystem, database).
    """

    def __init__(self, credential_provider: ICredentialProvider, file_handler: IFileHandler, temp_file_path: str = ""):
        """
        Initializes the PolarsLoader with the provided credential provider, file handler, and temporary file path.

        Parameters:
            credential_provider (ICredentialProvider): The credential provider instance for managing credentials.
                                                       Used indirectly by the file handler.
            file_handler (IFileHandler): The file handler instance responsible for performing file operations.
            temp_file_path (str, optional): The path to a temporary directory for file operations.
                                            Defaults to a `data` directory relative to the module.

        Notes:
            - The `temp_file_path` is used for intermediate file operations when reading from or writing to Polars DataFrames.
        """
        self.credential_provider = credential_provider
        self.file_handler = file_handler
        self.temp_file_path = temp_file_path

        # Default to a local `data` directory if no temp_file_path is specified
        if not temp_file_path:
            self.temp_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

        os.makedirs(self.temp_file_path, exist_ok=True)

    def upload_polars_df(self, df: pl.DataFrame, key: str) -> None:
        """
        Uploads a Polars DataFrame to the storage backend via the file handler.

        Parameters:
            df (pl.DataFrame): The Polars DataFrame to upload.
            key (str): The destination path or key in the storage backend (e.g., S3 object key).

        Workflow:
            1. The DataFrame is written to an in-memory Parquet binary stream.
            2. The file handler uploads the binary stream to the storage backend.

        Notes:
            - The `key` parameter must include the appropriate path structure in the backend (if required).
            - Ensure the file handler is configured correctly for the chosen backend.

        Raises:
            Exception: If the upload operation fails.
        """
        buffer = BytesIO()
        df.write_parquet(buffer)
        buffer.seek(0)
        logger.debug(f"Polars DataFrame serialized to in-memory Parquet stream for key: {key}")
        try:
            self.file_handler.upload_stream(buffer, key)
            logger.info(f"Uploaded Polars DataFrame to {key} successfully.")
        except Exception as e:
            logger.error(f"Failed to upload Polars DataFrame to {key}: {e}")
            raise

    def download_to_polars_df(self, key: str) -> pl.DataFrame:
        """
        Downloads a file from the storage backend via the file handler and loads it as a Polars DataFrame.

        Parameters:
            key (str): The source path or key in the storage backend (e.g., S3 object key).

        Returns:
            pl.DataFrame: The downloaded Polars DataFrame.

        Workflow:
            1. The file handler downloads the file from the backend into an in-memory binary stream.
            2. The binary stream is read into a Polars DataFrame.

        Notes:
            - The `key` parameter must correspond to a valid file in the storage backend.
            - Ensure the file handler is configured correctly for the chosen backend.

        Raises:
            Exception: If the download or file reading operation fails.
        """
        try:
            buffer = self.file_handler.download_stream(key)
            df = pl.read_parquet(buffer)
            logger.info(f"Downloaded and loaded Polars DataFrame from {key} successfully.")
            return df
        except Exception as e:
            logger.error(f"Failed to download Polars DataFrame from {key}: {e}")
            raise
