"""
This module provides a concrete implementation of the IFileHandler interface for interacting
with the local filesystem. The LocalFileHandler class handles file uploads and downloads
within the local filesystem, managing file paths and ensuring proper file operations.
"""

import os
import shutil
import logging
from bet_edge.data_io.interfaces import IFileHandler
from io import BytesIO

logger = logging.getLogger(__name__)


class LocalFileHandler(IFileHandler):
    """
    A file handler implementation for managing file uploads and downloads within the local filesystem.

    This class facilitates copying files from one local path to another or handling in-memory streams,
    ensuring that the destination directories exist and handling any file operation errors gracefully.
    """

    def __init__(self, base_directory: str = ""):
        """
        Initializes the LocalFileHandler with a specified base directory.

        Parameters:
            base_directory (str, optional): The base directory for file operations.
                                            If not provided, defaults to the current working directory.

        Notes:
            - The base_directory serves as the root for all relative file paths used in upload and download operations.
            - Ensure that the application has the necessary permissions to read from and write to the specified directories.
        """
        if not base_directory:
            base_directory = os.getcwd()
        self.base_directory = os.path.abspath(base_directory)
        logger.debug(f"LocalFileHandler initialized with base directory: {self.base_directory}")

    def upload(self, source_path: str, destination_path: str) -> None:
        """
        Uploads a file from the source path to the destination path within the local filesystem.

        Parameters:
            source_path (str): The local file path of the file to upload (copy).
            destination_path (str): The local file path where the file will be copied to.

        Raises:
            FileNotFoundError: If the source file does not exist.
            Exception: If the file fails to copy due to other issues.

        Notes:
            - Both source_path and destination_path can be absolute or relative to the base_directory.
            - Ensures that the destination directory exists; if not, it creates the necessary directories.
        """
        abs_source = self._resolve_path(source_path)
        abs_destination = self._resolve_path(destination_path)

        logger.debug(f"Uploading file from {abs_source} to {abs_destination}")

        if not os.path.isfile(abs_source):
            logger.error(f"Source file does not exist: {abs_source}")
            raise FileNotFoundError(f"Source file does not exist: {abs_source}")

        destination_dir = os.path.dirname(abs_destination)
        if not os.path.exists(destination_dir):
            try:
                os.makedirs(destination_dir, exist_ok=True)
                logger.debug(f"Created destination directory: {destination_dir}")
            except Exception as e:
                logger.error(f"Failed to create destination directory {destination_dir}: {e}")
                raise

        try:
            shutil.copy2(abs_source, abs_destination)
            logger.info(f"Copied file from {abs_source} to {abs_destination}")
        except Exception as e:
            logger.error(f"Failed to copy file: {e}")
            raise

    def download(self, source_path: str, destination_path: str) -> None:
        """
        Downloads a file from the source path to the destination path within the local filesystem.

        Parameters:
            source_path (str): The local file path of the file to download (copy).
            destination_path (str): The local file path where the file will be copied to.

        Raises:
            FileNotFoundError: If the source file does not exist.
            Exception: If the file fails to copy due to other issues.

        Notes:
            - Both source_path and destination_path can be absolute or relative to the base_directory.
            - Ensures that the destination directory exists; if not, it creates the necessary directories.
        """
        # The download operation is effectively the same as upload in a local context
        self.upload(source_path, destination_path)

    def upload_stream(self, data: BytesIO, destination_path: str) -> None:
        """
        Uploads data from a binary stream to the specified destination path within the local filesystem.

        Parameters:
            data (BytesIO): A binary stream containing the data to upload.
            destination_path (str): The local file path where the data will be written.

        Raises:
            Exception: If the file fails to write due to I/O issues.

        Notes:
            - Ensures that the destination directory exists; if not, it creates the necessary directories.
        """
        abs_destination = self._resolve_path(destination_path)
        destination_dir = os.path.dirname(abs_destination)
        if not os.path.exists(destination_dir):
            try:
                os.makedirs(destination_dir, exist_ok=True)
                logger.debug(f"Created destination directory for stream upload: {destination_dir}")
            except Exception as e:
                logger.error(f"Failed to create destination directory {destination_dir}: {e}")
                raise

        try:
            with open(abs_destination, "wb") as f:
                shutil.copyfileobj(data, f)
            logger.info(f"Uploaded data stream to {abs_destination}")
        except Exception as e:
            logger.error(f"Failed to upload data stream: {e}")
            raise

    def download_stream(self, source_path: str) -> BytesIO:
        """
        Downloads data from the specified source path within the local filesystem and returns it as a binary stream.

        Parameters:
            source_path (str): The local file path of the file to download.

        Returns:
            BytesIO: A binary stream containing the downloaded data.

        Raises:
            FileNotFoundError: If the source file does not exist.
            Exception: If the file fails to read due to I/O issues.

        Notes:
            - Ensures that the source file exists before attempting to read.
        """
        abs_source = self._resolve_path(source_path)

        logger.debug(f"Downloading data stream from {abs_source}")

        if not os.path.isfile(abs_source):
            logger.error(f"Source file does not exist: {abs_source}")
            raise FileNotFoundError(f"Source file does not exist: {abs_source}")

        buffer = BytesIO()
        try:
            with open(abs_source, "rb") as f:
                shutil.copyfileobj(f, buffer)
            buffer.seek(0)
            logger.info(f"Downloaded data stream from {abs_source}")
            return buffer
        except Exception as e:
            logger.error(f"Failed to download data stream: {e}")
            raise

    def _resolve_path(self, path: str) -> str:
        """
        Resolves a given path to an absolute path based on the base_directory.

        Parameters:
            path (str): The file path to resolve.

        Returns:
            str: The absolute file path.
        """
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(self.base_directory, path))
