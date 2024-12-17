"""
This module provides interfaces for interacting with files and managing credentials.
The interfaces define contracts for file handling operations (e.g., upload, download)
and credential retrieval, which can be implemented by various providers and handlers.
"""

from abc import ABC, abstractmethod
from io import BytesIO


class ICredentialProvider(ABC):
    """
    An interface for credential providers that fetch and supply credentials for
    accessing various services (e.g., AWS, databases, etc.).

    Implementations of this interface must define a method for retrieving credentials
    in the form of a dictionary.
    """

    @abstractmethod
    def get_credentials(self) -> dict:
        """
        Fetches credentials and returns them as a dictionary.

        Returns:
            dict: A dictionary containing credentials, e.g.:
                {
                    "aws_access_key_id": ...,
                    "aws_secret_access_key": ...,
                    "aws_session_token": ...,
                    "aws_default_region": ...
                }

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        pass


class IFileHandler(ABC):
    """
    An interface for file handlers that define operations for managing files, such as
    uploading and downloading them to/from various storage backends (e.g., AWS S3, local filesystem).
    """

    @abstractmethod
    def upload(self, source_path: str, destination_path: str) -> None:
        """
        Uploads a file from a source path to a specified destination.

        Parameters:
            source_path (str): The local or remote path to the file to be uploaded.
            destination_path (str): The destination path or identifier (e.g., S3 object key, local file path).

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        pass

    @abstractmethod
    def download(self, source_path: str, destination_path: str) -> None:
        """
        Downloads a file from a specified source to a destination path.

        Parameters:
            source_path (str): The source path or identifier (e.g., S3 object key, local file path).
            destination_path (str): The local or remote path where the downloaded file will be saved.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        pass

    @abstractmethod
    def upload_stream(self, data: BytesIO, destination_path: str) -> None:
        """
        Uploads data from a binary stream to a specified destination.

        Parameters:
            data (BytesIO): A binary stream containing the data to upload.
            destination_path (str): The destination path or identifier.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        pass

    @abstractmethod
    def download_stream(self, source_path: str) -> BytesIO:
        """
        Downloads data from a specified source and returns it as a binary stream.

        Parameters:
            source_path (str): The source path or identifier.

        Returns:
            BytesIO: A binary stream using an in-memory bytes buffer containing the downloaded data.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        pass
