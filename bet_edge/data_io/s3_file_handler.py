import os
import logging
import boto3
from bet_edge.data_io.interfaces import IFileHandler, ICredentialProvider
from io import BytesIO

logger = logging.getLogger(__name__)


class S3FileHandler(IFileHandler):
    """
    A file handler implementation for managing file uploads and downloads to/from AWS S3.

    This class depends on an ICredentialProvider to supply credentials dynamically,
    making it flexible and adaptable to different credential sources.
    """

    def __init__(self, bucket_name: str, credential_provider: ICredentialProvider, temp_file_path: str = ""):
        """
        Initializes the S3FileHandler with a specified bucket and credentials.

        Parameters:
            bucket_name (str): The name of the S3 bucket to interact with.
            credential_provider (ICredentialProvider): An instance of a credential provider that supplies
                                                      AWS credentials dynamically.
            temp_file_path (str, optional): The path to the temporary directory for local file operations.
                                            Defaults to a `data` directory relative to the module.

        Notes:
            - The S3FileHandler uses boto3 to interact with S3. Ensure that the provided credentials
              have the necessary permissions for the desired operations (e.g., s3:PutObject, s3:GetObject).
        """
        self.bucket_name = bucket_name
        creds = credential_provider.get_credentials()

        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=creds["aws_access_key_id"],
            aws_secret_access_key=creds["aws_secret_access_key"],
            aws_session_token=creds.get("aws_session_token"),
            region_name=creds.get("aws_default_region"),
        )

        # If no temp path is provided, default to a `data` directory adjacent to the module
        if not temp_file_path:
            temp_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
        self.temp_file_path = temp_file_path

    def upload(self, source_path: str, destination_path: str) -> None:
        """
        Uploads a local file to the specified S3 bucket.

        Parameters:
            source_path (str): The local file path of the file to upload.
            destination_path (str): The S3 object key where the file will be stored.

        Raises:
            Exception: If the file fails to upload.

        Notes:
            - Ensure the `destination_path` includes any necessary path structure in the bucket (e.g., "folder/file.txt").
        """
        try:
            self.s3.upload_file(source_path, self.bucket_name, destination_path)
            logger.info(f"Uploaded {destination_path} to S3 bucket {self.bucket_name}")
        except Exception as e:
            logger.error(f"Failed to upload file to S3: {e}")
            raise

    def download(self, source_path: str, destination_path: str) -> None:
        """
        Downloads a file from the specified S3 bucket to a local path.

        Parameters:
            source_path (str): The S3 object key of the file to download.
            destination_path (str): The local file path where the downloaded file will be saved.

        Raises:
            Exception: If the file fails to download.

        Notes:
            - Ensure the `source_path` corresponds to a valid object in the bucket.
        """
        try:
            self.s3.download_file(self.bucket_name, source_path, destination_path)
            logger.debug(f"Downloaded {source_path} to {destination_path}")
        except Exception as e:
            logger.error(f"Failed to download file from S3: {e}")
            raise

    def upload_stream(self, data: BytesIO, destination_path: str) -> None:
        """
        Uploads data from a binary stream to the specified S3 bucket.

        Parameters:
            data (BytesIO): A binary stream containing the data to upload.
            destination_path (str): The S3 object key where the data will be stored.

        Raises:
            Exception: If the upload operation fails.

        Notes:
            - Ensure the `destination_path` includes any necessary path structure in the bucket (e.g., "folder/file.txt").
        """
        try:
            self.s3.upload_fileobj(data, self.bucket_name, destination_path)
            logger.info(f"Uploaded data stream to {destination_path} in S3 bucket {self.bucket_name}")
        except Exception as e:
            logger.error(f"Failed to upload data stream to S3: {e}")
            raise

    def download_stream(self, source_path: str) -> BytesIO:
        """
        Downloads data from the specified S3 bucket and returns it as a binary stream.

        Parameters:
            source_path (str): The S3 object key of the file to download.

        Returns:
            BytesIO: A binary stream containing the downloaded data.

        Raises:
            Exception: If the download operation fails.

        Notes:
            - Ensure the `source_path` corresponds to a valid object in the bucket.
        """
        buffer = BytesIO()
        try:
            self.s3.download_fileobj(self.bucket_name, source_path, buffer)
            buffer.seek(0)
            logger.info(f"Downloaded data stream from {source_path} in S3 bucket {self.bucket_name}")
            return buffer
        except Exception as e:
            logger.error(f"Failed to download data stream from S3: {e}")
            raise
