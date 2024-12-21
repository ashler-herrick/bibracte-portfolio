import re
from io import BytesIO
from typing import Optional
import boto3
from ..interfaces import IDataStorage, ICredentialProvider

class S3DataStorage(IDataStorage):
    """
    Implementation of the IDataStorage interface for AWS S3.

    This class provides methods for storing and retrieving data from AWS S3 buckets using
    S3 URIs.

    Attributes:
        S3_URI_REGEX (re.Pattern): A compiled regular expression to parse S3 URIs.
        s3 (boto3.client): The boto3 client used to interact with S3.
    """

    S3_URI_REGEX = re.compile(r'^s3://([^/]+)/(.+)$')

    def __init__(self, credential_provider: Optional[ICredentialProvider] = None):
        """
        Initializes the S3 storage with an optional credential provider.

        Args:
            credential_provider (Optional[ICredentialProvider]): A credential provider for handling authentication.

        Raises:
            boto3.exceptions.Boto3Error: If the boto3 client cannot be initialized.
        """
        super().__init__(credential_provider=credential_provider)
        creds = self.credential_provider.get_credentials() if self.credential_provider else {}
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=creds.get('aws_access_key_id'),
            aws_secret_access_key=creds.get('aws_secret_access_key'),
            aws_session_token=creds.get('aws_session_token'),
            region_name=creds.get('aws_default_region')
        )

    def _parse_s3_uri(self, uri: str):
        """
        Parses an S3 URI into its bucket and key components.

        Args:
            uri (str): The S3 URI to parse (e.g., "s3://bucket-name/key").

        Returns:
            tuple: A tuple containing the bucket name and key.

        Raises:
            ValueError: If the URI is not a valid S3 URI.
        """
        match = self.S3_URI_REGEX.match(uri)
        if not match:
            raise ValueError(f"Invalid S3 URI: {uri}")
        bucket, key = match.groups()
        return bucket, key

    def write_from_file(self, source_file_path: str, destination_identifier: str, format: str = "binary") -> None:
        """
        Writes data from a local file to the specified S3 location.

        Args:
            source_file_path (str): The path to the local file to upload.
            destination_identifier (str): The S3 URI where the file will be uploaded.
            format (str, optional): The format of the file (ignored, default is "binary").
        """
        bucket, key = self._parse_s3_uri(destination_identifier)
        self.s3.upload_file(source_file_path, bucket, key)

    def read_to_file(self, source_identifier: str, destination_file_path: str, format: str = "binary") -> None:
        """
        Reads data from the specified S3 location and writes it to a local file.

        Args:
            source_identifier (str): The S3 URI of the file to download.
            destination_file_path (str): The local file path where the data will be written.
            format (str, optional): The format of the file (ignored, default is "binary").
        """
        bucket, key = self._parse_s3_uri(source_identifier)
        self.s3.download_file(bucket, key, destination_file_path)

    def write(self, data: BytesIO, destination_identifier: str, format: str = "binary") -> None:
        """
        Writes data from a BytesIO object to the specified S3 location.

        Args:
            data (BytesIO): The data to upload.
            destination_identifier (str): The S3 URI where the data will be uploaded.
            format (str, optional): The format of the data (ignored, default is "binary").
        """
        bucket, key = self._parse_s3_uri(destination_identifier)
        self.s3.put_object(Bucket=bucket, Key=key, Body=data.getvalue())

    def read(self, source_identifier: str, format: str = "binary") -> BytesIO:
        """
        Reads data from the specified S3 location into a BytesIO object.

        Args:
            source_identifier (str): The S3 URI of the file to read.
            format (str, optional): The format of the file (ignored, default is "binary").

        Returns:
            BytesIO: A BytesIO object containing the data read from S3.
        """
        bucket, key = self._parse_s3_uri(source_identifier)
        obj = self.s3.get_object(Bucket=bucket, Key=key)
        return BytesIO(obj['Body'].read())
