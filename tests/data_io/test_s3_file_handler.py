from io import BytesIO

import pytest
from moto import mock_aws
import boto3

from bet_edge.data_io.interfaces import ICredentialProvider
from bet_edge.data_io.s3_file_handler import S3FileHandler


# Define a simple CredentialProvider for testing purposes
class MockCredentialProvider(ICredentialProvider):
    def get_credentials(self):
        return {
            "aws_access_key_id": "testing",
            "aws_secret_access_key": "testing",
            "aws_session_token": "testing",
            "aws_default_region": "us-east-1",
        }


@pytest.fixture
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    return {
        "aws_access_key_id": "testing",
        "aws_secret_access_key": "testing",
        "aws_session_token": "testing",
        "aws_default_region": "us-east-1",
    }


@pytest.fixture
def s3_mock(aws_credentials):
    """Start a mock S3 service."""
    with mock_aws():
        # Initialize boto3 client with mocked credentials
        boto3_client = boto3.client(
            "s3",
            region_name=aws_credentials["aws_default_region"],
            aws_access_key_id=aws_credentials["aws_access_key_id"],
            aws_secret_access_key=aws_credentials["aws_secret_access_key"],
            aws_session_token=aws_credentials["aws_session_token"],
        )
        # Create a mock S3 bucket
        bucket_name = "test-bucket"
        boto3_client.create_bucket(Bucket=bucket_name)

        yield bucket_name  # Provide the bucket name to the tests


@pytest.fixture
def s3_handler(s3_mock, aws_credentials):
    """Initialize the S3FileHandler with mocked S3."""
    credential_provider = MockCredentialProvider()
    handler = S3FileHandler(bucket_name=s3_mock, credential_provider=credential_provider)
    return handler


def test_upload_stream_success(s3_handler, s3_mock):
    """
    Test successful upload of data from an in-memory stream to the S3 bucket.
    """
    # Prepare in-memory data
    data = BytesIO(b"Test data for upload stream.")
    destination_key = "uploads/test_upload_stream.txt"

    # Perform upload
    s3_handler.upload_stream(data, destination_key)

    # Initialize boto3 client to verify upload
    s3_client = boto3.client("s3", region_name="us-east-1")
    response = s3_client.get_object(Bucket=s3_mock, Key=destination_key)
    uploaded_data = response["Body"].read()

    assert uploaded_data == b"Test data for upload stream."


def test_download_stream_success(s3_handler, s3_mock):
    """
    Test successful download of data into an in-memory stream from the S3 bucket.
    """
    # Initialize boto3 client to upload test data
    s3_client = boto3.client("s3", region_name="us-east-1")
    source_key = "downloads/test_download_stream.txt"
    test_data = b"Test data for download stream."
    s3_client.put_object(Bucket=s3_mock, Key=source_key, Body=test_data)

    # Perform download
    downloaded_stream = s3_handler.download_stream(source_key)

    # Read data from the downloaded stream
    downloaded_data = downloaded_stream.read()

    assert downloaded_data == test_data


def test_upload_stream_empty(s3_handler, s3_mock):
    """
    Test uploading an empty in-memory stream to the S3 bucket.
    """
    # Prepare empty in-memory data
    data = BytesIO(b"")
    destination_key = "uploads/empty_upload_stream.txt"

    # Perform upload
    s3_handler.upload_stream(data, destination_key)

    # Initialize boto3 client to verify upload
    s3_client = boto3.client("s3", region_name="us-east-1")
    response = s3_client.get_object(Bucket=s3_mock, Key=destination_key)
    uploaded_data = response["Body"].read()

    assert uploaded_data == b""


def test_download_stream_non_existent(s3_handler, s3_mock):
    """
    Test that downloading a non-existent key raises an appropriate exception.
    """
    source_key = "downloads/non_existent_stream.txt"

    # Perform download and expect an exception
    with pytest.raises(Exception) as exc_info:
        s3_handler.download_stream(source_key)

    assert "Not Found" in str(exc_info.value)


def test_upload_download_stream_binary(s3_handler, s3_mock):
    """
    Test uploading and downloading binary data to ensure data integrity.
    """
    # Prepare binary data
    binary_data = BytesIO(b"\x00\xff\x00\xff")
    destination_key = "uploads/binary_upload_stream.bin"

    # Perform upload
    s3_handler.upload_stream(binary_data, destination_key)

    # Perform download
    downloaded_stream = s3_handler.download_stream(destination_key)
    downloaded_data = downloaded_stream.read()

    assert downloaded_data == b"\x00\xff\x00\xff"


def test_multiple_uploads_downloads(s3_handler, s3_mock):
    """
    Test uploading and downloading multiple streams to ensure isolation and correctness.
    """
    # Prepare multiple in-memory data streams
    data1 = BytesIO(b"First stream data.")
    data2 = BytesIO(b"Second stream data.")
    key1 = "uploads/first_stream.txt"
    key2 = "uploads/second_stream.txt"

    # Perform uploads
    s3_handler.upload_stream(data1, key1)
    s3_handler.upload_stream(data2, key2)

    # Perform downloads
    downloaded_stream1 = s3_handler.download_stream(key1)
    downloaded_stream2 = s3_handler.download_stream(key2)

    # Read data from downloaded streams
    downloaded_data1 = downloaded_stream1.read()
    downloaded_data2 = downloaded_stream2.read()

    assert downloaded_data1 == b"First stream data."
    assert downloaded_data2 == b"Second stream data."
