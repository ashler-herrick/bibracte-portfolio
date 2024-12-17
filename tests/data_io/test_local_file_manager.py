# tests/data_io/test_local_file_manager.py

from io import BytesIO
import pytest
from pathlib import Path

# Import the LocalFileHandler class
# Adjust the import path based on your project structure
from bet_edge.data_io.local_file_handler import LocalFileHandler


@pytest.fixture
def base_dir(tmp_path):
    """
    Fixture to create a base directory for LocalFileHandler within a temporary path.
    """
    base_dir = tmp_path / "base_dir"
    base_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory is created
    return base_dir


@pytest.fixture
def handler(base_dir):
    """
    Fixture to initialize the LocalFileHandler with the provided base directory.
    """
    return LocalFileHandler(base_directory=str(base_dir))


def test_upload_success(handler, base_dir):
    """
    Test successful upload (copy) of a file within the local filesystem.
    """
    # Create source file
    source_file = base_dir / "source.txt"
    source_file.write_text("This is a test file.")

    # Define destination path
    destination_path = "destination/destination.txt"

    # Perform upload
    handler.upload(str(source_file), destination_path)

    # Assert the destination file exists and has the correct content
    dest_file = base_dir / destination_path
    assert dest_file.exists()
    assert dest_file.read_text() == "This is a test file."


def test_upload_relative_paths(handler, base_dir):
    """
    Test uploading using relative paths.
    """
    # Create source file in a subdirectory
    sub_dir = base_dir / "subdir"
    sub_dir.mkdir()
    source_file = sub_dir / "source.txt"
    source_file.write_text("Relative path test.")

    # Define destination path relative to base_directory
    destination_path = "../destination/destination.txt"

    # Perform upload
    handler.upload(str(source_file), destination_path)

    # Resolve expected destination path
    expected_dest = base_dir.parent / "destination" / "destination.txt"
    assert expected_dest.exists()
    assert expected_dest.read_text() == "Relative path test."


def test_upload_creates_destination_directory(handler, base_dir):
    """
    Test that uploading creates the destination directory if it does not exist.
    """
    # Create source file
    source_file = base_dir / "source.txt"
    source_file.write_text("Directory creation test.")

    # Define destination path in a nested directory
    destination_path = "nested/dir/destination.txt"

    # Perform upload
    handler.upload(str(source_file), destination_path)

    # Assert the destination directory and file exist
    dest_file = base_dir / destination_path
    assert dest_file.exists()
    assert dest_file.read_text() == "Directory creation test."


def test_upload_source_file_not_found(handler, base_dir):
    """
    Test that uploading a non-existent source file raises FileNotFoundError.
    """
    # Define non-existent source file
    source_file = base_dir / "non_existent.txt"

    # Define destination path
    destination_path = "destination/non_existent.txt"

    # Perform upload and expect FileNotFoundError
    with pytest.raises(FileNotFoundError) as exc_info:
        handler.upload(str(source_file), destination_path)

    assert "Source file does not exist" in str(exc_info.value)


def test_download_success(handler, base_dir):
    """
    Test successful download (copy) of a file within the local filesystem.
    """
    # Create source file
    source_file = base_dir / "source.txt"
    source_file.write_text("Download test file.")

    # Define destination path
    destination_path = "downloaded/destination.txt"

    # Perform download
    handler.download(str(source_file), destination_path)

    # Assert the destination file exists and has the correct content
    dest_file = base_dir / destination_path
    assert dest_file.exists()
    assert dest_file.read_text() == "Download test file."


def test_download_relative_paths(handler, base_dir):
    """
    Test downloading using relative paths.
    """
    # Create source file in a subdirectory
    sub_dir = base_dir / "subdir"
    sub_dir.mkdir()
    source_file = sub_dir / "source.txt"
    source_file.write_text("Relative download path test.")

    # Define destination path relative to base_directory
    destination_path = "../downloaded/destination.txt"

    # Perform download
    handler.download(str(source_file), destination_path)

    # Resolve expected destination path
    expected_dest = base_dir.parent / "downloaded" / "destination.txt"
    assert expected_dest.exists()
    assert expected_dest.read_text() == "Relative download path test."


def test_download_creates_destination_directory(handler, base_dir):
    """
    Test that downloading creates the destination directory if it does not exist.
    """
    # Create source file
    source_file = base_dir / "source.txt"
    source_file.write_text("Download directory creation test.")

    # Define destination path in a nested directory
    destination_path = "nested/dir/downloaded.txt"

    # Perform download
    handler.download(str(source_file), destination_path)

    # Assert the destination directory and file exist
    dest_file = base_dir / destination_path
    assert dest_file.exists()
    assert dest_file.read_text() == "Download directory creation test."


def test_download_source_file_not_found(handler, base_dir):
    """
    Test that downloading a non-existent source file raises FileNotFoundError.
    """
    # Define non-existent source file
    source_file = base_dir / "non_existent_download.txt"

    # Define destination path
    destination_path = "downloaded/non_existent_download.txt"

    # Perform download and expect FileNotFoundError
    with pytest.raises(FileNotFoundError) as exc_info:
        handler.download(str(source_file), destination_path)

    assert "Source file does not exist" in str(exc_info.value)


def test_upload_absolute_path(handler, base_dir):
    """
    Test uploading using absolute paths.
    """
    # Create source file
    source_file = base_dir / "absolute_source.txt"
    source_file.write_text("Absolute path upload test.")

    # Define absolute destination path within the temporary directory
    destination_dir = base_dir / "absolute_destination_dir"
    destination_path = str(destination_dir / "absolute_destination.txt")

    # Perform upload
    handler.upload(str(source_file), destination_path)

    # Assert the destination file exists and has the correct content
    dest_file = Path(destination_path)
    assert dest_file.exists()
    assert dest_file.read_text() == "Absolute path upload test."


def test_download_absolute_path(handler, base_dir):
    """
    Test downloading using absolute paths.
    """
    # Create source file
    source_file = base_dir / "absolute_source_download.txt"
    source_file.write_text("Absolute path download test.")

    # Define absolute destination path within the temporary directory
    destination_dir = base_dir / "absolute_download_dir"
    destination_path = str(destination_dir / "absolute_destination_download.txt")

    # Perform download
    handler.download(str(source_file), destination_path)

    # Assert the destination file exists and has the correct content
    dest_file = Path(destination_path)
    assert dest_file.exists()
    assert dest_file.read_text() == "Absolute path download test."


def test_upload_overwrite_existing_file(handler, base_dir):
    """
    Test uploading to a destination path where the file already exists.
    The existing file should be overwritten.
    """
    # Create source file
    source_file = base_dir / "source_overwrite.txt"
    source_file.write_text("Original content.")

    # Define destination path
    destination_path = "overwrite/destination.txt"

    # Perform initial upload
    handler.upload(str(source_file), destination_path)

    # Verify initial content
    dest_file = base_dir / destination_path
    assert dest_file.exists()
    assert dest_file.read_text() == "Original content."

    # Modify source file content
    source_file.write_text("Updated content.")

    # Perform upload again to overwrite
    handler.upload(str(source_file), destination_path)

    # Verify updated content
    assert dest_file.read_text() == "Updated content."


def test_download_overwrite_existing_file(handler, base_dir):
    """
    Test downloading to a destination path where the file already exists.
    The existing file should be overwritten.
    """
    # Create source file
    source_file = base_dir / "source_overwrite_download.txt"
    source_file.write_text("Original download content.")

    # Define destination path
    destination_path = "overwrite_download/destination_download.txt"

    # Perform initial download
    handler.download(str(source_file), destination_path)

    # Verify initial content
    dest_file = base_dir / destination_path
    assert dest_file.exists()
    assert dest_file.read_text() == "Original download content."

    # Modify source file content
    source_file.write_text("Updated download content.")

    # Perform download again to overwrite
    handler.download(str(source_file), destination_path)

    # Verify updated content
    assert dest_file.read_text() == "Updated download content."


def test_handler_with_empty_base_directory():
    """
    Test initializing LocalFileHandler without specifying a base directory,
    which should default to the current working directory.
    """
    # Capture the current working directory
    current_dir = Path.cwd()

    # Initialize handler without base_directory
    handler = LocalFileHandler()

    # The base_directory should be the current working directory
    assert Path(handler.base_directory) == current_dir


def test_upload_stream_success(tmp_path):
    """
    Test successful upload of data from an in-memory stream to the local filesystem.
    """
    handler = LocalFileHandler(base_directory=str(tmp_path))
    data = BytesIO(b"Test data for upload stream.")
    destination_path = "uploaded/test_stream.txt"

    handler.upload_stream(data, destination_path)

    # Verify the file exists and contains the correct data
    dest_file = tmp_path / "uploaded" / "test_stream.txt"
    assert dest_file.exists()
    assert dest_file.read_bytes() == b"Test data for upload stream."


def test_download_stream_success(tmp_path):
    """
    Test successful download of data into an in-memory stream from the local filesystem.
    """
    handler = LocalFileHandler(base_directory=str(tmp_path))
    source_file = tmp_path / "downloaded" / "test_stream_download.txt"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_bytes(b"Test data for download stream.")

    stream = handler.download_stream("downloaded/test_stream_download.txt")

    # Verify the stream contains the correct data
    assert stream.read() == b"Test data for download stream."
