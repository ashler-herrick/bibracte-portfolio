import shutil
from io import BytesIO
from typing import Optional
from ..interfaces import IDataStorage, ICredProvider


class LocalStorage(IDataStorage):
    """
    Implementation of the IDataStorage interface for local file system storage.

    This class provides methods to read and write data to and from the local file system.
    """

    def __init__(self, credential_provider: Optional[ICredProvider] = None):
        """
        Initializes the local storage with an optional credential provider.

        Args:
            credential_provider (Optional[ICredProvider]): An optional credential provider
                for handling authentication (not typically used in local storage).
        """
        super().__init__(credential_provider=credential_provider)

    def write_from_file(self, source_file_path: str, destination_identifier: str, format: str = "binary") -> None:
        """
        Writes data from a local file to another location in the local file system.

        Args:
            source_file_path (str): The path of the source file on the local file system.
            destination_identifier (str): The path of the destination file in the local file system.
            format (str, optional): The format of the data (e.g., "binary", "text").
                Defaults to "binary".
        """
        shutil.copyfile(source_file_path, destination_identifier)

    def read_to_file(self, source_identifier: str, destination_file_path: str, format: str = "binary") -> None:
        """
        Reads data from a local file and writes it to another file in the local file system.

        Args:
            source_identifier (str): The path of the source file in the local file system.
            destination_file_path (str): The path of the destination file on the local file system.
            format (str, optional): The format of the data (e.g., "binary", "text").
                Defaults to "binary".
        """
        shutil.copyfile(source_identifier, destination_file_path)

    def write(self, data: BytesIO, destination_identifier: str, format: str = "binary") -> None:
        """
        Writes data from a BytesIO object to a file in the local file system.

        Args:
            data (BytesIO): The data to be written.
            destination_identifier (str): The path of the destination file in the local file system.
            format (str, optional): The format of the data (e.g., "binary", "text").
                Defaults to "binary".
        """
        with open(destination_identifier, "wb") as f:
            f.write(data.getvalue())

    def read(self, source_identifier: str, format: str = "binary") -> BytesIO:
        """
        Reads data from a local file into a BytesIO object.

        Args:
            source_identifier (str): The path of the source file in the local file system.
            format (str, optional): The format of the data (e.g., "binary", "text").
                Defaults to "binary".

        Returns:
            BytesIO: The data read from the file as a BytesIO object.
        """
        with open(source_identifier, "rb") as f:
            return BytesIO(f.read())
