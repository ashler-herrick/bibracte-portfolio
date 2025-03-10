from abc import ABC, abstractmethod
from io import BytesIO
from typing import Optional


class ICredProvider(ABC):
    """
    An interface for credential providers that fetch and supply credentials for
    accessing various services (e.g., AWS, databases, etc.).
    """

    @abstractmethod
    def get_credentials(self) -> dict:
        """
        Fetches credentials and returns them as a dictionary.

        Returns:
            dict: A dictionary containing credentials.
        """
        pass


class IDataStorage(ABC):
    """
    Abstract base class for defining data storage operations.

    This class provides a blueprint for implementing various data storage systems
    that support reading and writing data in different formats.
    """

    def __init__(self, credential_provider: Optional[ICredProvider] = None):
        """
        Initializes the storage class with an optional credential provider.

        Args:
            credential_provider (Optional[ICredProvider]): An optional credential provider
                for handling authentication and access control.
        """
        self.credential_provider = credential_provider

    @abstractmethod
    def write_from_file(self, source_file_path: str, destination_identifier: str, format: str = "binary") -> None:
        """
        Writes data from a local file to the storage.

        Args:
            source_file_path (str): The path of the local file to be written to storage.
            destination_identifier (str): The identifier (e.g., file path or key) for the destination in storage.
            format (str, optional): The format in which the data should be stored (e.g., "binary", "text").
                Defaults to "binary".
        """
        pass

    @abstractmethod
    def read_to_file(self, source_identifier: str, destination_file_path: str, format: str = "binary") -> None:
        """
        Reads data from the storage and writes it to a local file.

        Args:
            source_identifier (str): The identifier (e.g., file path or key) of the data in the storage.
            destination_file_path (str): The path of the local file where the data will be written.
            format (str, optional): The format in which the data should be read (e.g., "binary", "text").
                Defaults to "binary".
        """
        pass

    @abstractmethod
    def write(self, data: BytesIO, destination_identifier: str, format: str = "binary") -> None:
        """
        Writes data from a BytesIO object to the storage.

        Args:
            data (BytesIO): The data to be written to storage.
            destination_identifier (str): The identifier (e.g., file path or key) for the destination in storage.
            format (str, optional): The format in which the data should be stored (e.g., "binary", "text").
                Defaults to "binary".
        """
        pass

    @abstractmethod
    def read(self, source_identifier: str, format: str = "binary") -> BytesIO:
        """
        Reads data from the storage into a BytesIO object.

        Args:
            source_identifier (str): The identifier (e.g., file path or key) of the data in the storage.
            format (str, optional): The format in which the data should be read (e.g., "binary", "text").
                Defaults to "binary".

        Returns:
            BytesIO: The data read from the storage as a BytesIO object.
        """
        pass
