import polars as pl
from io import BytesIO
from typing import Optional, Dict
from ..interfaces import IDataStorage, ICredentialProvider

class PolarsDataFrameStorage(IDataStorage):
    """
    Implementation of the IDataStorage interface for managing Polars DataFrames in memory.

    This class allows for storing, retrieving, and transferring Polars DataFrames using
    file-like operations and in-memory storage.
    """

    def __init__(self, credential_provider: Optional[ICredentialProvider] = None):
        """
        Initializes the storage with an optional credential provider.

        Args:
            credential_provider (Optional[ICredentialProvider]): An optional credential provider
                for handling authentication (not typically used for this implementation).
        """
        super().__init__(credential_provider=credential_provider)
        self._dfs: Dict[str, pl.DataFrame] = {}

    def _read_df_from_file(self, path: str, format: str) -> pl.DataFrame:
        """
        Reads a Polars DataFrame from a file.

        Args:
            path (str): The path to the source file.
            format (str): The format of the file ("csv" or "parquet").

        Returns:
            pl.DataFrame: The DataFrame read from the file.

        Raises:
            ValueError: If the specified format is not supported.
        """
        if format == "csv":
            return pl.read_csv(path)
        elif format == "parquet":
            return pl.read_parquet(path)
        else:
            raise ValueError(f"Format {format} not supported by Polars storage.")

    def _write_df_to_file(self, df: pl.DataFrame, path: str, format: str):
        """
        Writes a Polars DataFrame to a file.

        Args:
            df (pl.DataFrame): The DataFrame to write.
            path (str): The destination file path.
            format (str): The format to write the file ("csv" or "parquet").

        Raises:
            ValueError: If the specified format is not supported.
        """
        if format == "csv":
            df.write_csv(path)
        elif format == "parquet":
            df.write_parquet(path)
        else:
            raise ValueError(f"Format {format} not supported by Polars storage.")

    def _read_df_from_bytes(self, data: BytesIO, format: str) -> pl.DataFrame:
        """
        Reads a Polars DataFrame from a BytesIO object.

        Args:
            data (BytesIO): The source data in memory.
            format (str): The format of the data ("csv" or "parquet").

        Returns:
            pl.DataFrame: The DataFrame read from the BytesIO object.

        Raises:
            ValueError: If the specified format is not supported.
        """
        data.seek(0)
        if format == "csv":
            return pl.read_csv(data)
        elif format == "parquet":
            return pl.read_parquet(data)
        else:
            raise ValueError(f"Format {format} not supported by Polars storage.")

    def _write_df_to_bytes(self, df: pl.DataFrame, format: str) -> BytesIO:
        """
        Writes a Polars DataFrame to a BytesIO object.

        Args:
            df (pl.DataFrame): The DataFrame to write.
            format (str): The format to write the data ("csv" or "parquet").

        Returns:
            BytesIO: A BytesIO object containing the written data.

        Raises:
            ValueError: If the specified format is not supported.
        """
        buf = BytesIO()
        if format == "csv":
            df.write_csv(buf)
        elif format == "parquet":
            df.write_parquet(buf)
        else:
            raise ValueError(f"Format {format} not supported by Polars storage.")
        buf.seek(0)
        return buf

    def write_from_file(self, source_file_path: str, destination_identifier: str, format: str = "csv") -> None:
        """
        Writes a DataFrame to in-memory storage from a file.

        Args:
            source_file_path (str): The path to the source file.
            destination_identifier (str): The identifier for the in-memory DataFrame.
            format (str, optional): The format of the file ("csv" or "parquet"). Defaults to "csv".
        """
        df = self._read_df_from_file(source_file_path, format)
        self._dfs[destination_identifier] = df

    def read_to_file(self, source_identifier: str, destination_file_path: str, format: str = "csv") -> None:
        """
        Reads a DataFrame from in-memory storage and writes it to a file.

        Args:
            source_identifier (str): The identifier for the in-memory DataFrame.
            destination_file_path (str): The path to the destination file.
            format (str, optional): The format of the file ("csv" or "parquet"). Defaults to "csv".

        Raises:
            ValueError: If the identifier does not exist in in-memory storage.
        """
        df = self._dfs.get(source_identifier)
        if df is None:
            raise ValueError(f"No DataFrame found for {source_identifier}")
        self._write_df_to_file(df, destination_file_path, format)

    def write(self, data: BytesIO, destination_identifier: str, format: str = "csv") -> None:
        """
        Writes a DataFrame to in-memory storage from a BytesIO object.

        Args:
            data (BytesIO): The data in memory to write.
            destination_identifier (str): The identifier for the in-memory DataFrame.
            format (str, optional): The format of the data ("csv" or "parquet"). Defaults to "csv".
        """
        df = self._read_df_from_bytes(data, format)
        self._dfs[destination_identifier] = df

    def read(self, source_identifier: str, format: str = "csv") -> BytesIO:
        """
        Reads a DataFrame from in-memory storage into a BytesIO object.

        Args:
            source_identifier (str): The identifier for the in-memory DataFrame.
            format (str, optional): The format of the data ("csv" or "parquet"). Defaults to "csv".

        Returns:
            BytesIO: A BytesIO object containing the DataFrame data.

        Raises:
            ValueError: If the identifier does not exist in in-memory storage.
        """
        df = self._dfs.get(source_identifier)
        if df is None:
            raise ValueError(f"No DataFrame found for {source_identifier}")
        return self._write_df_to_bytes(df, format)
