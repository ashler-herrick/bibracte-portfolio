import psycopg2
from io import BytesIO
from typing import Optional
from ..interfaces import IDataStorage, ICredProvider

class PostgresStorage(IDataStorage):
    """
    Implementation of the IDataStorage interface for PostgreSQL.

    This class provides methods for storing and retrieving data from a PostgreSQL database
    using the `COPY` command with CSV format.

    Attributes:
        conn (psycopg2.connection): The PostgreSQL database connection.
    """

    def __init__(self, credential_provider: Optional[ICredProvider] = None):
        """
        Initializes the PostgreSQL storage with an optional credential provider.

        Args:
            credential_provider (Optional[ICredProvider]): A credential provider for handling authentication.

        Raises:
            psycopg2.OperationalError: If the connection to the PostgreSQL database fails.
        """
        super().__init__(credential_provider=credential_provider)
        creds = self.credential_provider.get_credentials() if self.credential_provider else {}
        self.conn = psycopg2.connect(
            host=creds.get('postgres_host'),
            port=creds.get('postgres_port'),
            dbname=creds.get('postgres_db'),
            user=creds.get('postgres_user'),
            password=creds.get('postgres_password')
        )
        self.conn.autocommit = True

    def _ensure_csv(self, format: str):
        """
        Ensures that the format is CSV.

        Args:
            format (str): The format to validate.

        Raises:
            ValueError: If the format is not "csv".
        """
        if format != "csv":
            raise ValueError("PostgresStorage currently only supports CSV format.")

    def write_from_file(self, source_file_path: str, destination_identifier: str, format: str = "csv") -> None:
        """
        Writes data from a local CSV file to the specified PostgreSQL table using client-side COPY.

        Args:
            source_file_path (str): The path to the local CSV file.
            destination_identifier (str): The name of the PostgreSQL table where data will be written.
            format (str, optional): The format of the file (must be "csv"). Defaults to "csv".

        Raises:
            ValueError: If the format is not "csv".
        """
        self._ensure_csv(format)
        with self.conn.cursor() as cur:
            with open(source_file_path, 'r', encoding='utf-8') as f:
                cur.copy_expert(f"COPY {destination_identifier} FROM STDIN WITH CSV HEADER", f)

    def read_to_file(self, source_identifier: str, destination_file_path: str, format: str = "csv") -> None:
        """
        Reads data from the specified PostgreSQL table and writes it to a local CSV file using client-side COPY.

        Args:
            source_identifier (str): The name of the PostgreSQL table to read data from.
            destination_file_path (str): The path to the local CSV file where data will be written.
            format (str, optional): The format of the file (must be "csv"). Defaults to "csv".

        Raises:
            ValueError: If the format is not "csv".
        """
        self._ensure_csv(format)
        with self.conn.cursor() as cur:
            with open(destination_file_path, 'w', encoding='utf-8') as f:
                cur.copy_expert(f"COPY {source_identifier} TO STDOUT WITH CSV HEADER", f)

    def write(self, data: BytesIO, destination_identifier: str, format: str = "csv") -> None:
        """
        Writes data from a BytesIO object to the specified PostgreSQL table using client-side COPY.

        Args:
            data (BytesIO): The in-memory data to write.
            destination_identifier (str): The name of the PostgreSQL table where data will be written.
            format (str, optional): The format of the data (must be "csv"). Defaults to "csv".

        Raises:
            ValueError: If the format is not "csv".
        """
        self._ensure_csv(format)
        with self.conn.cursor() as cur:
            data.seek(0)
            cur.copy_expert(f"COPY {destination_identifier} FROM STDIN CSV HEADER", data) #type: ignore

    def read(self, source_identifier: str, format: str = "csv") -> BytesIO:
        """
        Reads data from the specified PostgreSQL table into a BytesIO object using client-side COPY.

        Args:
            source_identifier (str): The name of the PostgreSQL table to read data from.
            format (str, optional): The format of the data (must be "csv"). Defaults to "csv".

        Returns:
            BytesIO: A BytesIO object containing the data in CSV format.

        Raises:
            ValueError: If the format is not "csv".
        """
        self._ensure_csv(format)
        buf = BytesIO()
        with self.conn.cursor() as cur:
            cur.copy_expert(f"COPY {source_identifier} TO STDOUT CSV HEADER", buf) #type: ignore
        buf.seek(0)
        return buf
