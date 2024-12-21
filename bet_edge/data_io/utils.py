from .interfaces import IDataStorage
from typing import Optional, List, Tuple
from bet_edge.data_io.storage.postgres_data import PostgreSQLDataStorage


def transfer_data(
    source_storage: IDataStorage, 
    source_identifier: str, 
    destination_storage: IDataStorage, 
    destination_identifier: str, 
    format: str = "binary"
):
    """
    Transfers data from a source storage to a destination storage.

    Args:
        source_storage (IDataStorage): The storage object from which data will be read.
        source_identifier (str): The identifier (e.g., file path or key) of the data in the source storage.
        destination_storage (IDataStorage): The storage object to which data will be written.
        destination_identifier (str): The identifier (e.g., file path or key) for the data in the destination storage.
        format (str, optional): The format in which the data is transferred (e.g., "binary", "text"). Defaults to "binary".

    Returns:
        None
    """
    data = source_storage.read(source_identifier, format=format)
    destination_storage.write(data, destination_identifier, format=format)



def execute_sql(storage: PostgreSQLDataStorage, sql: str) -> Optional[List[Tuple]]:
    """
    Executes a SQL statement against the given PostgreSQLDataStorage.

    If the statement is a SELECT query, the results are returned as a list of tuples.
    Otherwise, it executes the statement and returns None.

    Parameters:
        storage (PostgreSQLDataStorage): The storage object connected to the database.
        sql (str): The SQL statement to execute.

    Returns:
        Optional[List[Tuple]]: A list of result rows if it's a SELECT query, otherwise None.
    """
    # Simple heuristic: if it starts with 'select', treat it as a query returning results
    is_select = sql.strip().lower().startswith("select")
    with storage.conn.cursor() as cur:
        cur.execute(sql)
        if is_select:
            return cur.fetchall()
        # For non-select queries, no return data
        return None