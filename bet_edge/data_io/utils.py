from typing import Optional, List, Tuple
from bet_edge.data_io.storage.postgres_data import PostgresStorage


def execute_sql(storage: PostgresStorage, sql: str) -> Optional[List[Tuple]]:
    """
    Executes a SQL statement against the given PostgresStorage.

    If the statement is a SELECT query, the results are returned as a list of tuples.
    Otherwise, it executes the statement and returns None.

    Parameters:
        storage (PostgresStorage): The storage object connected to the database.
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
