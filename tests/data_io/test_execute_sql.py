# tests/test_postgres_sql.py

import pytest
from bet_edge.data_io.storage.postgres_data import PostgresStorage
from bet_edge.data_io.env_cred_provider import EnvCredProvider
from bet_edge.data_io.utils import execute_sql


@pytest.fixture
def postgres_storage():
    return PostgresStorage(credential_provider=EnvCredProvider())


def test_execute_sql_non_select(postgres_storage):
    # Create a temporary table
    execute_sql(postgres_storage, "DROP TABLE IF EXISTS test_table;")
    execute_sql(postgres_storage, "CREATE TABLE test_table (id INT, name TEXT);")

    # Insert data
    execute_sql(postgres_storage, "INSERT INTO test_table (id, name) VALUES (1, 'Alice');")
    execute_sql(postgres_storage, "INSERT INTO test_table (id, name) VALUES (2, 'Bob');")

    # Count rows to ensure insert worked
    results = execute_sql(postgres_storage, "SELECT COUNT(*) FROM test_table;")
    assert results is not None
    assert results[0][0] == 2


def test_execute_sql_select(postgres_storage):
    # Ensure table and data are present (from previous test or setup)
    # If tests run in isolation, run setup steps again or consider a pytest fixture that creates it.
    execute_sql(postgres_storage, "DROP TABLE IF EXISTS test_table;")
    execute_sql(postgres_storage, "CREATE TABLE test_table (id INT, name TEXT);")
    execute_sql(postgres_storage, "INSERT INTO test_table (id, name) VALUES (1, 'Alice');")
    execute_sql(postgres_storage, "INSERT INTO test_table (id, name) VALUES (2, 'Bob');")

    # Select data
    results = execute_sql(postgres_storage, "SELECT id, name FROM test_table ORDER BY id;")
    assert results == [(1, "Alice"), (2, "Bob")]


def test_execute_sql_with_csv_copy(postgres_storage, tmp_path):
    # We can also use the IDataStorage interface to verify integration with CSV using COPY

    # Create a table to test CSV loading
    execute_sql(postgres_storage, "DROP TABLE IF EXISTS csv_test;")
    execute_sql(postgres_storage, "CREATE TABLE csv_test (id INT, value TEXT);")

    # Create a local CSV file
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("id,value\n10,Hello\n20,World\n")

    # Use PostgresStorage directly to load CSV into the table
    postgres_storage.write_from_file(str(csv_path), "csv_test", format="csv")

    # Verify the data is loaded correctly
    results = execute_sql(postgres_storage, "SELECT id, value FROM csv_test ORDER BY id;")
    assert results == [(10, "Hello"), (20, "World")]


def test_execute_sql_insert_and_select(postgres_storage):
    # Create table
    execute_sql(postgres_storage, "DROP TABLE IF EXISTS sql_test;")
    execute_sql(postgres_storage, "CREATE TABLE sql_test (id INT, name TEXT);")

    # Insert data using execute_sql
    execute_sql(postgres_storage, "INSERT INTO sql_test (id, name) VALUES (9, 'Ivan');")
    execute_sql(postgres_storage, "INSERT INTO sql_test (id, name) VALUES (10, 'Judy');")

    # Select data
    results = execute_sql(postgres_storage, "SELECT id, name FROM sql_test ORDER BY id;")
    assert results == [(9, "Ivan"), (10, "Judy")]


def test_execute_sql_error_handling(postgres_storage):
    # Attempt to execute invalid SQL
    with pytest.raises(Exception):
        execute_sql(postgres_storage, "SELECT * FROM nonexistent_table;")
