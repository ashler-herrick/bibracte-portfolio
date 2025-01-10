import pytest
import os
from io import BytesIO
from tempfile import TemporaryDirectory

# Import storage backends and utilities
from bet_edge.data_io.storage.local_data import LocalStorage
from bet_edge.data_io.storage.s3_data import S3Storage
from bet_edge.data_io.storage.postgres_data import PostgresStorage
from bet_edge.data_io.env_cred_provider import EnvCredProvider
from bet_edge.data_io.utils import execute_sql

import polars as pl

# -------------------------------
# Fixtures for Storage Backends
# -------------------------------

@pytest.fixture(scope="session")
def tmp_dir():
    with TemporaryDirectory() as tmp:
        yield tmp


@pytest.fixture
def s3_storage():
    
    return S3Storage(credential_provider=EnvCredProvider())


@pytest.fixture
def postgres_storage():

    storage = PostgresStorage(credential_provider=EnvCredProvider())
    yield storage
    # Cleanup: Drop tables if necessary
    execute_sql(storage, "DROP TABLE IF EXISTS test_table;")
    execute_sql(storage, "DROP TABLE IF EXISTS csv_test;")

@pytest.fixture
def local_storage(tmp_dir):
    return LocalStorage()

# -------------------------------
# Helper Functions
# -------------------------------

def create_sample_csv(file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("id,name\n1,Alice\n2,Bob\n")

def create_sample_parquet(file_path):
    df = pl.DataFrame({
        "id": [1, 2],
        "name": ["Alice", "Bob"]
    })
    df.write_parquet(file_path)

# -------------------------------
# Tests for LocalStorage
# -------------------------------

def test_local_write_read_binary(local_storage, tmp_dir):
    source_content = b"Hello, Local Storage!"
    source_path = os.path.join(tmp_dir, "source.bin")
    dest_path = os.path.join(tmp_dir, "dest.bin")
    
    # Write binary data
    with open(source_path, 'wb') as f:
        f.write(source_content)
    
    local_storage.write_from_file(source_path, dest_path)
    
    # Read back the data
    data = local_storage.read(dest_path)
    assert data.getvalue() == source_content

def test_local_write_read_csv(local_storage, tmp_dir):
    source_path = os.path.join(tmp_dir, "data.csv")
    dest_path = os.path.join(tmp_dir, "dest.csv")
    create_sample_csv(source_path)
    
    # Write CSV data
    local_storage.write_from_file(source_path, dest_path, format="csv")
    
    # Read back the data
    data = local_storage.read(dest_path, format="csv")
    df = pl.read_csv(data)
    assert df.to_dict(as_series=False) == {"id": [1, 2], "name": ["Alice", "Bob"]}

# -------------------------------
# Tests for S3Storage
# -------------------------------

def test_s3_write_read_binary(s3_storage):
    bucket = "bet-edge"
    key = "folder/test.bin"
    content = b"Hello, S3 Storage!"
    
    # Write binary data
    data = BytesIO(content)
    s3_storage.write(data, f"s3://{bucket}/{key}", format="binary")
    
    # Read back the data
    read_data = s3_storage.read(f"s3://{bucket}/{key}", format="binary")
    assert read_data.getvalue() == content

def test_s3_write_read_csv(s3_storage):
    bucket = "bet-edge"
    key = "folder/data.csv"
    csv_content = "id,name\n1,Alice\n2,Bob\n"
    
    # Write CSV data
    data = BytesIO(csv_content.encode('utf-8'))
    s3_storage.write(data, f"s3://{bucket}/{key}", format="csv")
    
    # Read back the data
    read_data = s3_storage.read(f"s3://{bucket}/{key}", format="csv")
    df = pl.read_csv(read_data)
    assert df.to_dict(as_series=False) == {"id": [1, 2], "name": ["Alice", "Bob"]}

# -------------------------------
# Tests for PostgresStorage
# -------------------------------

def test_postgres_write_read_binary(postgres_storage, tmp_dir):
    # PostgresStorage does not support binary formats directly; it's limited to CSV.
    # Therefore, this test will be skipped or expected to raise an error.
    source_content = b"binary data"
    data = BytesIO(source_content)
    with pytest.raises(ValueError):
        postgres_storage.write(data, "binary_test", format="binary")

def test_postgres_write_read_csv(postgres_storage, tmp_dir):
    # Create a CSV file
    csv_path = os.path.join(tmp_dir, "data.csv")
    create_sample_csv(csv_path)
    
    # Create a table
    execute_sql(postgres_storage, "DROP TABLE IF EXISTS csv_test;")
    execute_sql(postgres_storage, "CREATE TABLE csv_test (id INT, name TEXT);")
    
    # Write CSV to PostgreSQL
    postgres_storage.write_from_file(csv_path, "csv_test", format="csv")
    
    # Read back the data
    read_data = postgres_storage.read("csv_test", format="csv")
    df = pl.read_csv(read_data)
    assert df.to_dict(as_series=False) == {"id": [1, 2], "name": ["Alice", "Bob"]}

def test_postgres_execute_sql(postgres_storage):
    # Create a table
    execute_sql(postgres_storage, "DROP TABLE IF EXISTS test_table;")
    execute_sql(postgres_storage, "CREATE TABLE test_table (id INT, name TEXT);")
    
    # Insert data
    execute_sql(postgres_storage, "INSERT INTO test_table (id, name) VALUES (1, 'Alice');")
    execute_sql(postgres_storage, "INSERT INTO test_table (id, name) VALUES (2, 'Bob');")
    
    # Select data
    results = execute_sql(postgres_storage, "SELECT id, name FROM test_table ORDER BY id;")
    assert results == [(1, 'Alice'), (2, 'Bob')]

def test_postgres_write_read_via_utils(postgres_storage, tmp_dir):
    # Create a CSV file
    csv_path = os.path.join(tmp_dir, "utils_data.csv")
    create_sample_csv(csv_path)
    
    # Create a table
    execute_sql(postgres_storage, "DROP TABLE IF EXISTS utils_test;")
    execute_sql(postgres_storage, "CREATE TABLE utils_test (id INT, name TEXT);")
    
    # Write CSV to PostgreSQL using IDataStorage
    postgres_storage.write_from_file(csv_path, "utils_test", format="csv")
    
    # Read back the data using IDataStorage
    read_data = postgres_storage.read("utils_test", format="csv")
    df = pl.read_csv(read_data)
    assert df.to_dict(as_series=False) == {"id": [1, 2], "name": ["Alice", "Bob"]}




# -------------------------------
# Tests for Error Handling
# -------------------------------

def test_unsupported_format_postgres(postgres_storage, tmp_dir):
    csv_path = os.path.join(tmp_dir, "data.csv")
    create_sample_csv(csv_path)
    
    # Attempt to write with unsupported format
    with pytest.raises(ValueError, match="currently only supports CSV format"):
        postgres_storage.write_from_file(csv_path, "csv_test", format="parquet")

def test_invalid_identifier_s3(s3_storage):
    # Attempt to read from a non-existent S3 key
    with pytest.raises(Exception):
        s3_storage.read("s3://bet-edge/nonexistent.bin", format="binary")

def test_invalid_identifier_local(local_storage, tmp_dir):
    # Attempt to read from a non-existent local file
    with pytest.raises(FileNotFoundError):
        local_storage.read(os.path.join(tmp_dir, "nonexistent.bin"), format="binary")




