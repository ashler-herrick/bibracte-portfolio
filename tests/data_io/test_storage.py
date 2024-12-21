import pytest
import os
from io import BytesIO
from tempfile import TemporaryDirectory

# Import storage backends and utilities
from bet_edge.data_io.storage.local_data import LocalDataStorage
from bet_edge.data_io.storage.s3_data import S3DataStorage
from bet_edge.data_io.storage.postgres_data import PostgreSQLDataStorage
from bet_edge.data_io.storage.polars_data import PolarsDataFrameStorage
from bet_edge.data_io.env_credential_provider import EnvironmentCredentialProvider
from bet_edge.data_io.utils import transfer_data, execute_sql


import polars as pl
from testcontainers.postgres import PostgresContainer

# -------------------------------
# Fixtures for Storage Backends
# -------------------------------

@pytest.fixture(scope="session")
def tmp_dir():
    with TemporaryDirectory() as tmp:
        yield tmp


@pytest.fixture
def s3_storage():
    
    return S3DataStorage(credential_provider=EnvironmentCredentialProvider())


@pytest.fixture
def postgres_storage():

    storage = PostgreSQLDataStorage(credential_provider=EnvironmentCredentialProvider())
    yield storage
    # Cleanup: Drop tables if necessary
    execute_sql(storage, "DROP TABLE IF EXISTS test_table;")
    execute_sql(storage, "DROP TABLE IF EXISTS csv_test;")

@pytest.fixture
def local_storage(tmp_dir):
    return LocalDataStorage()

@pytest.fixture
def polars_storage():
    return PolarsDataFrameStorage()

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
# Tests for LocalDataStorage
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
# Tests for S3DataStorage
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
# Tests for PostgreSQLDataStorage
# -------------------------------

def test_postgres_write_read_binary(postgres_storage, tmp_dir):
    # PostgreSQLDataStorage does not support binary formats directly; it's limited to CSV.
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
# Tests for PolarsDataFrameStorage
# -------------------------------

def test_polars_write_read_csv(polars_storage, tmp_dir):
    # Write CSV data
    csv_path = os.path.join(tmp_dir, "data.csv")
    create_sample_csv(csv_path)
    polars_storage.write_from_file(csv_path, "df_csv", format="csv")
    
    # Read back to a new CSV file
    output_path = os.path.join(tmp_dir, "output.csv")
    polars_storage.read_to_file("df_csv", output_path, format="csv")
    
    # Verify the content
    with open(output_path, 'r', encoding='utf-8') as f:
        content = f.read()
    assert content == "id,name\n1,Alice\n2,Bob\n"

def test_polars_write_read_parquet(polars_storage, tmp_dir):
    # Write Parquet data
    parquet_path = os.path.join(tmp_dir, "data.parquet")
    create_sample_parquet(parquet_path)
    polars_storage.write_from_file(parquet_path, "df_parquet", format="parquet")
    
    # Read back to a new Parquet file
    output_path = os.path.join(tmp_dir, "output.parquet")
    polars_storage.read_to_file("df_parquet", output_path, format="parquet")
    
    # Verify the content
    df = pl.read_parquet(output_path)
    assert df.to_dict(as_series=False) == {"id": [1, 2], "name": ["Alice", "Bob"]}

def test_polars_write_read_bytes_io(polars_storage):
    # Create DataFrame and write to BytesIO
    df = pl.DataFrame({
        "id": [1, 2],
        "name": ["Alice", "Bob"]
    })
    csv_bytes = BytesIO()
    df.write_csv(csv_bytes)
    
    # Write BytesIO to Polars storage
    polars_storage.write(csv_bytes, "df_io", format="csv")
    
    # Read back as BytesIO
    read_bytes = polars_storage.read("df_io", format="csv")
    df_read = pl.read_csv(read_bytes)
    
    assert df_read.to_dict(as_series=False) == {"id": [1, 2], "name": ["Alice", "Bob"]}

# -------------------------------
# Tests for Data Transfer
# -------------------------------

def test_transfer_local_to_s3(local_storage, s3_storage, tmp_dir):
    # Create a local file
    source_path = os.path.join(tmp_dir, "transfer.bin")
    content = b"Transfer this to S3."
    with open(source_path, 'wb') as f:
        f.write(content)
    
    destination_identifier = "s3://bet-edge/transfer.bin"
    
    # Transfer from Local to S3
    transfer_data(local_storage, source_path, s3_storage, destination_identifier, format="binary")
    
    # Read back from S3
    read_data = s3_storage.read(destination_identifier, format="binary")
    assert read_data.getvalue() == content

def test_transfer_s3_to_polars(s3_storage, polars_storage, tmp_dir):
    # Create CSV data in S3
    bucket = "bet-edge"
    key = "transfer_data.csv"
    csv_content = "id,name\n3,Charlie\n4,David\n"
    s3_storage.write(BytesIO(csv_content.encode('utf-8')), f"s3://{bucket}/{key}", format="csv")
    
    # Transfer from S3 to Polars
    destination_identifier = "df_transfer"
    transfer_data(s3_storage, f"s3://{bucket}/{key}", polars_storage, destination_identifier, format="csv")
    
    # Read back from Polars and verify
    read_data = polars_storage.read(destination_identifier, format="csv")
    df = pl.read_csv(read_data)
    assert df.to_dict(as_series=False) == {"id": [3, 4], "name": ["Charlie", "David"]}

def test_transfer_postgres_to_local(postgres_storage, local_storage, tmp_dir):
    # Setup PostgreSQL table
    execute_sql(postgres_storage, "DROP TABLE IF EXISTS transfer_test;")
    execute_sql(postgres_storage, "CREATE TABLE transfer_test (id INT, name TEXT);")
    execute_sql(postgres_storage, "INSERT INTO transfer_test (id, name) VALUES (5, 'Eve'), (6, 'Frank');")
    
    # Destination file path
    dest_path = os.path.join(tmp_dir, "postgres_data.csv")
    
    # Transfer from PostgreSQL to Local
    transfer_data(postgres_storage, "transfer_test", local_storage, dest_path, format="csv")
    
    # Verify the file content
    with open(dest_path, 'r', encoding='utf-8') as f:
        content = f.read()
    assert content == "id,name\n5,Eve\n6,Frank\n"

def test_transfer_polars_to_postgres(polars_storage, postgres_storage, tmp_dir):
    # Create and store DataFrame in Polars
    df = pl.DataFrame({
        "id": [7, 8],
        "name": ["Grace", "Heidi"]
    })
    csv_bytes = BytesIO()
    df.write_csv(csv_bytes)
    polars_storage.write(csv_bytes, "df_postgres", format="csv")
    
    # Create PostgreSQL table
    execute_sql(postgres_storage, "DROP TABLE IF EXISTS transfer_pg;")
    execute_sql(postgres_storage, "CREATE TABLE transfer_pg (id INT, name TEXT);")
    
    # Transfer from Polars to PostgreSQL
    transfer_data(polars_storage, "df_postgres", postgres_storage, "transfer_pg", format="csv")
    
    # Verify data in PostgreSQL
    results = execute_sql(postgres_storage, "SELECT id, name FROM transfer_pg ORDER BY id;")
    assert results == [(7, 'Grace'), (8, 'Heidi')]

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

def test_invalid_format_polars(polars_storage, tmp_dir):
    # Attempt to write with unsupported format
    csv_path = os.path.join(tmp_dir, "data.csv")
    create_sample_csv(csv_path)
    
    with pytest.raises(ValueError, match="Format xml not supported"):
        polars_storage.write_from_file(csv_path, "df_invalid", format="xml")




