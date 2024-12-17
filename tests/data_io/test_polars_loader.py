import polars as pl

from bet_edge.data_io.polars_loader import PolarsLoader
from bet_edge.data_io.s3_file_handler import S3FileHandler
from bet_edge.data_io.env_credential_provider import EnvironmentCredentialProvider

S3_KEY = "data/sample.parquet"
S3_BUCKET = "bet-edge"
credential_manager = EnvironmentCredentialProvider()
file_handler = S3FileHandler(S3_BUCKET, credential_manager)
polars_loader = PolarsLoader(credential_manager, file_handler)


def test_upload():
    sample_df = pl.DataFrame({"name": ["alice", "bob", "charlie"], "age": [25, 30, 35]})
    polars_loader.upload_polars_df(sample_df, S3_KEY)


def test_download():
    df = polars_loader.download_to_polars_df(S3_KEY)
    assert not df.is_empty()
