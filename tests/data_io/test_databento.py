import pytest
from pathlib import Path

from bet_edge.data_io.databento import get_dbn_files, dbn_to_polars_dataframe, combine_dbn_files_to_polars

test_path = Path(r"C:\Users\Ashle\OneDrive\Documents\bet_edge\data")


def test_get_dbn_files():
    dbn_files = get_dbn_files(r"C:\Users\Ashle\OneDrive\Documents\bet_edge\data")
    assert len(dbn_files) == 1


def test_dbn_to_polars_dataframe():
    # Since the sample DBN file is empty, the function should handle it gracefully
    df = dbn_to_polars_dataframe(test_path)
    assert df is None  # Expecting None due to empty or invalid DBN file


def test_combine_dbn_files_to_polars():
    # Create another sample DBN file
    another_dbn = Path(tmp_dir) / "another_sample.dbn.zst"
    another_dbn.touch()

    combined_df = combine_dbn_files_to_polars(str(tmp_dir))
    assert combined_df.shape[0] == 0  # No data due to empty DBN files


def test_combine_dbn_files_no_files():
    with pytest.raises(FileNotFoundError):
        combine_dbn_files_to_polars(str(tmp_dir))
