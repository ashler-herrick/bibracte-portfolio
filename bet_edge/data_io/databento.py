from typing import List, Optional
from pathlib import Path

import polars as pl
from databento import DBNStore


def get_dbn_files(directory: str, extension: str = ".dbn.zst") -> List[Path]:
    """
    Traverse the specified directory and return a list of all DBN files with the given extension.

    Args:
        directory (str): Path to the directory to search for DBN files.
        extension (str, optional): File extension to filter DBN files. Defaults to ".dbn.zst".

    Returns:
        List[Path]: List of Path objects pointing to DBN files.
    """
    path = Path(directory)
    if not path.is_dir():
        raise ValueError(f"The provided path '{directory}' is not a directory or does not exist.")

    dbn_files = list(path.rglob(f"*{extension}"))
    if not dbn_files:
        print(f"No DBN files with extension '{extension}' found in directory '{directory}'.")
    return dbn_files


def dbn_to_polars_dataframe(dbn_file: Path) -> Optional[pl.DataFrame]:
    """
    Convert a single DBN file to a Polars DataFrame.

    Args:
        dbn_file (Path): Path object pointing to the DBN file.

    Returns:
        Optional[pl.DataFrame]: Polars DataFrame if conversion is successful, else None.
    """
    try:
        # Initialize DBNStore with the DBN file
        dbn_store = DBNStore.from_file(str(dbn_file))

        # Convert DBN data to Pandas DataFrame
        pandas_df = dbn_store.to_df()
        # Convert Pandas DataFrame to Polars DataFrame
        polars_df = pl.from_pandas(pandas_df, include_index=True)

        return polars_df

    except Exception as e:
        print(f"Failed to process '{dbn_file}': {e}")
        return None


def combine_dbn_files_to_polars(directory: str, extension: str = ".dbn.zst") -> pl.DataFrame:
    """
    Read all DBN files from the specified directory and combine them into a single Polars DataFrame.

    Args:
        directory (str): Path to the directory containing DBN files.
        extension (str, optional): File extension to filter DBN files. Defaults to ".dbn.zst".

    Returns:
        pl.DataFrame: Combined Polars DataFrame containing data from all DBN files.
    """
    dbn_files = get_dbn_files(directory, extension)
    if not dbn_files:
        raise FileNotFoundError(f"No DBN files with extension '{extension}' found in '{directory}'.")

    polars_dfs = []
    for dbn_file in dbn_files:
        df = dbn_to_polars_dataframe(dbn_file)
        if df is not None:
            polars_dfs.append(df)

    if not polars_dfs:
        raise ValueError("No valid DBN files were processed successfully.")

    # Concatenate all Polars DataFrames
    combined_df = pl.concat(polars_dfs, rechunk=True)

    return combined_df
