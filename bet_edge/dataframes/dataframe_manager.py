"""
This module provides functionality around the primary key, dimensions and measures of a Polars DataFrame.
"""

import logging
from typing import List

import polars as pl
from polars._typing import UniqueKeepStrategy

from bet_edge.dataframes.column_helpers import intersection, sym_diff


logger = logging.getLogger(__name__)


class DataFrameManager:
    """
    Manages a Polars DataFrame with functionality to handle primary keys, deduplication,
    and other DataFrame operations.

    Attributes:
        primary_key (List[str]): A list of column names that constitute the primary key.
        dimensions (List[str]): A list of column names that represent the dimensions present in the DataFrame.
        measures (List[str]): A list of column names that represent the measures present in the DataFrame.
    """

    def __init__(
        self, dataframe: pl.DataFrame, primary_key: List[str], dimensions: List[str], measures: List[str] = []
    ):
        """
        Initializes the DataFrameManager with a DataFrame and its primary key. 

        Both a primary key list and list of dimensions are required because a table can have multiple sets of 
        columns constituting a primary key but only one list of dimensions. They will often be the same, but 
        sometimes they won't.

        Args:
            dataframe (pl.DataFrame):
                The Polars DataFrame to be managed.
            primary_key (List[str]):
                A list of column names that form the primary key for the DataFrame.
            dimensions (List[str]):
                A list of column names that represent the dimensions present in the DataFrame.
            measures (List[str]):
                A list of column names that represent the measures present in the DataFrame.

        Raises:
            ValueError:
                If the primary_key list is empty.
            ValueError:
                If the dimensions list is empty.
        """
        if not primary_key:
            raise ValueError("Primary key list cannot be empty.")
        if not dimensions:
            raise ValueError("Dimensions must be provided.")
        self._dataframe = dataframe
        self.primary_key = primary_key
        self.dimensions = dimensions
        self.measures = measures
        if not measures:
            self.measures = sym_diff(dataframe.columns, dimensions)

    @property
    def dataframe(self) -> pl.DataFrame:
        """
        Provides access to the underlying DataFrame.

        Returns:
            pl.DataFrame: The managed Polars DataFrame.
        """
        return self._dataframe

    def assert_valid_pk(self) -> None:
        """
        Validates that the specified columns constitute a unique primary key in the DataFrame.

        Raises:
            ValueError:
                If the combination of specified columns does not uniquely identify each row.
        """
        if not self._is_pk():
            logger.error(f"DataFrame does not have unique primary key columns: {self.primary_key}.")
            duplicates = self._get_non_pk_rows()
            logger.error(f"Duplicate rows in DataFrame:\n{duplicates}")
            raise AssertionError("DataFrame does not have unique primary key columns.")

    def dedupe_on_pk(self, keep: UniqueKeepStrategy = "first") -> "DataFrameManager":
        """
        Removes duplicate rows in the DataFrame based on the primary key columns.

        Args:
            keep (UniqueKeepStrategy, optional):
                Specifies which duplicate to keep. Options are:
                - 'first': Keeps the first occurrence (default).
                - 'last': Keeps the last occurrence.
                - 'any': Arbitrary selection.
                - 'none': Drops all duplicates.

        Returns:
            DataFrameManager: A new DataFrameManager with duplicates removed.
        """
        valid_options = {"first", "last", "any", "none"}
        if keep not in valid_options:
            raise ValueError(f"Invalid value for 'keep': {keep}. Must be one of {valid_options}.")

        if not self._is_pk():
            duplicates = self._get_non_pk_rows()
            logger.info(f"Found {duplicates.height} duplicate rows based on primary key {self.primary_key}.")
            logger.info(f"Deduplicating DataFrame on primary key {self.primary_key} by keeping '{keep}' entries...")
            df = self._dataframe.unique(subset=self.primary_key, keep=keep)
            dropped = self._dataframe.height - df.height
            logger.info(f"Successfully dropped {dropped} duplicate rows by deduping on primary key {self.primary_key}.")
            return DataFrameManager(df, self.primary_key, self.dimensions)
        else:
            logger.info(f"No duplicates found based on primary key {self.primary_key}. No action taken.")
            return self

    def dedupe_if_not_pk(self) -> "DataFrameManager":
        """
        Ensures the DataFrame has a valid primary key by deduplicating if necessary.

        Returns:
            DataFrameManager: A new DataFrameManager with duplicates removed if necessary.

        Raises:
            ValueError:
                If deduplication fails to enforce a unique primary key.
        """
        try:
            self.assert_valid_pk()
            return self
        except AssertionError as e:
            logger.warning(f"Primary key validation failed: {e}")
            logger.info(f"Deduplicating DataFrame to enforce primary key {self.primary_key}.")
            deduped_manager = self.dedupe_on_pk(keep="first")
            # Re-validate after deduplication
            deduped_manager.assert_valid_pk()
            return deduped_manager

    def _is_pk(self) -> bool:
        """
        Determines whether the specified columns form a unique primary key in the DataFrame.

        Returns:
            bool:
                True if the combination of primary key columns uniquely identifies each row,
                False otherwise.
        """
        self.validate_columns_exist(self.primary_key)

        unique_rows = self._dataframe.unique(subset=self.primary_key).height
        total_rows = self._dataframe.height
        is_unique = unique_rows == total_rows
        if not is_unique:
            logger.warning(f"Duplicate rows found based on primary key columns: {self.primary_key}")
        return is_unique

    def _get_non_pk_rows(self) -> pl.DataFrame:
        """
        Retrieves rows that have duplicate values based on the specified primary key columns.

        Returns:
            pl.DataFrame:
                A DataFrame containing all rows that have duplicate primary key combinations.
        """
        duplicated_mask = self._dataframe.select(self.primary_key).is_duplicated()
        return self._dataframe.filter(duplicated_mask)

    def validate_columns_exist(self, cols: List[str]) -> None:
        """
        Checks whether the specified columns exist in the DataFrame.

        Args:
            cols (List[str]):
                List of columns to check.

        Raises:
            KeyError:
                If any of the specified columns are not present in the DataFrame.
        """
        missing_columns = [col for col in cols if col not in self._dataframe.columns]
        if missing_columns:
            raise KeyError(f"The following columns are missing in the DataFrame: {missing_columns}")

    def get_foreign_key(self, other: "DataFrameManager") -> List[str]:
        """
        Identifies the foreign key columns shared between this DataFrameManager and another.

        Args:
            other (DataFrameManager):
                The other DataFrameManager instance.

        Returns:
            List[str]:
                A list of column names that are present in both primary keys, representing the foreign key.
                If there are no common columns, an empty list is returned.
        """
        foreign_keys = intersection(self.dimensions, other.dimensions)
        logger.debug(f"Identified foreign key columns between DataFrames: {foreign_keys}")
        return foreign_keys

    def get_col_diff(self, other: "DataFrameManager") -> List[str]:
        """
        Identifies columns present in the other DataFrameManager that are not in this one.

        Args:
            other (DataFrameManager):
                The other DataFrameManager instance to compare against.

        Returns:
            List[str]:
                A list of column names that are present in the other DataFrame but not in this one.
                If there are no new columns, an empty list is returned.
        """
        col_diff = sym_diff(other._dataframe.columns, self._dataframe.columns)
        logger.debug(f"Columns in other DataFrame not in this one: {col_diff}")
        return col_diff
