"""
This module provides functionality around the primary key, dimensions and measures of a Polars DataFrame.
"""

import logging
from typing import List, Optional

import polars as pl
from polars._typing import UniqueKeepStrategy

from bet_edge.dataframes.column_helpers import intersection, sym_diff

logger = logging.getLogger(__name__)


def validate_columns_exist(df_columns: List[str], cols: List[str], raise_err: bool = True) -> bool:
    """
    Standalone helper to check whether the given columns exist in the provided column list.
   
    Args:
        df_columns (List[str]): The list of columns from the DataFrame.
        cols (List[str]): The columns to check for.
        raise_err (bool): If True, raises a KeyError if any columns are missing.
   
    Returns:
        bool: True if all columns are present, False otherwise.
    """
    missing_columns = [col for col in cols if col not in df_columns]
    if missing_columns and raise_err:
        raise KeyError(f"The following columns are missing in the DataFrame: {missing_columns}")
    elif missing_columns:
        logger.info(f"The following columns are missing in the DataFrame: {missing_columns}")
        return False
    return True


###############################################################################
# LFManager: Works exclusively with LazyFrames.
###############################################################################
class LFManager:
    """
    Manages a Polars LazyFrame with functionality around primary keys, dimensions, and measures.
    Only methods that make sense on lazy data (such as foreign key operations) are implemented.
   
    Use to_df_manager() to collect the lazy operations into an eager DFManager.

    Attributes:
        primary_key (List[str]): A list of column names that constitute the primary key.
        dimensions (List[str]): A list of column names representing dimensions.
        measures (List[str]): A list of column names representing measures.
        lf (pl.LazyFrame): The underlying LazyFrame.
    """
    def __init__(
        self,
        lf: pl.LazyFrame,
        primary_key: List[str],
        dimensions: List[str] = [],
        measures: List[str] = []
    ):
        if not primary_key:
            raise ValueError("Primary key list cannot be empty.")
        self._lf = lf
        self.primary_key = primary_key
        self.dimensions = dimensions
        # If measures are not explicitly provided, use sym_diff from the lazy frame’s columns.
        self.measures = measures or sym_diff(lf.columns, dimensions)

    @property
    def lf(self) -> pl.LazyFrame:
        """
        Returns the underlying LazyFrame.
        """
        return self._lf

    def to_df_manager(self) -> "DFManager":
        """
        Collects the lazy frame to create an eager DataFrame and returns a new DFManager.
       
        Returns:
            DFManager: A new DFManager wrapping an eager DataFrame.
        """
        df = self._lf.collect()
        return DFManager(df, self.primary_key, self.dimensions, self.measures)

    def get_foreign_key(self, other: "LFManager") -> List[str]:
        """
        Identifies the foreign key columns shared between this LFManager and another.
       
        Args:
            other (LFManager): Another LFManager instance.
       
        Returns:
            List[str]: The list of common primary key columns.
        """
        foreign_keys = intersection(self.primary_key, other.primary_key)
        logger.debug(f"Identified foreign key columns between LazyFrames: {foreign_keys}")
        return foreign_keys

    def get_col_diff(self, other: "LFManager") -> List[str]:
        """
        Identifies columns present in the other LFManager that are not in this one.
       
        Args:
            other (LFManager): The other LFManager instance to compare against.
       
        Returns:
            List[str]: A list of column names present in the other LazyFrame but not in this one.
        """
        col_diff = sym_diff(other.lf.columns, self._lf.columns)
        logger.debug(f"Columns in other LazyFrame not in this one: {col_diff}")
        return col_diff


###############################################################################
# DFManager: Works exclusively with eager DataFrames.
###############################################################################
class DFManager:
    """
    Manages an eager Polars DataFrame with functionality to handle primary keys,
    deduplication, and other DataFrame operations.
   
    Attributes:
        primary_key (List[str]): A list of column names that constitute the primary key.
        dimensions (List[str]): A list of column names representing dimensions.
        measures (List[str]): A list of column names representing measures.
        df (pl.DataFrame): The underlying DataFrame.
    """
    def __init__(
        self,
        df: pl.DataFrame,
        primary_key: List[str],
        dimensions: List[str] = [],
        measures: List[str] = []
    ):
        if not primary_key:
            raise ValueError("Primary key list cannot be empty.")
        if isinstance(df, pl.LazyFrame):
            raise TypeError("Eager DFManager received LazyFrame.")
        self._df = df  # Now always an eager DataFrame.
        self.primary_key = primary_key
        self.dimensions = dimensions
        # If measures are not provided, use all columns not in dimensions.
        self.measures = measures or [c for c in df.columns if c not in dimensions]

    @property
    def df(self) -> pl.DataFrame:
        """
        Returns the underlying eager DataFrame.
        """
        return self._df

    def to_lf_manager(self) -> LFManager:
        """
        Converts this DFManager's eager DataFrame into a LazyFrame and returns a new LFManager.
       
        Returns:
            LFManager: A new LFManager wrapping the LazyFrame.
        """
        lazy_df = self._df.lazy()
        return LFManager(lazy_df, self.primary_key, self.dimensions, self.measures)

    def collect(self) -> "DFManager":
        """
        As DFManager always wraps an eager DataFrame, collect() simply returns itself.
       
        Returns:
            DFManager: Self.
        """
        return self

    # --------------------------------------------------------------------------
    # Methods that require an eager DataFrame.
    # --------------------------------------------------------------------------
    def _get_null_columns(self, cols: Optional[List[str]]) -> List[str]:
        """
        Checks if the specified columns (or all columns if None) contain any null values.
       
        Returns:
            List[str]: The list of column names that contain null values.
        """
        if cols is None:
            cols = self._df.columns
        self.validate_columns_exist(cols)
        # Get null count and return any column with count > 0.
        null_counts = self._df.select(cols).null_count()
        counts = null_counts.row(0)
        return [col for col, count in zip(cols, counts) if count > 0]
   
    def get_null_rows(self, cols: Optional[List[str]] = None) -> pl.DataFrame:
        """
        Returns rows from the DataFrame that contain at least one null value in the specified columns.
        """
        if cols is None:
            cols = self._df.columns
        self.validate_columns_exist(cols)
        conditions = [pl.col(col).is_null() for col in cols]
        combined_condition = conditions[0]
        for cond in conditions[1:]:
            combined_condition = combined_condition | cond
        return self._df.filter(combined_condition)

    def null_check(self, cols: Optional[List[str]] = None) -> None:
        """
        Raises a ValueError if any of the specified columns contain null values.
        """
        null_cols = self._get_null_columns(cols)
        if null_cols:
            raise ValueError(f"Found {len(null_cols)} columns with nulls: {null_cols}.")
       
    def check_completeness(
        self,
        other: "DFManager",
        key_cols: Optional[List[str]] = None
    ) -> None:
        """
        Checks that every combination of values in key_cols of the other DFManager is present
        in this DFManager. Raises a ValueError if any combination is missing.
        """
        if key_cols is None:
            key_cols = self.primary_key
        self.validate_columns_exist(key_cols)
        other.validate_columns_exist(key_cols)
        # Use a left anti join via lazy API, then collect.
        missing = other.df.lazy().join(self.df.lazy(), how="anti", on=key_cols).collect()
        if missing.height > 0:
            missing_str = missing.to_pandas().to_string(index=False)
            msg = (
                f"Completeness check failed. The following key combination(s) in the other DFManager "
                f"are not present in this DFManager:\n{missing_str}"
            )
            raise ValueError(msg)

    def assert_valid_pk(self) -> None:
        """
        Asserts that the primary key columns uniquely identify each row.
        """
        if not self._is_pk():
            logger.error(f"DataFrame does not have unique primary key columns: {self.primary_key}.")
            duplicates = self._get_non_pk_rows()
            logger.error(f"Found {duplicates.height} duplicate rows in DataFrame:\n{duplicates}")
            raise AssertionError("DataFrame does not have unique primary key columns.")

    def dedupe_on_pk(self, keep: UniqueKeepStrategy = "first") -> "DFManager":
        """
        Removes duplicate rows based on primary key columns.
        """
        valid_options = {"first", "last", "any", "none"}
        if keep not in valid_options:
            raise TypeError(f"Invalid value for 'keep': {keep}. Must be one of {valid_options}.")
        if not self._is_pk():
            df_before = self._df
            df_dedup = self._df.unique(subset=self.primary_key, keep=keep)
            dropped = df_before.height - df_dedup.height  # type: ignore
            logger.info(f"Found {dropped} duplicate rows based on primary key {self.primary_key}.")
            logger.info(f"Deduplicating DataFrame on primary key {self.primary_key} by keeping '{keep}' entries...")
            logger.info(f"Successfully dropped {dropped} duplicate rows.")
            return DFManager(df_dedup, self.primary_key, self.dimensions, self.measures)
        else:
            logger.info(f"No duplicates found based on primary key {self.primary_key}. No action taken.")
            return self

    def dedupe_if_not_pk(self) -> "DFManager":
        """
        Ensures the DataFrame has a valid primary key by deduplicating if necessary.
        """
        try:
            self.assert_valid_pk()
            return self
        except AssertionError as e:
            logger.warning(f"Primary key validation failed: {e}")
            logger.info(f"Deduplicating DataFrame to enforce primary key {self.primary_key}.")
            deduped_manager = self.dedupe_on_pk(keep="first")
            deduped_manager.assert_valid_pk()
            return deduped_manager

    def _is_pk(self) -> bool:
        """
        Determines whether the primary key uniquely identifies each row.
        """
        self.validate_columns_exist(self.primary_key)
        eager_df = self._df
        unique_rows = eager_df.unique(subset=self.primary_key).height
        total_rows = eager_df.height
        is_unique = unique_rows == total_rows
        if not is_unique:
            logger.warning(f"Duplicate rows found based on primary key columns: {self.primary_key}")
        return is_unique

    def _get_non_pk_rows(self) -> pl.DataFrame:
        """
        Retrieves rows with duplicate primary key combinations.
        """
        duplicated_mask = self._df.select(self.primary_key).is_duplicated()
        return self._df.filter(duplicated_mask)

    def validate_columns_exist(self, cols: List[str], raise_err: bool = True) -> bool:
        """
        Checks if specified columns exist in the DataFrame.
        """
        # Note: our helper expects (existing_columns, cols)
        return validate_columns_exist(self._df.columns, cols, raise_err)

    def get_foreign_key(self, other: "DFManager") -> List[str]:
        """
        Identifies foreign key columns shared between this DFManager and another.
        """
        foreign_keys = intersection(self.primary_key, other.primary_key)
        logger.debug(f"Identified foreign key columns between DataFrames: {foreign_keys}")
        return foreign_keys

    def get_col_diff(self, other: "DFManager") -> List[str]:
        """
        Identifies columns present in the other DFManager that are not in this one.
        """
        col_diff = sym_diff(other.df.columns, self._df.columns)
        logger.debug(f"Columns in other DataFrame not in this one: {col_diff}")
        return col_diff