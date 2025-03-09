import pytest
import polars as pl
from polars.testing import assert_frame_equal

from bet_edge.dataframes.managers import DataFrameManager


@pytest.fixture
def sample_dfm() -> DataFrameManager:
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "email": [
                "alice@example.com",
                "bob@example.com",
                "charlie@example.com",
                "david@example.com",
                "eve@example.com",
            ],
        }
    )
    return DataFrameManager(df, ["id"])


@pytest.fixture
def sample_dfm2() -> DataFrameManager:
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        }
    )
    return DataFrameManager(df, ["id"])


@pytest.fixture
def duplicate_dfm() -> DataFrameManager:
    df = pl.DataFrame(
        {
            "id": [1, 2, 2, 4, 5],
            "name": ["Alice", "Bob", "Bob", "David", "Eve"],
            "email": [
                "alice@example.com",
                "bob@example.com",
                "bob.duplicate@example.com",
                "david@example.com",
                "eve@example.com",
            ],
        }
    )
    return DataFrameManager(df, ["id"])


@pytest.fixture
def duplicate_df_rows():
    return pl.DataFrame(
        {
            "id": [2, 2],
            "name": ["Bob", "Bob"],
            "email": [
                "bob@example.com",
                "bob.duplicate@example.com",
            ],
        }
    )


def test_valid_primary_key(sample_dfm):
    # Test with 'id' as primary key
    assert sample_dfm._is_pk() is True

    sample_dfm.primary_key = ["id", "email"]
    # Test with ['id', 'email'] as primary key
    assert sample_dfm._is_pk() is True


def test_invalid_primary_key(duplicate_dfm):
    # 'id' has duplicates
    assert duplicate_dfm._is_pk() is False

    duplicate_dfm.primary_key = ["name", "email"]
    # ['name', 'email'] is unique
    assert duplicate_dfm._is_pk() is True

    duplicate_dfm.primary_key = ["id", "name"]
    # ['id', 'name'] has duplicates because 'id' is duplicated
    assert duplicate_dfm._is_pk() is False


def test_missing_columns(sample_dfm):
    sample_dfm.primary_key = ["id", "nonexistent_column"]
    with pytest.raises(KeyError) as excinfo:
        sample_dfm._is_pk()
    assert "The following columns are missing in the DataFrame" in str(excinfo.value)


def test_single_column_primary_key(sample_dfm):
    # 'email' should be unique
    sample_dfm.primary_key = ["email"]
    assert sample_dfm._is_pk() is True


def test_all_columns_primary_key(sample_dfm):
    # Using all columns as primary key should be unique
    sample_dfm.primary_key = ["id", "name", "email"]
    assert sample_dfm._is_pk() is True


def test_partial_unique_columns(duplicate_dfm):
    # In duplicate_df, 'name' and 'email' are unique
    duplicate_dfm.primary_key = ["name", "email"]
    assert duplicate_dfm._is_pk() is True

    # 'id' is not unique, even when combined with 'name'
    duplicate_dfm.primary_key = ["id", "name"]
    assert duplicate_dfm._is_pk() is False


def test_duplicate_pk_rows(duplicate_dfm, duplicate_df_rows):
    assert_frame_equal(duplicate_dfm._get_non_pk_rows(), duplicate_df_rows)


def test_assert_valid_pk_invalid(duplicate_dfm):
    with pytest.raises(AssertionError):
        duplicate_dfm.assert_valid_pk()


def test_assert_valid_pk_valid(sample_dfm):
    sample_dfm.assert_valid_pk()


def test_dedupe_on_pk(duplicate_dfm):
    with pytest.raises(AssertionError):
        duplicate_dfm.assert_valid_pk()

    dfm = duplicate_dfm.dedupe_on_pk()
    dfm.assert_valid_pk()


def test_dedupe_if_not_pk(sample_dfm, duplicate_dfm):
    dfm = sample_dfm.dedupe_if_not_pk()
    assert dfm.dataframe.height == 5

    dfm = duplicate_dfm.dedupe_if_not_pk()
    assert dfm.dataframe.height == 4


def test_validate_columns_exist(sample_dfm):
    sample_dfm.validate_columns_exist(["id"])

    with pytest.raises(KeyError) as excinfo:
        sample_dfm.validate_columns_exist(["not_a_column"])
    assert "The following columns are missing in the DataFrame" in str(excinfo.value)


def tet_get_col_diff(sample_dfm, sample_dfm2, duplicate_dfm):
    assert sample_dfm.get_col_diff(sample_dfm2) == ["email"]

    assert sample_dfm2.get_col_diff(sample_dfm) == ["email"]

    assert sample_dfm.get_col_diff(duplicate_dfm) == []

def test_get_foreign_key():
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 1, 2],
            "season": [2020, 2020, 2020, 2021, 2021],
            "week": [1, 1, 1, 1, 1],
            "team": ["A", "B", "C", "A","B"],
            "position": ["QB", "TE", "RB", "QB", "TE"]
        }
    )
    dfm = DataFrameManager(df, ["id", "season", "week"], ["id", "season", "week", "team", "position"])

    team = pl.DataFrame(
        {
            "season": [2020, 2020, 2020, 2021, 2021],
            "team": ["A", "B", "C", "A", "B"]
        }
    )
    team_dfm = DataFrameManager(team, ["season", "team"], ["season", "team"])

    fk = dfm.get_foreign_key(team_dfm)
    assert set(fk) == set(["season", "team"])

