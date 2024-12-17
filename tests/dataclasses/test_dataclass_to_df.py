from dataclasses import dataclass
from typing import List

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bet_edge.dataclasses.dataclass_to_df import dataclass_to_polars_df


@dataclass
class Address:
    street: str
    city: str
    zip_code: str


@dataclass
class Person:
    id: int
    name: str
    age: int
    addresses: List[Address]


@pytest.fixture
def person():
    return Person(
        id=100,
        name="John Doe",
        age=30,
        addresses=[
            Address(street="123 Elm St", city="Springfield", zip_code="12345"),
            Address(street="456 Oak St", city="Shelbyville", zip_code="67890"),
        ],
    )


def test_dataclass_to_polars_df(person):
    df = dataclass_to_polars_df(person)

    expected = {
        "id": [100, 100],
        "name": ["John Doe", "John Doe"],
        "age": [30, 30],
        "street": ["123 Elm St", "456 Oak St"],
        "city": ["Springfield", "Shelbyville"],
        "zip_code": ["12345", "67890"],
    }
    expected_df = pl.DataFrame(expected)
    print(df)
    assert_frame_equal(expected_df, df)
