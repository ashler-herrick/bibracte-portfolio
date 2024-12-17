# tests/test_dataclass_utils.py

import pytest
from dataclasses import dataclass, field, fields
from typing import Optional, List, Union

from bet_edge.dataclasses.dataclass_from_dict import (
    is_optional,
    get_default_value,
    determine_actual_type,
    process_field_value,
    dataclass_from_dict,
)


# Sample Dataclasses for Testing


@dataclass
class Address:
    street: str
    city: str
    zipcode: Optional[str] = None


@dataclass
class Person:
    name: str
    age: int
    address: Address
    phone_numbers: Optional[List[str]] = None


@dataclass
class Config:
    retries: int = 3
    timeout: Optional[int] = None
    endpoints: List[str] = field(default_factory=lambda: ["https://api.example.com"])


# Helper Classes for Testing Nested Structures


@dataclass
class Employee:
    name: str
    position: str


@dataclass
class Company:
    name: str
    employees: List[Employee]


# Pytest Fixtures (if needed)

# None needed in this simple example, but you can add fixtures for complex setups.


# Test Cases for `is_optional`


@pytest.mark.parametrize(
    "field_type,expected",
    [
        (Optional[int], True),
        (int, False),
        (Union[str, None], True),
        (Union[str, int], False),
        (Union[str, int, None], True),
        (List[int], False),
        (Optional[List[int]], True),
    ],
)
def test_is_optional(field_type, expected):
    assert is_optional(field_type) == expected


# Test Cases for `get_default_value`


def test_get_default_value():
    @dataclass
    class Example:
        a: int
        b: int = field(default_factory=lambda: 20)
        c: int = 10

    example_field_a = fields(Example)[0]
    example_field_b = fields(Example)[1]
    example_field_c = fields(Example)[2]

    assert get_default_value(example_field_a) is None
    assert get_default_value(example_field_b) == 20
    assert get_default_value(example_field_c) == 10


# Test Cases for `determine_actual_type`


@pytest.mark.parametrize(
    "field_type,expected",
    [
        (Optional[int], int),
        (int, int),
        (Union[str, None], str),
        (Union[str, int, None], Union[str, int, None]),
        (List[int], List[int]),
    ],
)
def test_determine_actual_type(field_type, expected):
    assert determine_actual_type(field_type) == expected


# Test Cases for `process_field_value`


def test_process_field_value_with_dataclass():
    address_data = {"street": "123 Main St", "city": "Anytown"}
    address = process_field_value(Address, address_data)
    assert isinstance(address, Address)
    assert address.street == "123 Main St"
    assert address.city == "Anytown"
    assert address.zipcode is None


def test_process_field_value_with_list_of_dataclasses():
    employee_data = [{"name": "Alice", "position": "Developer"}, {"name": "Bob", "position": "Designer"}]
    company = process_field_value(List[Employee], employee_data)
    assert isinstance(company, list)
    assert len(company) == 2
    assert all(isinstance(emp, Employee) for emp in company)
    assert company[0].name == "Alice"
    assert company[1].position == "Designer"


def test_process_field_value_with_list_of_primitives():
    phone_numbers = ["123-456-7890", "987-654-3210"]
    processed = process_field_value(List[str], phone_numbers)
    assert processed == phone_numbers


def test_process_field_value_with_invalid_type():
    with pytest.raises(TypeError) as exc_info:
        process_field_value(Address, ["not", "a", "dict"])
    assert "Expected a dict for type" in str(exc_info.value)


# Test Cases for `dataclass_from_dict`


def test_dataclass_from_dict_with_optional_fields():
    data = {
        "name": "John Doe",
        "age": 30,
        "address": {
            "street": "123 Main St",
            "city": "Anytown",
            # 'zipcode' is missing, which is optional
        },
        # 'phone_numbers' is missing, which is optional
    }

    person = dataclass_from_dict(Person, data)
    assert person.name == "John Doe"
    assert person.age == 30
    assert isinstance(person.address, Address)
    assert person.address.street == "123 Main St"
    assert person.address.city == "Anytown"
    assert person.address.zipcode is None
    assert person.phone_numbers is None


def test_dataclass_from_dict_with_all_fields():
    data = {
        "name": "Jane Smith",
        "age": 25,
        "address": {"street": "456 Elm St", "city": "Othertown", "zipcode": "12345"},
        "phone_numbers": ["555-1234", "555-5678"],
    }

    person = dataclass_from_dict(Person, data)
    assert person.name == "Jane Smith"
    assert person.age == 25
    assert isinstance(person.address, Address)
    assert person.address.zipcode == "12345"
    assert person.phone_numbers == ["555-1234", "555-5678"]


def test_dataclass_from_dict_with_missing_required_field():
    data = {
        "name": "Jane Doe",
        # 'age' is missing
        "address": {"street": "789 Oak St", "city": "Sometown"},
    }

    with pytest.raises(KeyError) as exc_info:
        dataclass_from_dict(Person, data)
    assert "Missing required field" in str(exc_info.value)


def test_dataclass_from_dict_with_type_mismatch():
    data = {
        "name": "Jane Doe",
        "age": "twenty-five",  # Should be int
        "address": {"street": "789 Oak St", "city": "Sometown"},
    }

    with pytest.raises(TypeError) as exc_info:
        dataclass_from_dict(Person, data)
    assert "Error processing field 'age'" in str(exc_info.value)


def test_dataclass_from_dict_with_default_factory():
    data = {
        # 'retries' is missing, should use default value 3
        "endpoints": ["https://api.another.com"]
    }

    config = dataclass_from_dict(Config, data)
    assert config.retries == 3
    assert config.timeout is None
    assert config.endpoints == ["https://api.another.com"]


def test_dataclass_from_dict_with_default_factory_called():
    data = {
        "retries": 5,
        # 'endpoints' is missing, should use default factory
    }

    config = dataclass_from_dict(Config, data)
    assert config.retries == 5
    assert config.endpoints == ["https://api.example.com"]
    assert config.timeout is None


def test_dataclass_from_dict_with_nested_dataclasses():
    @dataclass
    class Project:
        name: str
        company: Company

    company_data = {
        "name": "Tech Corp",
        "employees": [{"name": "Alice", "position": "Developer"}, {"name": "Bob", "position": "Designer"}],
    }

    project_data = {"name": "New Project", "company": company_data}

    project = dataclass_from_dict(Project, project_data)
    assert project.name == "New Project"
    assert isinstance(project.company, Company)
    assert project.company.name == "Tech Corp"
    assert len(project.company.employees) == 2
    assert project.company.employees[0].name == "Alice"
    assert project.company.employees[1].position == "Designer"
