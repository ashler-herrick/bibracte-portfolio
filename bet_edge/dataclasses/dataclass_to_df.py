from dataclasses import is_dataclass, fields
from typing import Any, Dict, List
import polars as pl


def _merge_records(base_records: List[Dict[str, Any]], new_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merges two lists of records by combining each base record with each new record.

    Args:
        base_records: The existing list of records.
        new_records: The new list of records to merge.

    Returns:
        A new list of merged records.
    """
    if not base_records:
        return new_records
    if not new_records:
        return base_records

    merged = []
    for base in base_records:
        for new in new_records:
            merged_record = base.copy()
            merged_record.update(new)
            merged.append(merged_record)
    return merged


def flatten_dataclass(instance: Any, sep: str = "_") -> List[Dict[str, Any]]:
    """
    Recursively flattens a dataclass instance into a list of dictionaries.

    Args:
        instance: The dataclass instance to flatten.
        parent_key: The base key string for nested fields.
        sep: Separator used between parent and child keys.

    Returns:
        A list of flattened dictionaries.
    """
    if not is_dataclass(instance):
        raise ValueError(f"Expected dataclass instance, got {type(instance)}")

    records = [{}]

    for field in fields(instance):
        value = getattr(instance, field.name)
        key = field.name

        if is_dataclass(value):
            # Recursively flatten nested dataclass and merge with existing records
            nested_records = flatten_dataclass(value, sep=sep)
            records = _merge_records(records, nested_records)

        elif isinstance(value, list):
            # Check if the list contains dataclass instances
            if value and all(is_dataclass(item) for item in value):
                # For each item in the list, flatten and merge
                nested_records = []
                for item in value:
                    nested = flatten_dataclass(item, sep=sep)
                    nested_records.extend(nested)
                records = _merge_records(records, nested_records)
            else:
                # List of primitives; add as-is
                for record in records:
                    record[key] = value
        else:
            # Primitive field; add to all records
            for record in records:
                record[key] = value

    return records


def dataclass_to_polars_df(instance: Any) -> pl.DataFrame:
    """
    Converts a dataclass instance into a Polars DataFrame, handling nested dataclasses.

    Args:
        instance: The dataclass instance to convert.
        sep: Separator used between parent and child keys in flattened fields.

    Returns:
        A Polars DataFrame containing the flattened data.
    """
    flattened_records = flatten_dataclass(instance)
    return pl.DataFrame(flattened_records)
