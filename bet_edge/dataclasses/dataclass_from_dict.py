from dataclasses import is_dataclass, fields, MISSING, Field
from typing import Any, Type, TypeVar, List, Dict, Union, get_origin, get_args, cast

T = TypeVar("T")
FieldType = type[Any] | str | Any


def is_optional(field_type: FieldType) -> bool:
    """
    Determines whether a given type annotation is an Optional field.

    Args:
        field_type (Type[Any]): The type annotation to check.

    Returns:
        bool: True if the type annotation is Optional, False otherwise.

    Notes:
        - This function specifically checks if the type is a Union containing `NoneType`.
        - It is commonly used to identify fields annotated as `Optional[...]` in dataclasses.
    """
    origin = get_origin(field_type)
    if origin is Union:
        args = get_args(field_type)
        return type(None) in args
    return False


def get_default_value(field: Field) -> Any:
    """
    Retrieves the default value for a dataclass field, considering default and default_factory.

    Args:
        field (Field): The dataclass field.

    Returns:
        Any: The default value for the field, or None if no default is provided.

    Raises:
        TypeError: If the field has a default_factory but it's not callable.
    """
    if field.default is not MISSING:
        return field.default
    elif field.default_factory is not MISSING:
        if callable(field.default_factory):
            return field.default_factory()
        else:
            raise TypeError(f"Default factory for field '{field.name}' is not callable.")
    else:
        return None


def determine_actual_type(field_type: FieldType) -> FieldType:
    """
    Determines the actual type of a field by removing Optional if present.

    Args:
        field_type (Type[Any]): The type annotation of the field.

    Returns:
        Type[Any]: The actual type of the field without Optional.
    """
    origin = get_origin(field_type)
    args = get_args(field_type)

    if origin is Union:
        non_none_types = [arg for arg in args if arg is not type(None)]
        if len(non_none_types) == 1:
            return non_none_types[0]
    return field_type


def process_field_value(field_type: FieldType, value: Any) -> Any:
    """
    Processes the value based on its type, handling nested dataclasses and lists.
    Includes type validation to ensure the value matches the expected type.

    Args:
        field_type (Type[Any]): The type annotation of the field.
        value (Any): The value to process.

    Returns:
        Any: The processed value, potentially a dataclass instance or a list of processed values.

    Raises:
        TypeError: If the value does not match the expected type.
    """
    actual_type = determine_actual_type(field_type)
    origin = get_origin(actual_type)
    args = get_args(actual_type)

    if origin in (list, List):
        item_type = args[0] if args else Any
        if is_dataclass(item_type):
            # Inform the type checker that item_type is indeed a Type[T]
            item_type_casted = cast(Type[Any], item_type)
            if not isinstance(value, list):
                raise TypeError(f"Expected a list for type '{actual_type}', got '{type(value).__name__}'")
            return [dataclass_from_dict(item_type_casted, item) if isinstance(item, dict) else item for item in value]
        else:
            if not isinstance(value, list):
                raise TypeError(f"Expected a list for type '{actual_type}', got '{type(value).__name__}'")
            # Type validation for list items
            for item in value:
                if not isinstance(item, item_type) and item_type is not Any:  # type: ignore
                    raise TypeError(f"List item '{item}' does not match expected type '{item_type.__name__}'")
            return value
    elif is_dataclass(actual_type):
        # Inform the type checker that actual_type is indeed a Type[T]
        actual_type_casted = cast(Type[Any], actual_type)
        if not isinstance(value, dict):
            raise TypeError(f"Expected a dict for type '{actual_type}', got '{type(value).__name__}'")
        return dataclass_from_dict(actual_type_casted, value)
    else:
        # Type validation for simple types
        if not isinstance(value, actual_type) and actual_type is not Any:  # type: ignore
            raise TypeError(f"Expected type '{actual_type.__name__}' for value '{value}', got '{type(value).__name__}'")  # type: ignore
        return value


def dataclass_from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    """
    Recursively parses a dictionary into a dataclass instance.

    Args:
        cls (Type[T]): The dataclass type to instantiate.
        data (Dict[str, Any]): The dictionary containing the data to populate the dataclass.

    Returns:
        T: An instance of the dataclass populated with the provided data.

    Raises:
        ValueError: If the provided `cls` is not a dataclass.
        KeyError: If a required field is missing from the input dictionary.
        TypeError: If the type of a value in the dictionary does not match the expected type.

    Behavior:
        - Nested dataclasses are automatically instantiated recursively.
        - Optional fields (`Optional[...]`) that are missing from the dictionary are set to `None`.
        - Fields with default values or default factories are populated with those defaults if missing.
        - Supports lists of nested dataclasses, recursively converting their elements if needed.

    Notes:
        - Type annotations are used to determine how fields are processed.
        - For `List` types, the function assumes the list elements are either plain types or dictionaries corresponding to dataclasses.
        - Complex `Union` types (beyond `Optional[T]`) are not fully supported and may require additional logic.
    """
    if not is_dataclass(cls):
        raise ValueError(f"{cls} is not a dataclass.")

    init_kwargs = {}

    for field in fields(cls):
        field_name = field.name
        field_type = field.type

        if field_name not in data:
            # Check if the field is Optional or has a default value
            if is_optional(field_type) or field.default is not MISSING or field.default_factory is not MISSING:
                init_kwargs[field_name] = get_default_value(field)
            else:
                raise KeyError(f"Missing required field: '{field_name}'")
            continue

        value = data[field_name]

        try:
            processed_value = process_field_value(field_type, value)
            init_kwargs[field_name] = processed_value
        except Exception as e:
            raise TypeError(f"Error processing field '{field_name}': {e}") from e

    return cls(**init_kwargs)
