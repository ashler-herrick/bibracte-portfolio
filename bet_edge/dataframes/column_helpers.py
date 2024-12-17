from typing import List
from functools import reduce


def intersection(*lists: List[str]) -> List[str]:
    """
    Returns the intersection of multiple lists.

    Useful for determining common elements across several lists, such as foreign key columns for a join.

    Args:
        *lists (List[str]): Variable number of lists to intersect.

    Returns:
        List[str]: A list containing elements common to all input lists.
    """
    if not lists:
        return []
    sets = map(set, lists)
    result = reduce(lambda acc, s: acc & s, sets)
    return list(result)


def union(*lists: List[str]) -> List[str]:
    """
    Returns the union of multiple lists.

    Useful for combining elements from several lists, such as aggregating all unique columns from multiple tables.

    Args:
        *lists (List[str]): Variable number of lists to unite.

    Returns:
        List[str]: A list containing all unique elements from all input lists.
    """
    if not lists:
        return []
    sets = map(set, lists)
    result = reduce(lambda acc, s: acc | s, sets)
    return list(result)


def sym_diff(*lists: List[str]) -> List[str]:
    """
    Returns the symmetric difference of multiple lists.

    Useful for finding elements present in one list but not others.

    Args:
        *lists (List[str]): Variable number of lists to compute the symmetric difference.

    Returns:
        List[str]: A list containing elements that are in an odd number of input lists.
    """
    if not lists:
        return []
    sets = map(set, lists)
    result = reduce(lambda acc, s: acc ^ s, sets)
    return list(result)
