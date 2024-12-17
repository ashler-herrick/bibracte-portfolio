from bet_edge.dataframes.column_helpers import sym_diff, intersection, union


def test_intersection():
    res = intersection(["1"], ["1", "2"], ["1", "2", "3"])
    assert set(res) == set(["1"])

    res = intersection(["1"], [])
    assert res == []


def test_union():
    res = union(["1"], ["2"], ["3"])
    assert set(res) == set(["1", "2", "3"])


def test_sym_diff():
    res = sym_diff(["1"], ["2"])
    assert set(res) == set(["1", "2"])

    res = sym_diff(["1", "2"], ["2"])
    assert set(res) == set(["1"])

    res = sym_diff(["1", "2"], ["1", "2"])
    assert res == []

    res = sym_diff(["1", "2"], ["2", "1"])
    assert res == []

    res = sym_diff(["1", "2"], ["1"], ["3"])
    assert set(res) == set(["2", "3"])
