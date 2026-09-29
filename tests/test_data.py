import dataclasses
import enum
from typing import NamedTuple

import pytest

from cell import DataError
from cell.data import check, digest, encode, flatten, unflatten


@dataclasses.dataclass(frozen=True)
class Point:
    x: int
    y: float


@dataclasses.dataclass
class Mutable:
    x: int


class Pair(NamedTuple):
    a: int
    b: str


class Color(enum.Enum):
    RED = 1


class Level(enum.IntEnum):
    LOW = 1


VALUES = [
    None,
    True,
    3,
    2.5,
    "s",
    b"b",
    Color.RED,
    Level.LOW,
    (1, "a"),
    [1, [2, 3]],
    {"a": 1, "b": (2,)},
    Point(1, 2.0),
    Pair(1, "x"),
    {"points": [Point(1, 1.0), Point(2, 2.0)], "pair": Pair(3, "y")},
]


@pytest.mark.parametrize("value", VALUES, ids=repr)
def test_flatten_roundtrip(value):
    leaves, tree = flatten(value)
    assert tree.num_leaves == len(leaves)
    rebuilt = unflatten(tree, leaves)
    assert rebuilt == value
    assert type(rebuilt) is type(value)


def test_unflatten_replaces_leaves():
    leaves, tree = flatten({"p": Point(1, 2.0), "n": [3]})
    assert unflatten(tree, [x * 10 for x in leaves]) == {"p": Point(10, 20.0), "n": [30]}


def test_unflatten_checks_leaf_count():
    _, tree = flatten((1, 2))
    with pytest.raises(ValueError):
        unflatten(tree, [1])
    with pytest.raises(ValueError):
        unflatten(tree, [1, 2, 3])


def test_extra_leaves():
    marker = object()
    leaves, tree = flatten({"a": marker, "b": 1}, is_leaf=lambda v: v is marker)
    assert leaves == [marker, 1]
    with pytest.raises(DataError):
        check({"a": marker})


@pytest.mark.parametrize(
    "value",
    [{1: "int key"}, {1, 2}, Mutable(1), object(), frozenset(), lambda: 0],
    ids=["int-key", "set", "mutable-dataclass", "object", "frozenset", "function"],
)
def test_not_data(value):
    with pytest.raises(DataError):
        check(value)


def test_digest_distinguishes_types():
    values = [1, 1.0, True, "1", b"1", (1,), [1], Level.LOW, None, 0, False, ""]
    assert len({digest(v) for v in values}) == len(values)


def test_digest_ignores_dict_order():
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})


def test_digest_distinguishes_structure():
    assert digest(Point(1, 2.0)) != digest(Pair(1, "2"))
    assert digest({"x": 1, "y": 2.0}) != digest(Point(1, 2.0))
    assert digest(("a", "b")) != digest(("ab",))


def test_encode_is_stable():
    assert encode({"b": [1, 2.5], "a": Point(1, 0.5)}) == (
        'd{"a":ctest_data.Point(x=i1,y=f0.5),"b":l[i1,f2.5]}'
    )
