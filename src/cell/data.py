"""Data: the values cells take and return (DESIGN §1.2).

Data is immutable, serializable and has a known structure, which makes it
a pytree: it can be flattened into leaves plus a structure, and rebuilt.
The tracer and vectorization both build on this. The eager runtime uses it
to validate values, to find handles passed as arguments (DESIGN §1.5), and
to compute the digests journal entries are keyed by (DESIGN §3).

Leaves are None, bool, int, float, str, bytes and Enum members. Containers
are tuple, list, dict with str keys, NamedTuples, and frozen dataclasses.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import importlib
import json
import math
from collections.abc import Callable, Iterator
from typing import Any

from .errors import DataError

_PRIMITIVES = (type(None), bool, int, float, str, bytes)


@dataclasses.dataclass(frozen=True)
class TreeDef:
    """The structure of a flattened value: everything except its leaves."""

    kind: str  # leaf, tuple, list, dict, namedtuple, dataclass
    type: type | None = None
    keys: tuple[str, ...] = ()  # dict keys or dataclass field names
    children: tuple[TreeDef, ...] = ()

    @property
    def num_leaves(self) -> int:
        if self.kind == "leaf":
            return 1
        return sum(c.num_leaves for c in self.children)


LEAF = TreeDef("leaf")


def is_primitive(value: Any) -> bool:
    return isinstance(value, _PRIMITIVES) or isinstance(value, enum.Enum)


def flatten(
    value: Any, *, is_leaf: Callable[[Any], bool] | None = None
) -> tuple[list[Any], TreeDef]:
    """Split a data value into its leaves and its structure.

    `is_leaf` lets callers treat extra values as leaves; the runtime uses it
    for handles nested inside arguments. Anything else that is not data
    raises DataError.
    """
    leaves: list[Any] = []
    tree = _flatten(value, leaves, is_leaf)
    return leaves, tree


def _flatten(value: Any, leaves: list[Any], is_leaf: Callable[[Any], bool] | None) -> TreeDef:
    # is_leaf goes first: the tracer's values must not reach isinstance.
    if (is_leaf is not None and is_leaf(value)) or is_primitive(value):
        leaves.append(value)
        return LEAF
    t = type(value)
    if isinstance(value, tuple):
        children = tuple(_flatten(v, leaves, is_leaf) for v in value)
        if hasattr(t, "_fields"):
            return TreeDef("namedtuple", t, tuple(t._fields), children)
        if t is not tuple:
            raise DataError(f"tuple subclass {_qualname(t)} is not data")
        return TreeDef("tuple", children=children)
    if t is list:
        return TreeDef("list", children=tuple(_flatten(v, leaves, is_leaf) for v in value))
    if t is dict:
        for k in value:
            if not isinstance(k, str):
                raise DataError(f"dict keys must be str to be data, not {type(k).__qualname__}")
        keys = tuple(value)
        return TreeDef("dict", keys=keys, children=tuple(_flatten(value[k], leaves, is_leaf) for k in keys))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        if not t.__dataclass_params__.frozen:  # type: ignore[attr-defined]
            raise DataError(f"{_qualname(t)} is not data: dataclasses must be declared frozen=True")
        fields = dataclasses.fields(value)
        for f in fields:
            if not f.init:
                raise DataError(f"{_qualname(t)} is not data: field {f.name!r} has init=False")
        names = tuple(f.name for f in fields)
        children = tuple(_flatten(getattr(value, n), leaves, is_leaf) for n in names)
        return TreeDef("dataclass", t, names, children)
    raise DataError(f"{_qualname(t)} is not data (see DESIGN §1.2)")


def unflatten(tree: TreeDef, leaves: list[Any]) -> Any:
    """Rebuild a value from its structure and (possibly replaced) leaves."""
    it = iter(leaves)
    try:
        value = _unflatten(tree, it)
    except StopIteration:
        raise ValueError("too few leaves for tree") from None
    if next(it, _END) is not _END:
        raise ValueError("too many leaves for tree")
    return value


_END = object()


def _unflatten(tree: TreeDef, it: Iterator[Any]) -> Any:
    kind = tree.kind
    if kind == "leaf":
        return next(it)
    children = [_unflatten(c, it) for c in tree.children]
    if kind == "tuple":
        return tuple(children)
    if kind == "list":
        return children
    if kind == "dict":
        return dict(zip(tree.keys, children))
    if kind == "namedtuple":
        return tree.type(*children)  # type: ignore[misc]
    if kind == "dataclass":
        return tree.type(**dict(zip(tree.keys, children)))  # type: ignore[misc]
    raise AssertionError(kind)


def check(value: Any) -> None:
    """Raise DataError unless value is data."""
    flatten(value)


def encode(value: Any) -> str:
    """A canonical string encoding of a data value.

    Equal data encodes equally, and values of different types never do
    (1, 1.0, True and "1" are all distinct). Dict keys are sorted.

    This is for comparing and hashing values (see `digest`), not for
    serialization: there is no decoder, and the encoding may change.
    """
    leaves, tree = flatten(value)
    return _encode(tree, iter(leaves))


def _encode(tree: TreeDef, it: Iterator[Any]) -> str:
    kind = tree.kind
    if kind == "leaf":
        return _encode_leaf(next(it))
    parts = [_encode(c, it) for c in tree.children]
    if kind == "tuple":
        return "t(" + ",".join(parts) + ")"
    if kind == "list":
        return "l[" + ",".join(parts) + "]"
    if kind == "dict":
        items = sorted(zip(tree.keys, parts))
        return "d{" + ",".join(json.dumps(k) + ":" + p for k, p in items) + "}"
    if kind == "namedtuple":
        return "n" + _qualname(tree.type) + "(" + ",".join(parts) + ")"  # type: ignore[arg-type]
    if kind == "dataclass":
        fields = ",".join(f"{k}={p}" for k, p in zip(tree.keys, parts))
        return "c" + _qualname(tree.type) + "(" + fields + ")"  # type: ignore[arg-type]
    raise AssertionError(kind)


def _encode_leaf(v: Any) -> str:
    if v is None:
        return "N"
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, enum.Enum):
        return f"e{_qualname(type(v))}.{v.name}"
    if isinstance(v, int):
        return f"i{int(v)}"
    if isinstance(v, float):
        return f"f{float(v)!r}"
    if isinstance(v, str):
        return "s" + json.dumps(v)
    if isinstance(v, bytes):
        return "b" + v.hex()
    raise DataError(f"{_qualname(type(v))} is not data (see DESIGN §1.2)")


def digest(value: Any) -> str:
    """A short, stable hash of a data value, from its canonical encoding."""
    return hashlib.sha256(encode(value).encode()).hexdigest()[:32]


def _qualname(t: type) -> str:
    return f"{t.__module__}.{t.__qualname__}"


# Serialization.


def to_json(value: Any) -> Any:
    """A JSON-compatible form of a data value, which `from_json` reverses.

    Lists, str-keyed dicts' contents, str, int, float, bool and None map to
    themselves; everything else is tagged with a "$" key. Types are named by
    `module:qualname`, so they must be importable to be loaded again.
    """
    leaves, tree = flatten(value)
    return _to_json(tree, iter(leaves))


def _to_json(tree: TreeDef, it: Iterator[Any]) -> Any:
    kind = tree.kind
    if kind == "leaf":
        return _leaf_to_json(next(it))
    children = [_to_json(c, it) for c in tree.children]
    if kind == "tuple":
        return {"$tuple": children}
    if kind == "list":
        return children
    if kind == "dict":
        return {"$dict": dict(zip(tree.keys, children))}
    if kind == "namedtuple":
        return {"$namedtuple": type_name(tree.type), "items": children}  # type: ignore[arg-type]
    if kind == "dataclass":
        return {"$dataclass": type_name(tree.type), "fields": dict(zip(tree.keys, children))}  # type: ignore[arg-type]
    raise AssertionError(kind)


def _leaf_to_json(v: Any) -> Any:
    if isinstance(v, enum.Enum):
        return {"$enum": type_name(type(v)), "name": v.name}
    if isinstance(v, float) and not math.isfinite(v):
        return {"$float": repr(v)}
    if isinstance(v, bytes):
        return {"$bytes": v.hex()}
    return v  # None, bool, int, float, str


def from_json(j: Any) -> Any:
    """The data value `to_json` produced `j` from."""
    if j is None or isinstance(j, (bool, int, float, str)):
        return j
    if isinstance(j, list):
        return [from_json(x) for x in j]
    if not isinstance(j, dict):
        raise DataError(f"not a serialized data value: {j!r}")
    if "$tuple" in j:
        return tuple(from_json(x) for x in j["$tuple"])
    if "$dict" in j:
        return {k: from_json(v) for k, v in j["$dict"].items()}
    if "$dataclass" in j:
        return resolve_type(j["$dataclass"])(**{k: from_json(v) for k, v in j["fields"].items()})
    if "$namedtuple" in j:
        return resolve_type(j["$namedtuple"])(*(from_json(x) for x in j["items"]))
    if "$enum" in j:
        return resolve_type(j["$enum"])[j["name"]]
    if "$bytes" in j:
        return bytes.fromhex(j["$bytes"])
    if "$float" in j:
        return float(j["$float"])
    raise DataError(f"not a serialized data value: {j!r}")


def treedef_to_json(tree: TreeDef) -> Any:
    out: dict[str, Any] = {"kind": tree.kind}
    if tree.type is not None:
        out["type"] = type_name(tree.type)
    if tree.keys:
        out["keys"] = list(tree.keys)
    if tree.children:
        out["children"] = [treedef_to_json(c) for c in tree.children]
    return out


def treedef_from_json(j: Any) -> TreeDef:
    if j["kind"] == "leaf":
        return LEAF
    return TreeDef(
        j["kind"],
        resolve_type(j["type"]) if "type" in j else None,
        tuple(j.get("keys", ())),
        tuple(treedef_from_json(c) for c in j.get("children", ())),
    )


def type_name(t: type) -> str:
    return f"{t.__module__}:{t.__qualname__}"


def resolve_type(name: str) -> Any:
    """The type `type_name` named. Types defined inside functions can't be resolved."""
    module, _, qualname = name.partition(":")
    obj: Any = importlib.import_module(module)
    for part in qualname.split("."):
        if part == "<locals>":
            raise DataError(f"type {name} is local to a function and cannot be loaded")
        obj = getattr(obj, part)
    return obj
