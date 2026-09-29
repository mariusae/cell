"""The graph IR: a cell's dataflow, as recorded by the tracer (DESIGN §4).

A graph is a list of nodes in SSA form, stored in program order (which is
also a topological order). A node's inputs are either references to
earlier nodes or constants. Constants are operands rather than nodes, so
they print inline and don't need their own ids.

Besides data inputs, an effectful call carries control edges (§4.3):

- `path`: earlier nodes that must have completed successfully, so the
  effect only happens on the path eager execution takes;
- `effect`: earlier effects in the same domain that must have completed;
- `after`: the previous effect in the same domain, which must have been
  issued first.

Edges implied by others are left out: if a node depends on X, it doesn't
also list X's ancestors.
"""

from __future__ import annotations

import hashlib
import json
import operator
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from . import data
from .data import TreeDef
from .semantics import UNIQUE

KINDS = ("param", "call", "op", "pack", "source", "guard", "deopt", "return", "enter", "exit")


@dataclass(frozen=True)
class Const:
    """A constant operand."""

    value: Any

    def __repr__(self) -> str:
        return repr(self.value)


type Input = int | Const


def _method(obj: Any, name: str, *args: Any) -> Any:
    return getattr(obj, name)(*args)


BUILTINS: dict[str, Callable[..., Any]] = {
    "getattr": getattr,
    "getitem": operator.getitem,
    "len": len,
    "truth": operator.truth,
    "contains": operator.contains,
    "method": _method,
    **{
        name: getattr(operator, name)
        for name in (
            "add", "sub", "mul", "truediv", "floordiv", "mod", "pow", "lshift", "rshift",
            "and_", "or_", "xor", "eq", "ne", "lt", "le", "gt", "ge", "neg", "pos", "invert", "abs",
        )
    },
}
"""Ops the tracer records for operations on traced values (DESIGN §5.2)."""

_INFIX = {
    "add": "+", "sub": "-", "mul": "*", "truediv": "/", "floordiv": "//", "mod": "%", "pow": "**",
    "lshift": "<<", "rshift": ">>", "and_": "&", "or_": "|", "xor": "^",
    "eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">=",
}


@dataclass
class Node:
    id: int
    kind: str
    inputs: list[Input] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    path: list[int] = field(default_factory=list)
    effect: list[int] = field(default_factory=list)
    after: int | None = None
    site: str | None = None  # path:line of the code that produced the node

    @property
    def refs(self) -> list[int]:
        """Nodes this one must wait for to complete: data inputs, path and effect edges."""
        return [i for i in self.inputs if isinstance(i, int)] + self.path + self.effect

    @property
    def effectful(self) -> bool:
        return self.kind in ("call", "enter") and self.attrs["effectful"]

    @property
    def scope(self) -> tuple[int, ...]:
        """The call path, from the graph's root, of the invocation that issues this node."""
        return tuple(self.attrs.get("scope", ()))


@dataclass
class Graph:
    """One specialization of a cell (DESIGN §4.1)."""

    cell: str  # cell id
    code: str  # the cell's code hash
    params: list[str]
    nodes: list[Node] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """True if the graph ends in a return rather than a deopt."""
        return bool(self.nodes) and self.nodes[-1].kind == "return"

    @property
    def guards(self) -> list[Node]:
        return [n for n in self.nodes if n.kind in ("guard", "deopt")]

    @property
    def hash(self) -> str:
        """A content hash of the graph, ignoring source locations (DESIGN §4.5)."""
        j = self.to_json()
        for n in j["nodes"]:
            n.pop("site", None)
        encoded = json.dumps(j, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()[:16]

    # Serialization

    def to_json(self) -> dict[str, Any]:
        return {
            "cell": self.cell,
            "code": self.code,
            "params": list(self.params),
            "nodes": [_node_to_json(n) for n in self.nodes],
        }

    @classmethod
    def from_json(cls, j: dict[str, Any]) -> Graph:
        return cls(j["cell"], j["code"], list(j["params"]), [_node_from_json(n) for n in j["nodes"]])

    # Printing

    def format(self, root: str | None = None) -> str:
        """A readable listing (the basis of EXPLAIN).

        Sites print relative to `root` (by default, the current directory)
        for files under it, and as absolute paths otherwise.
        """
        root = os.getcwd() if root is None else root
        header = f"graph {_short(self.cell)} #{self.hash[:8]} ({', '.join(self.params)})"
        rows = [(lhs, tags, _relative(site, root)) for lhs, tags, site in (_format_node(self, n) for n in self.nodes)]
        w1 = max((len(r[0]) for r in rows), default=0)
        w2 = max((len(r[1]) for r in rows), default=0)
        lines = [header]
        for lhs, tags, site in rows:
            line = f"  {lhs:<{w1}}  {tags:<{w2}}  {site}".rstrip()
            lines.append(line)
        return "\n".join(lines)


def _relative(site: str, root: str) -> str:
    if not site:
        return site
    prefix = os.path.join(os.path.abspath(root), "")
    return site[len(prefix):] if site.startswith(prefix) else site


def _short(qualified: str) -> str:
    return qualified.rsplit(".", 1)[-1]


def _ref(i: Input) -> str:
    return f"%{i}" if isinstance(i, int) else repr(i.value)


def _format_node(g: Graph, n: Node) -> tuple[str, str, str]:
    a = n.attrs
    args = ", ".join(_ref(i) for i in n.inputs)
    tags: list[str] = []
    if n.kind == "param":
        body = f"param {a['name']}"
    elif n.kind == "call" and a["cell"].startswith("cell.mapping.map_") and isinstance(n.inputs[0], Const):
        # ctx.map: inputs are the target, its code, the mapped parameter, the items, the fixed arguments.
        target, param, items, fixed = n.inputs[0].value, n.inputs[2].value, n.inputs[3], n.inputs[4]  # type: ignore[union-attr]
        rest = "" if isinstance(fixed, Const) and not fixed.value else f", {_ref(fixed)}"
        body = f"map {_short(target)}({param} in {_ref(items)}{rest})"
        tags.append(f"effectful[{a['domain']}]" if a["effectful"] else "pure")
        tags.append("seq=" + ".".join(str(s) for s in (*n.scope, a["seq"])))
    elif n.kind in ("call", "enter"):
        body = f"{n.kind} {_short(a['cell'])}({args})"
        tags.append(f"effectful[{a['domain']}]" if a["effectful"] else "pure")
        tags.append("seq=" + ".".join(str(s) for s in (*n.scope, a["seq"])))
    elif n.kind == "exit":
        body = f"exit {_short(a['cell'])} {args}"
    elif n.kind == "op":
        body = "op " + _format_op(n)
    elif n.kind == "pack":
        body = "pack " + _format_tree(a["tree"], iter(n.inputs))
    elif n.kind == "source":
        body = f"source {a['name']}({', '.join(repr(v) for v in a['args'].values())})"
        tags.append("seq=" + ".".join(str(s) for s in (*n.scope, a["seq"])))
    elif n.kind == "guard":
        body = f"guard {_ref(n.inputs[0])} == {a['expected']!r}"
    elif n.kind == "deopt":
        body = f"deopt: {a['reason']}"
    elif n.kind == "return":
        return (f"return {args}", "", n.site or "")
    else:
        raise AssertionError(n.kind)
    if n.after is not None:
        tags.append(f"after=%{n.after}")
    if n.effect:
        tags.append("effect=[" + ", ".join(f"%{e}" for e in n.effect) + "]")
    if n.path:
        tags.append("path=[" + ", ".join(f"%{p}" for p in n.path) + "]")
    return (f"%{n.id} = {body}", "  ".join(tags), n.site or "")


def _format_op(n: Node) -> str:
    a = n.attrs
    name = a["op"]
    refs = [_ref(i) for i in n.inputs]
    if not a.get("builtin"):
        return f"{_short(name)}({', '.join(refs)})"
    if name in _INFIX and len(refs) == 2:
        return f"{refs[0]} {_INFIX[name]} {refs[1]}"
    if name == "getattr":
        return f"{refs[0]}.{n.inputs[1].value}"  # type: ignore[union-attr]
    if name == "getitem":
        return f"{refs[0]}[{refs[1]}]"
    if name == "method":
        return f"{refs[0]}.{a['method']}({', '.join(refs[1:])})"
    return f"{name}({', '.join(refs)})"


def _format_tree(tree: TreeDef, it: Iterator[Input]) -> str:
    if tree.kind == "leaf":
        return _ref(next(it))
    parts = [_format_tree(c, it) for c in tree.children]
    if tree.kind == "tuple":
        return "(" + ", ".join(parts) + ("," if len(parts) == 1 else "") + ")"
    if tree.kind == "list":
        return "[" + ", ".join(parts) + "]"
    if tree.kind == "dict":
        return "{" + ", ".join(f"{k!r}: {p}" for k, p in zip(tree.keys, parts)) + "}"
    if tree.kind == "namedtuple":
        return f"{tree.type.__name__}(" + ", ".join(parts) + ")"  # type: ignore[union-attr]
    if tree.kind == "dataclass":
        fields = ", ".join(f"{k}={p}" for k, p in zip(tree.keys, parts))
        return f"{tree.type.__name__}({fields})"  # type: ignore[union-attr]
    raise AssertionError(tree.kind)


# JSON


def _input_to_json(i: Input) -> Any:
    return i if isinstance(i, int) else {"const": data.to_json(i.value)}


def _input_from_json(j: Any) -> Input:
    return j if isinstance(j, int) else Const(data.from_json(j["const"]))


def _domain_to_json(d: Any) -> Any:
    return {"unique": True} if d is UNIQUE else d


def _domain_from_json(j: Any) -> Any:
    return UNIQUE if isinstance(j, dict) else j


def _attrs_to_json(kind: str, attrs: dict[str, Any]) -> dict[str, Any]:
    out = dict(attrs)
    if kind in ("call", "enter"):
        out["domain"] = _domain_to_json(attrs["domain"])
    elif kind == "pack":
        out["tree"] = data.treedef_to_json(attrs["tree"])
    elif kind == "source":
        out["args"] = data.to_json(attrs["args"])
    elif kind == "guard":
        out["expected"] = data.to_json(attrs["expected"])
    return out


def _attrs_from_json(kind: str, j: dict[str, Any]) -> dict[str, Any]:
    out = dict(j)
    if kind in ("call", "enter"):
        out["domain"] = _domain_from_json(j["domain"])
    elif kind == "pack":
        out["tree"] = data.treedef_from_json(j["tree"])
    elif kind == "source":
        out["args"] = data.from_json(j["args"])
    elif kind == "guard":
        out["expected"] = data.from_json(j["expected"])
    return out


def _node_to_json(n: Node) -> dict[str, Any]:
    out: dict[str, Any] = {"id": n.id, "kind": n.kind}
    if n.inputs:
        out["inputs"] = [_input_to_json(i) for i in n.inputs]
    if n.attrs:
        out["attrs"] = _attrs_to_json(n.kind, n.attrs)
    if n.path:
        out["path"] = list(n.path)
    if n.effect:
        out["effect"] = list(n.effect)
    if n.after is not None:
        out["after"] = n.after
    if n.site is not None:
        out["site"] = n.site
    return out


def _node_from_json(j: dict[str, Any]) -> Node:
    kind = j["kind"]
    return Node(
        id=j["id"],
        kind=kind,
        inputs=[_input_from_json(i) for i in j.get("inputs", [])],
        attrs=_attrs_from_json(kind, j.get("attrs", {})),
        path=list(j.get("path", [])),
        effect=list(j.get("effect", [])),
        after=j.get("after"),
        site=j.get("site"),
    )
