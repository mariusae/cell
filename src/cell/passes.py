"""Graph passes: graph-to-graph rewrites that preserve meaning (DESIGN §10, milestone M4).

Each pass returns a new graph. The logical graph stays the tracer's; a pass
produces the graph that runs. Passes only apply rewrites that the declared
semantics make legal (NOTES §2): pure calls may be shared, reordered or
dropped, effectful calls never.

- `dedup` shares identical pure computations: pure calls, ops, packs and
  guards with the same inputs.
- `inline` replaces calls to composite cells with the callees' graphs.
- `optimize` applies them in order.

A graph that has been through passes issues fewer pure calls than eager
execution would, so compiled runs match eager execution in their outcome
and effects (Journal.effects), not call for call.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from . import data
from .graph import BUILTINS, Const, Graph, Input, Node


def _renumber(graph: Graph, build: Callable[[Node, Callable[[Input], Input]], list[Node] | Node | int]) -> Graph:
    """Rebuild a graph node by node, renumbering references.

    `build` gets each node and a function mapping old operands to new ones,
    and returns new node(s) (their ids are assigned here), or the new id of
    an existing node to stand for this one.
    """
    new: list[Node] = []
    ids: dict[int, int] = {}

    def remap(i: Input) -> Input:
        return ids[i] if isinstance(i, int) else i

    for n in graph.nodes:
        out = build(n, remap)
        if isinstance(out, int):
            ids[n.id] = out
            continue
        for m in [out] if isinstance(out, Node) else out:
            m.id = len(new)
            new.append(m)
        ids[n.id] = new[-1].id
    return Graph(graph.cell, graph.code, list(graph.params), new)


def _copy(n: Node, remap: Callable[[Input], Input]) -> Node:
    refs = lambda xs: sorted({remap(x) for x in xs})  # noqa: E731
    return Node(
        n.id,
        n.kind,
        [remap(i) for i in n.inputs],
        dict(n.attrs),
        refs(n.path),  # type: ignore[arg-type]
        refs(n.effect),  # type: ignore[arg-type]
        None if n.after is None else remap(n.after),  # type: ignore[arg-type]
        n.site,
    )


def _operand(i: Input) -> Any:
    return i if isinstance(i, int) else ("const", data.digest(i.value))


def dedup(graph: Graph) -> Graph:
    """Share identical pure calls, ops, packs and guards."""
    seen: dict[Any, Node] = {}

    def build(n: Node, remap: Callable[[Input], Input]) -> Node | int:
        m = _copy(n, remap)
        key = _key(m)
        if key is None:
            return m
        if key in seen:
            return seen[key].id  # already renumbered: it was kept earlier
        seen[key] = m
        return m

    return _renumber(graph, build)


def _key(n: Node) -> Any:
    a = n.attrs
    inputs = tuple(_operand(i) for i in n.inputs)
    if n.kind == "call" and not a["effectful"]:
        # Across inlined scopes too: a pure call's result doesn't depend on who asks.
        return ("call", a["cell"], a["code"], tuple(a["params"]), inputs)
    if n.kind == "op":
        return ("op", a["op"], a.get("code"), a.get("method"), tuple(a.get("kwargs", ())), inputs)
    if n.kind == "pack":
        return ("pack", a["tree"], inputs)
    if n.kind == "guard":
        return ("guard", inputs, data.digest(a["expected"]))
    return None


def fold(graph: Graph) -> Graph:
    """Evaluate what depends only on constants.

    Builtin ops and packs whose inputs are all constants become constants.
    A guard on a constant that holds is dropped; one that can't hold is
    kept, so the graph still deopts there. Edges to what was folded are
    dropped: a constant has always completed.
    """
    new: list[Node] = []
    ids: dict[int, Input] = {}

    def value(i: Input) -> Input:
        return ids[i] if isinstance(i, int) else i

    def refs(xs: list[int]) -> list[int]:
        return sorted({r for x in xs if isinstance(r := ids[x], int)})

    for n in graph.nodes:
        inputs = [value(i) for i in n.inputs]
        constant = all(isinstance(i, Const) for i in inputs)
        a = n.attrs
        if constant and n.kind == "op" and a.get("builtin"):
            folded = _evaluate(a, [i.value for i in inputs])  # type: ignore[union-attr]
            if folded is not None:
                ids[n.id] = folded
                continue
        if constant and n.kind == "pack":
            ids[n.id] = Const(data.unflatten(a["tree"], [i.value for i in inputs]))  # type: ignore[union-attr]
            continue
        if constant and n.kind == "guard" and data.digest(inputs[0].value) == data.digest(a["expected"]):  # type: ignore[union-attr]
            ids[n.id] = Const(None)
            continue
        after = None if n.after is None else ids[n.after]
        m = Node(len(new), n.kind, inputs, dict(a), refs(n.path), refs(n.effect), after if isinstance(after, int) else None, n.site)
        new.append(m)
        ids[n.id] = m.id
    return Graph(graph.cell, graph.code, list(graph.params), new)


def _evaluate(attrs: Mapping[str, Any], args: list[Any]) -> Const | None:
    """A builtin op on constants, or None if it raises (it then deopts at run time)."""
    try:
        if attrs["op"] == "method":
            result = BUILTINS["method"](args[0], attrs["method"], *args[1:])
        else:
            result = BUILTINS[attrs["op"]](*args)
        data.check(result)
    except Exception:
        return None
    return Const(result)


def doomed(graph: Graph) -> set[tuple[int, ...]]:
    """Scopes of inlined callees with a guard on a constant that can't hold."""
    return {n.scope for n in graph.nodes if n.kind == "guard" and isinstance(n.inputs[0], Const) and n.scope}


def optimize(graph: Graph, callees: Mapping[str, Graph] | None = None) -> Graph:
    """The standard pipeline: inline composite callees, fold constants, share what's identical.

    A callee inlined at a call site where one of its guards can never hold
    would make every run deopt; such call sites are left as calls.
    """
    exclude: set[tuple[int, ...]] = set()
    while True:
        out = fold(inline(graph, callees or {}, exclude=exclude))
        bad = {scope[:1] for scope in doomed(out)} - exclude
        if not bad:
            return dedup(out)
        exclude |= bad


def inline(
    graph: Graph,
    callees: Mapping[str, Graph],
    _stack: tuple[str, ...] = (),
    exclude: set[tuple[int, ...]] = frozenset(),  # type: ignore[assignment]
) -> Graph:
    """Replace calls to cells with complete graphs in `callees` by those graphs.

    A call node becomes:

    - `enter`: issues the call in the parent's journal and starts the
      callee's invocation. It keeps the call's inputs and control edges.
    - the callee's nodes, with its parameters replaced by the call's
      inputs. Their calls and sources are issued from the callee's
      invocation, named by `scope` (the call path from the graph's root).
      Everything that issues waits for `enter`.
    - `exit`: the callee's result. It completes when the result is ready
      and every call issued in the callee's scope has completed.

    Uses of the call's value, and its completion, become uses of `exit`;
    its `after` edges (issue order) refer to `enter`. Callees are inlined
    recursively, but not into themselves, and not at call sites whose scope
    is in `exclude`.
    """
    new: list[Node] = []
    done: dict[int, int] = {}  # old id -> new id, for values and completion
    issued: dict[int, int] = {}  # old id -> new id, for issue order

    def add(n: Node) -> int:
        n.id = len(new)
        new.append(n)
        return n.id

    def remap(i: Input) -> Input:
        return done[i] if isinstance(i, int) else i

    def copy(n: Node) -> Node:
        m = _copy(n, remap)
        m.after = None if n.after is None else issued.get(n.after, done[n.after])
        return m

    for n in graph.nodes:
        callee = callees.get(n.attrs.get("cell", "")) if n.kind == "call" else None
        site = (*n.attrs.get("scope", []), n.attrs["seq"]) if n.kind == "call" else ()
        if (
            callee is None
            or not callee.complete
            or callee.code != n.attrs["code"]
            or callee.cell in (*_stack, graph.cell)
            or site in exclude
        ):
            done[n.id] = add(copy(n))
            continue
        callee = inline(callee, callees, _stack + (graph.cell,))
        enter = copy(n)
        enter.kind = "enter"
        eid = add(enter)
        scope = list(site)
        params = {p.id: enter.inputs[p.attrs["index"]] for p in callee.nodes if p.kind == "param"}
        inner: dict[int, Input] = dict(params)
        result: Input | None = None
        for c in callee.nodes:
            if c.kind == "param":
                continue
            if c.kind == "return":
                result = _map(c.inputs[0], inner)
                continue
            m = Node(
                -1,
                c.kind,
                [_map(i, inner) for i in c.inputs],
                dict(c.attrs),
                sorted({_ref(inner, r) for r in c.path}),
                sorted({_ref(inner, r) for r in c.effect}),
                None if c.after is None else _ref(inner, c.after),
                c.site,
            )
            m.attrs["scope"] = scope + list(c.attrs.get("scope", []))
            if c.kind in ("call", "source", "enter"):
                m.path = sorted(set(m.path) | {eid})
            inner[c.id] = add(m)
        assert result is not None
        exit_ = Node(-1, "exit", [result], {"cell": callee.cell, "scope": scope}, [eid])
        done[n.id] = add(exit_)
        issued[n.id] = eid
    return Graph(graph.cell, graph.code, list(graph.params), new)


def _map(i: Input, inner: Mapping[int, Input]) -> Input:
    return inner[i] if isinstance(i, int) else i


def _ref(inner: Mapping[int, Input], r: int) -> int:
    out = inner[r]
    assert isinstance(out, int), "control edges refer to nodes, never parameters"
    return out
