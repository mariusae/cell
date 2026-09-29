"""The tracer (DESIGN §5, milestone M1).

Tracing runs a real request eagerly, with the root cell's values wrapped
so that what the body does with them is recorded as a graph. Calls really
execute, through the ordinary runtime, so every tracer holds its concrete
value: the tracer never guesses, it only decides what to record.

- Parameters, call results and ctx sources become `Tracer`s.
- Calls return `TracedHandle`s; awaiting one yields a tracer.
- Operations on tracers become `op` nodes; `@op` calls become one node.
- Anything that needs a concrete value to decide control flow becomes a
  `guard` (truth, lengths, and values that are None, bool or an Enum).
- Anything else that would expose a value (str(), hashing, isinstance,
  mutation, handing it to unknown code) is a graph break: a `deopt` node.

At a break the body is aborted and the request is finished by replaying
it eagerly from the journal of what already ran (DESIGN §5.5), so effects
happen exactly once and the request's outcome is exactly eager's.

Only the root cell's body is traced. The cells it calls run eagerly, and
each is a single call node in the graph (DESIGN §5.7).
"""

from __future__ import annotations

import enum
import os
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, NoReturn

from . import data
from .context import Ctx, Handle
from .core import Cell, Op
from .errors import CellError, DataError
from .graph import BUILTINS, Const, Graph, Input, Node
from .runtime import Invocation, Run, Runtime, _Interrupt
from .semantics import UNIQUE, Domain

DEFAULT_MAX_GUARDS = 64


class GraphBreak(_Interrupt):
    """Aborts a traced body at a graph break (DESIGN §5.5)."""


class TraceError(CellError):
    """A traced value was used after its trace finished."""


@dataclass(frozen=True)
class Traced:
    graph: Graph
    run: Run  # the request's outcome, exactly as eager execution would produce it
    replayed: bool  # the body broke, and the request was finished by replay


async def trace(
    runtime: Runtime,
    cell: Cell,
    args: tuple[Any, ...] = (),
    kwargs: Mapping[str, Any] | None = None,
    *,
    request_id: str | None = None,
    max_guards: int = DEFAULT_MAX_GUARDS,
) -> Traced:
    """Run a cell as a request, recording its graph."""
    if not isinstance(cell, Cell):
        raise TypeError(f"{cell!r} is not a cell")
    arguments = cell.bind(tuple(args), dict(kwargs or {}))
    data.check(arguments)
    request_id = request_id or runtime.new_request_id()
    rec = Recorder(cell, max_guards)
    inv = await runtime.root(cell, arguments, request_id, make_ctx=rec.make_ctx, body=rec.body)
    rec.active = False
    if rec.broken or inv.interrupted:
        run = await runtime.execute(cell, (), arguments, request_id=request_id, replay=inv.journal)
        return Traced(rec.graph, run, True)
    return Traced(rec.graph, runtime.run_of(inv), False)


# Traced values


class Tracer:
    """A value produced in a traced body, and the node that produced it."""

    __slots__ = ("_t_value", "_t_node", "_t_rec")

    def __init__(self, value: Any, node: int, rec: Recorder):
        object.__setattr__(self, "_t_value", value)
        object.__setattr__(self, "_t_node", node)
        object.__setattr__(self, "_t_rec", rec)

    # Recorded as ops.

    def __getattr__(self, name: str) -> Any:
        return self._t_rec.attribute(self, name)

    def __getitem__(self, key: Any) -> Any:
        return self._t_rec.builtin("getitem", self, key)

    # Concretized, with a guard.

    def __bool__(self) -> bool:
        return self._t_rec.builtin("truth", self)

    def __len__(self) -> int:
        return self._t_rec.length(self)

    def __iter__(self) -> Iterator[Any]:
        return self._t_rec.iterate(self)

    def __contains__(self, item: Any) -> bool:
        return self._t_rec.builtin("contains", self, item)

    # Graph breaks.

    def __setattr__(self, name: str, value: Any) -> None:
        self._t_rec.break_("assigning to an attribute of a traced value (data is immutable)")

    def __delattr__(self, name: str) -> None:
        self._t_rec.break_("deleting an attribute of a traced value (data is immutable)")

    def __setitem__(self, key: Any, value: Any) -> None:
        self._t_rec.break_("assigning to an item of a traced value (data is immutable)")

    def __delitem__(self, key: Any) -> None:
        self._t_rec.break_("deleting an item of a traced value (data is immutable)")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._t_rec.break_("calling a traced value")

    def __repr__(self) -> str:
        rec = self._t_rec
        if rec.active and not rec.broken:
            rec.break_("repr() of a traced value; use an @op")
        return f"<tracer %{self._t_node}>"

    @property  # type: ignore[misc]
    def __class__(self) -> type:
        # isinstance() consults __class__, so this catches type tests.
        return self._t_rec.type_of(self)


def _binary(name: str) -> Any:
    def method(self: Tracer, other: Any) -> Any:
        return self._t_rec.builtin(name, self, other)

    return method


def _reflected(name: str) -> Any:
    def method(self: Tracer, other: Any) -> Any:
        return self._t_rec.builtin(name, other, self)

    return method


def _unary(name: str) -> Any:
    def method(self: Tracer) -> Any:
        return self._t_rec.builtin(name, self)

    return method


def _breaks(what: str) -> Any:
    def method(self: Tracer, *args: Any) -> Any:
        self._t_rec.break_(f"{what} of a traced value; use an @op")

    return method


for _name in ("add", "sub", "mul", "truediv", "floordiv", "mod", "pow", "lshift", "rshift", "and_", "or_", "xor"):
    _dunder = _name.rstrip("_")
    setattr(Tracer, f"__{_dunder}__", _binary(_name))
    setattr(Tracer, f"__r{_dunder}__", _reflected(_name))
for _name in ("eq", "ne", "lt", "le", "gt", "ge"):
    setattr(Tracer, f"__{_name}__", _binary(_name))
for _name in ("neg", "pos", "invert", "abs"):
    setattr(Tracer, f"__{_name}__", _unary(_name))
for _dunder, _what in {
    "hash": "hash()",
    "index": "using as an index",
    "int": "int()",
    "float": "float()",
    "complex": "complex()",
    "bytes": "bytes()",
    "str": "str()",
    "format": "formatting",
    "round": "round()",
    "trunc": "trunc()",
    "floor": "floor()",
    "ceil": "ceil()",
    "reversed": "reversed()",
}.items():
    setattr(Tracer, f"__{_dunder}__", _breaks(_what))


class TracedHandle:
    """A call issued from a traced body. Awaiting it yields a tracer."""

    __slots__ = ("_handle", "_node", "_rec", "_returns_none", "_result", "_done")

    def __init__(self, handle: Handle[Any], node: int, rec: Recorder, returns_none: bool):
        self._handle = handle
        self._node = node
        self._rec = rec
        self._returns_none = returns_none
        self._result: Any = None
        self._done = False

    def __await__(self) -> Any:
        rec = self._rec
        rec.observe(self._node)
        if not self._done:
            try:
                value = yield from self._handle.__await__()
            except Exception as e:
                rec.break_(f"%{self._node} failed while tracing ({type(e).__name__})")
            self._result = None if self._returns_none else rec.wrap(value, self._node)
            self._done = True
        return self._result

    def __bool__(self) -> bool:
        raise TypeError("a handle has no truth value; await it first")

    def __repr__(self) -> str:
        return f"<traced handle %{self._node}>"


class _Method:
    """A method of a traced str, tuple, dict, ...; calling it records an op."""

    __slots__ = ("_tracer", "_name")

    def __init__(self, tracer: Tracer, name: str):
        self._tracer = tracer
        self._name = name

    def __call__(self, *args: Any) -> Any:
        return self._tracer._t_rec.builtin("method", self._tracer, *args, method=self._name)


_METHODS: dict[type, frozenset[str]] = {
    str: frozenset(
        "capitalize casefold count endswith find index isalnum isalpha isdigit islower isspace "
        "isupper join lower lstrip partition removeprefix removesuffix replace rfind rpartition "
        "rsplit rstrip split splitlines startswith strip swapcase title upper zfill".split()
    ),
    bytes: frozenset("count decode endswith find hex index lower split startswith strip upper".split()),
    tuple: frozenset({"count", "index"}),
    list: frozenset({"count", "index"}),
    dict: frozenset({"get"}),
    int: frozenset({"bit_length"}),
    float: frozenset({"is_integer"}),
}
"""Pure methods of immutable (or, for data, value-like) builtins (DESIGN §5.2)."""


def _is_traced(value: Any) -> bool:
    t = type(value)
    return t is Tracer or t is TracedHandle


def _has_field(value: Any, name: str) -> bool:
    fields = getattr(type(value), "__dataclass_fields__", None)
    if fields is not None:
        return name in fields
    return isinstance(value, tuple) and name in getattr(type(value), "_fields", ())


def recorder_of(obj: Any) -> Recorder | None:
    """The recorder of any traced value inside obj. Used by @op to notice tracing."""
    t = type(obj)
    if t is Tracer:
        return obj._t_rec
    if t is TracedHandle:
        return obj._rec
    if t is dict:
        obj = obj.values()
    elif hasattr(t, "__dataclass_fields__"):
        obj = [getattr(obj, f) for f in t.__dataclass_fields__]
    elif not isinstance(obj, (tuple, list)):
        return None
    for x in obj:
        rec = recorder_of(x)
        if rec is not None:
            return rec
    return None


class TracingCtx(Ctx):
    """The ctx of a traced body: records calls and sources as nodes."""

    __slots__ = ("_rec",)

    def __init__(self, invocation: Invocation, rec: Recorder):
        super().__init__(invocation)
        self._rec = rec

    @property
    def request_id(self) -> str:
        self._rec.break_("reading ctx.request_id in a traced cell")

    @property
    def path(self) -> tuple[int, ...]:
        self._rec.break_("reading ctx.path in a traced cell")

    def now(self) -> float:
        return self._rec.source("now", {}, super().now())

    def random(self) -> float:
        return self._rec.source("random", {}, super().random())

    def config(self, key: str, default: Any = None) -> Any:
        if recorder_of((key, default)) is not None:
            self._rec.break_("a traced value as a config key or default")
        return self._rec.source("config", {"key": key, "default": default}, super().config(key, default))

    def resource(self, key: Any) -> Any:
        self._rec.break_("ctx.resource in a traced cell (only leaf cells may use resources)")

    def _call(self, cell: Cell, arguments: dict[str, Any]) -> Any:
        return self._rec.call(cell, arguments)


# The recorder


class Recorder:
    """Builds one graph while a body runs."""

    def __init__(self, cell: Cell, max_guards: int = DEFAULT_MAX_GUARDS):
        self.cell = cell
        self.graph = Graph(cell.id, cell.code_hash, [])
        self.inv: Invocation | None = None
        self.active = True
        self.broken = False
        self.max_guards = max_guards
        self.num_guards = 0
        self.observed: list[int] = []  # in program order (DESIGN §4.3)
        self._observed: set[int] = set()
        self._implied: list[frozenset[int]] = []  # per node: what its completion implies
        self._last_effect: dict[Domain, int] = {}
        self._packs: dict[Any, int] = {}  # identical packs share a node

    def make_ctx(self, inv: Invocation) -> Ctx:
        self.inv = inv
        return TracingCtx(inv, self)

    async def body(self, ctx: Ctx, arguments: dict[str, Any]) -> Any:
        traced = {}
        for i, (name, value) in enumerate(arguments.items()):
            self.graph.params.append(name)
            traced[name] = self.wrap(value, self.add("param", attrs={"name": name, "index": i}))
        try:
            result = await self.cell.fn(ctx, **traced)
        except GraphBreak:
            raise
        except Exception as e:
            # The error may be genuine, or caused by a tracer reaching code that
            # can't handle it. Either way, replay gives the true outcome.
            self.break_(f"the body raised {type(e).__name__} while tracing")
        self.add("return", [self.input_of(result)])
        return self.concrete(result)

    # Nodes and edges

    def check(self) -> None:
        if self.broken:
            raise GraphBreak()
        if not self.active:
            raise TraceError("a traced value was used after its trace finished")

    def add(
        self,
        kind: str,
        inputs: list[Input] | None = None,
        attrs: dict[str, Any] | None = None,
        *,
        path: list[int] | None = None,
        effect: list[int] | None = None,
        after: int | None = None,
    ) -> int:
        self.check()
        node = Node(len(self.graph.nodes), kind, inputs or [], attrs or {}, path or [], effect or [], after, _site())
        implied: set[int] = set()
        for d in node.refs:
            implied.add(d)
            implied |= self._implied[d]
        self.graph.nodes.append(node)
        self._implied.append(frozenset(implied))
        return node.id

    def observe(self, node: int) -> None:
        self.check()
        if node not in self._observed:
            self._observed.add(node)
            self.observed.append(node)

    def _reduce(self, nodes: list[int], implied: set[int]) -> list[int]:
        """The nodes not already implied, without any implied by another."""
        xs = [x for x in nodes if x not in implied]
        return sorted(x for x in xs if not any(x in self._implied[y] for y in xs if y != x))

    def _control(self, inputs: list[Input], domain: Domain | None) -> tuple[list[int], list[int], int | None]:
        """Effect edges, path edges and the issue-order edge for an effectful call (DESIGN §4.3)."""
        implied: set[int] = set()
        for i in inputs:
            if isinstance(i, int):
                implied |= {i} | self._implied[i]
        nodes = self.graph.nodes
        effect: list[int] = []
        after = None
        if domain is not UNIQUE:
            same = [o for o in self.observed if nodes[o].effectful and nodes[o].attrs["domain"] == domain]
            effect = self._reduce(same, implied)
            for e in effect:
                implied |= {e} | self._implied[e]
            prev = self._last_effect.get(domain)  # type: ignore[arg-type]
            if prev is not None and prev not in implied:
                after = prev
        pure = [
            o for o in self.observed
            if nodes[o].kind in ("op", "guard") or (nodes[o].kind == "call" and not nodes[o].effectful)
        ]
        return effect, self._reduce(pure, implied), after

    def break_(self, reason: str) -> NoReturn:
        """Record a graph break and abort the body (DESIGN §5.5)."""
        self.check()
        self.add("deopt", attrs={"reason": reason}, path=self._reduce(self.observed, set()))
        self.broken = True
        if self.inv is not None:
            self.inv.stop()  # nothing more may be issued, even from finally blocks
        raise GraphBreak(reason)

    # Values

    def wrap(self, value: Any, node: int) -> Any:
        """The value user code sees for a node.

        None, bools and Enum members are returned concretely, with a guard:
        code compares them with `is`, which a tracer can't intercept.
        """
        if value is None or type(value) is bool or isinstance(value, enum.Enum):
            self.guard(node, value)
            return value
        return Tracer(value, node, self)

    def guard(self, node: int, expected: Any) -> None:
        self.num_guards += 1
        if self.num_guards > self.max_guards:
            self.break_(f"more than {self.max_guards} guards; use an @op")
        self.observe(self.add("guard", [node], {"expected": expected}))

    def input_of(self, obj: Any, *, handles: bool = False) -> Input:
        """The operand for obj: a node, a constant, or a pack of both (DESIGN §5.3)."""
        try:
            leaves, tree = data.flatten(obj, is_leaf=_is_traced)
        except DataError as e:
            self.break_(f"a value that is not data used with traced values ({e})")
        refs: list[Input] = []
        for leaf in leaves:
            t = type(leaf)
            if t is Tracer:
                refs.append(leaf._t_node)
            elif t is TracedHandle:
                if not handles:
                    self.break_("an unawaited handle where a value is needed; await it first")
                refs.append(leaf._node)  # passed, not observed (DESIGN §1.5)
            else:
                refs.append(Const(leaf))
        if tree.kind == "leaf":
            return refs[0]
        if all(isinstance(r, Const) for r in refs):
            return Const(obj)
        key = (tree, tuple(r if isinstance(r, int) else data.digest(r.value) for r in refs))
        if key not in self._packs:
            self._packs[key] = self.add("pack", refs, {"tree": tree})
        return self._packs[key]

    def concrete(self, obj: Any) -> Any:
        """obj with tracers replaced by their values, and traced handles by real ones."""
        leaves, tree = data.flatten(obj, is_leaf=_is_traced)
        if not any(_is_traced(leaf) for leaf in leaves):
            return obj
        values = []
        for leaf in leaves:
            t = type(leaf)
            values.append(leaf._t_value if t is Tracer else leaf._handle if t is TracedHandle else leaf)
        return data.unflatten(tree, values)

    def _result(self, inputs: list[Input], attrs: dict[str, Any], result: Any, what: str) -> Any:
        try:
            data.check(result)
        except DataError:
            self.break_(f"{what} returned a value that is not data")
        node = self.add("op", inputs, attrs)
        self.observe(node)
        return self.wrap(result, node)

    # Operations

    def builtin(self, name: str, *operands: Any, method: str | None = None) -> Any:
        self.check()
        inputs = [self.input_of(o) for o in operands]
        values = [self.concrete(o) for o in operands]
        what = f".{method}()" if method else name
        try:
            if method is None:
                result = BUILTINS[name](*values)
            else:
                result = BUILTINS[name](values[0], method, *values[1:])
        except Exception as e:
            self.break_(f"{what} raised {type(e).__name__} while tracing")
        attrs: dict[str, Any] = {"op": name, "builtin": True}
        if method is not None:
            attrs["method"] = method
        return self._result(inputs, attrs, result, what)

    def op(self, op: Op, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        self.check()
        inputs = [self.input_of(a) for a in (*args, *kwargs.values())]
        cargs = [self.concrete(a) for a in args]
        ckwargs = {k: self.concrete(v) for k, v in kwargs.items()}
        name = op.id.rsplit(".", 1)[-1]
        try:
            result = op.fn(*cargs, **ckwargs)
        except Exception as e:
            self.break_(f"op {name} raised {type(e).__name__} while tracing")
        attrs: dict[str, Any] = {"op": op.id, "code": op.code_hash}
        if kwargs:
            attrs["kwargs"] = list(kwargs)
        return self._result(inputs, attrs, result, f"op {name}")

    def attribute(self, tracer: Tracer, name: str) -> Any:
        value = tracer._t_value
        if _has_field(value, name):
            return self.builtin("getattr", tracer, name)
        if name in _METHODS.get(type(value), ()):
            self.check()
            return _Method(tracer, name)
        self.break_(f"attribute {name!r} of a traced {type(value).__name__}")

    def length(self, tracer: Tracer) -> int:
        n = self.builtin("len", tracer)
        self.guard(n._t_node, n._t_value)
        return n._t_value

    def iterate(self, tracer: Tracer) -> Iterator[Any]:
        value = tracer._t_value
        if not isinstance(value, (tuple, list)):
            self.break_(f"iterating over a traced {type(value).__name__}")
        n = self.length(tracer)
        return (self.builtin("getitem", tracer, i) for i in range(n))

    def type_of(self, tracer: Tracer) -> type:
        if self.active and not self.broken:
            self.break_("the type of a traced value (isinstance or __class__)")
        return Tracer

    # Calls and sources

    def call(self, cell: Cell, arguments: dict[str, Any]) -> TracedHandle:
        self.check()
        assert self.inv is not None
        inputs = [self.input_of(v, handles=True) for v in arguments.values()]
        handle = self.inv.call(cell, {k: self.concrete(v) for k, v in arguments.items()})
        entry = handle._entry
        attrs = {
            "cell": cell.id,
            "code": cell.code_hash,
            "effectful": cell.effectful,
            "domain": entry.domain,
            "seq": entry.seq,
            "params": list(arguments),
        }
        effect: list[int] = []
        path: list[int] = []
        after = None
        if cell.effectful:
            effect, path, after = self._control(inputs, entry.domain)
        node = self.add("call", inputs, attrs, path=path, effect=effect, after=after)
        if cell.effectful and entry.domain is not UNIQUE:
            self._last_effect[entry.domain] = node  # type: ignore[index]
        return TracedHandle(handle, node, self, cell.returns_none)

    def source(self, name: str, args: dict[str, Any], value: Any) -> Any:
        assert self.inv is not None
        seq = self.inv.journal.entries[-1].seq
        return self.wrap(value, self.add("source", attrs={"name": name, "args": args, "seq": seq}))


_HERE = os.path.dirname(os.path.abspath(__file__))
_ASYNCIO = os.path.dirname(os.path.abspath(__import__("asyncio").__file__))


def _site() -> str | None:
    """file:line of the user code that caused the current node.

    None when there is no such code on the stack: parameters and the
    result are recorded by the tracer itself, and awaits inside
    asyncio.gather run in asyncio's own tasks.
    """
    f = sys._getframe(1)
    while f is not None and f.f_code.co_filename.startswith(_HERE):
        f = f.f_back
    if f is None or f.f_code.co_filename.startswith(_ASYNCIO):
        return None
    return f"{os.path.basename(f.f_code.co_filename)}:{f.f_lineno}"
