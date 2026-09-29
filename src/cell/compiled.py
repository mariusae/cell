"""Compiled execution: running a graph as dataflow (DESIGN §2, §6, milestone M2).

A graph is compiled into a `Plan` by resolving the cells and ops it refers
to, by id and code hash. Running a plan evaluates its nodes as soon as
their dependencies allow, rather than in program order:

- a node starts once every node it refers to (data inputs, `path` and
  `effect` edges) has completed successfully, and its `after` node has
  been issued;
- a call to a *pure* cell may also start when an input call has only been
  issued: the handle is passed and resolves before the callee starts
  (DESIGN §1.5). Effectful calls never do this: edge reduction leaves out
  a path edge to anything a data input implies, which is only sound if
  data inputs have completed, and an effectful call must only be issued
  where eager execution would issue it;
- pure nodes therefore run as early as their inputs allow, including
  speculatively ahead of guards; effectful calls wait for their edges.

Calls and sources are issued through the invocation at the `seq` the graph
recorded, so the journal is keyed exactly as eager execution would key it.

Any failure deopts: a guard that doesn't hold, a deopt node, a call or op
that raises. The run stops issuing nodes and returns `Deopt`, and the
runtime replays the invocation eagerly from its journal (DESIGN §6.3).
Errors therefore always come from eager execution, which is what defines
them. A compiled run succeeds only if every node ran and every call it
issued completed successfully.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from . import data
from .core import Cell, Op, find_cell, find_op
from .errors import CellError
from .graph import BUILTINS, Const, Graph, Input, Node
from .journal import Entry, Ok

if TYPE_CHECKING:
    from .context import Handle
    from .runtime import Invocation


class CompileError(CellError):
    """A graph refers to a cell or op whose code isn't loaded."""


@dataclass(frozen=True)
class Deopt:
    reason: str


@dataclass(eq=False)
class Plan:
    """A graph with its cells and ops resolved, ready to run."""

    graph: Graph
    cell: Cell
    cells: dict[int, Cell]  # call node -> callee
    ops: dict[int, Op]  # op node -> op, for ops that aren't builtins
    runs: int = 0
    deopts: Counter[str] = field(default_factory=Counter)  # by reason (DESIGN §8)
    fail_at: int | None = None  # for tests: deopt when this node is about to start


def compile_graph(graph: Graph) -> Plan:
    cell = find_cell(graph.cell, graph.code)
    if cell is None:
        raise CompileError(f"no loaded cell {graph.cell} with code {graph.code}")
    cells: dict[int, Cell] = {}
    ops: dict[int, Op] = {}
    for n in graph.nodes:
        a = n.attrs
        if n.kind in ("call", "enter"):
            callee = find_cell(a["cell"], a["code"])
            if callee is None:
                raise CompileError(f"%{n.id}: no loaded cell {a['cell']} with code {a['code']}")
            cells[n.id] = callee
        elif n.kind == "op" and not a.get("builtin"):
            fn = find_op(a["op"], a["code"])
            if fn is None:
                raise CompileError(f"%{n.id}: no loaded op {a['op']} with code {a['code']}")
            ops[n.id] = fn
    return Plan(graph, cell, cells, ops)


async def run_plan(plan: Plan, inv: Invocation, arguments: dict[str, Any]) -> Ok | Deopt:
    """Run a plan as an invocation. The caller replays eagerly on Deopt."""
    plan.runs += 1
    run = _Run(plan, inv, arguments)
    outcome = await run.run()
    if isinstance(outcome, Deopt):
        plan.deopts[outcome.reason] += 1
    return outcome


class _Run:
    def __init__(self, plan: Plan, inv: Invocation, arguments: dict[str, Any]):
        self.plan = plan
        self.inv = inv
        self.arguments = arguments
        self.nodes = plan.graph.nodes
        self.done: set[int] = set()  # completed successfully
        self.values: dict[int, Any] = {}
        self.issued: dict[int, Handle[Any]] = {}
        self.started: set[int] = set()
        self.tasks: dict[asyncio.Task[Any], int] = {}
        self.result: Ok | None = None
        self.deopt: str | None = None
        # Inlined callees (DESIGN §5.7): an invocation per scope, and the
        # entry each one's call has in its parent's journal.
        self.scopes: dict[tuple[int, ...], Invocation] = {(): inv}
        self.entries: dict[tuple[int, ...], Entry] = {}

    async def run(self) -> Ok | Deopt:
        try:
            return await self._run()
        finally:
            for scope, sub in self.scopes.items():
                if scope:
                    # Calls inlined callees left in flight complete before the
                    # invocation does, like the invocation's own.
                    sub.finished = True
                    self.inv.children.extend(sub.children)

    async def _run(self) -> Ok | Deopt:
        while True:
            self._start_ready()
            if self.deopt is not None:
                return Deopt(self.deopt)
            if not self.tasks:
                if self.result is not None and len(self.started) == len(self.nodes):
                    return self.result
                return Deopt("stalled: nodes that can never start")
            # Every call issued must complete, even if nothing uses its
            # result: structured concurrency (DESIGN §1.4, rule 3).
            finished, _ = await asyncio.wait(self.tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in finished:
                nid = self.tasks.pop(task)
                if task.cancelled():
                    self._fail(f"%{nid} was cancelled")
                elif (e := task.exception()) is not None:
                    self._fail(f"%{nid} {_short(self.nodes[nid].attrs['cell'])} raised {type(e).__name__}")
                else:
                    self._complete(nid, task.result())

    def _start_ready(self) -> None:
        progress = True
        while progress and self.deopt is None:
            progress = False
            for n in self.nodes:
                if n.id in self.started or not self._ready(n):
                    continue
                self.started.add(n.id)
                self._start(n)
                progress = True
                if self.deopt is not None:
                    return

    def _ready(self, n: Node) -> bool:
        if any(r not in self.done for r in n.path) or any(r not in self.done for r in n.effect):
            return False
        if n.after is not None and n.after not in self.issued:
            return False
        if n.kind == "exit":
            # A callee completes only when everything it issued has.
            scope = n.scope
            if any(self.nodes[t].scope[: len(scope)] == scope for t in self.tasks.values()):
                return False
        for i in n.inputs:
            if isinstance(i, int) and i not in self.done:
                # A pure call can take an input call's handle before it completes.
                if not (n.kind == "call" and not n.effectful and i in self.issued):
                    return False
        return True

    def _complete(self, nid: int, value: Any = None) -> None:
        self.values[nid] = value
        self.done.add(nid)

    def _fail(self, reason: str) -> None:
        if self.deopt is None:
            self.deopt = reason

    def _value(self, i: Input) -> Any:
        return i.value if isinstance(i, Const) else self.values[i]

    def _start(self, n: Node) -> None:
        if n.id == self.plan.fail_at:
            self._fail(f"forced deopt at %{n.id}")
            return
        a = n.attrs
        kind = n.kind
        try:
            if kind == "param":
                self._complete(n.id, self.arguments[a["name"]])
            elif kind == "op":
                self._complete(n.id, self._op(n))
            elif kind == "pack":
                self._complete(n.id, data.unflatten(a["tree"], [self._value(i) for i in n.inputs]))
            elif kind == "source":
                self._complete(n.id, self.scopes[n.scope].source(a["name"], a["args"], seq=a["seq"]))
            elif kind == "guard":
                value = self._value(n.inputs[0])
                if data.digest(value) != data.digest(a["expected"]):
                    self._fail(f"guard %{n.id} failed")
                else:
                    self._complete(n.id)
            elif kind == "call":
                self._call(n)
            elif kind == "enter":
                self._enter(n)
            elif kind == "exit":
                self._exit(n)
            elif kind == "deopt":
                self._fail(a["reason"])
            elif kind == "return":
                self.result = Ok(self._value(n.inputs[0]))
                self._complete(n.id)
            else:
                raise AssertionError(kind)
        except Exception as e:
            self._fail(f"%{n.id} {kind} raised {type(e).__name__}")

    def _op(self, n: Node) -> Any:
        a = n.attrs
        args = [self._value(i) for i in n.inputs]
        if a.get("builtin"):
            if a["op"] == "method":
                return BUILTINS["method"](args[0], a["method"], *args[1:])
            return BUILTINS[a["op"]](*args)
        names = a.get("kwargs", [])
        split = len(args) - len(names)
        return self.plan.ops[n.id].fn(*args[:split], **dict(zip(names, args[split:])))

    def _call(self, n: Node) -> None:
        a = n.attrs
        arguments = {}
        for param, i in zip(a["params"], n.inputs):
            if isinstance(i, int) and i not in self.done:
                arguments[param] = self.issued[i]  # pushdown: resolved before the callee starts
            else:
                arguments[param] = self._value(i)
        inv = self.scopes[n.scope]
        handle = inv.call(self.plan.cells[n.id], arguments, seq=a["seq"], domain=a["domain"])
        self.issued[n.id] = handle
        self.tasks[handle._task] = n.id

    def _enter(self, n: Node) -> None:
        """Issue an inlined call: an entry in the parent's journal, and the callee's invocation."""
        a = n.attrs
        parent = self.scopes[n.scope]
        cell = self.plan.cells[n.id]
        arguments = {p: self._value(i) for p, i in zip(a["params"], n.inputs)}
        entry = Entry(seq=a["seq"], kind="call", target=cell.id, effectful=cell.effectful, domain=a["domain"])
        entry.args = arguments
        entry.args_digest = data.digest(arguments)
        entry.started = True
        parent.journal.entries.append(entry)
        hit = parent.lookup(entry)  # when the parent is replaying, so does the callee
        sub = type(parent)(
            parent.runtime, cell, arguments, parent.journal.path + (a["seq"],), parent.journal.request_id,
            hit.child if hit is not None else None, None,
        )
        sub.task = parent.task
        sub.journal.mode = "compiled"
        entry.child = sub.journal
        scope = (*n.scope, a["seq"])
        self.scopes[scope] = sub
        self.entries[scope] = entry
        self.issued[n.id] = None  # type: ignore[assignment]
        self._complete(n.id)

    def _exit(self, n: Node) -> None:
        scope = n.scope
        sub, entry = self.scopes[scope], self.entries[scope]
        result = Ok(self._value(n.inputs[0]))
        sub.journal.entries.sort(key=lambda e: e.seq)
        sub.journal.outcome = result
        sub.finished = True
        entry.outcome = result
        self._complete(n.id, result.value)


def _short(cell_id: str) -> str:
    return cell_id.rsplit(".", 1)[-1]
