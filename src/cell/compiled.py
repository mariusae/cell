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

Batches (milestone M5). A run is a batch of *lanes*, one per request, all
running the same plan; a single request is a batch of one. Each lane keeps
its own state, invocation and journal, and deopts on its own: a guard that
fails in one lane partitions it out, and the rest go on. What lanes share
is vector calls. A pure call to a cell with a vector form (DESIGN §10), or
a ctx.map over one, is not issued when it's ready: it waits until every
lane still running has reached it, and then all such calls to the same
cell, across nodes and lanes, go out as one vector call. The results are
split back, and each lane journals its own call, marked `batched`. If the
vector call fails, the calls go out one by one, so errors are attributed.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from . import data
from .context import Handle
from .core import Cell, Op, find_cell, find_op
from .errors import CellError
from .graph import BUILTINS, Const, Graph, Input, Node
from .journal import Entry, Ok
from .mapping import MAP_CELLS, map_target

if TYPE_CHECKING:
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
    return (await run_batch(plan, [inv], [arguments]))[0]


async def run_batch(
    plan: Plan,
    invs: list[Invocation],
    arguments: list[dict[str, Any]],
    on_outcome: Callable[[int, Ok | Deopt], None] | None = None,
) -> list[Ok | Deopt]:
    """Run a plan as a batch of invocations, one lane each. The caller replays each lane that deopts.

    `on_outcome(i, outcome)` is called as soon as lane i is done, before the rest of the batch.
    """

    def count(i: int, outcome: Ok | Deopt) -> None:
        plan.runs += 1
        if isinstance(outcome, Deopt):
            plan.deopts[outcome.reason] += 1
        if on_outcome is not None:
            on_outcome(i, outcome)

    batch = _Batch(plan, [_Run(plan, inv, args) for inv, args in zip(invs, arguments)], count)
    return await batch.run()


class _Run:
    """One lane: a request running the plan."""

    def __init__(self, plan: Plan, inv: Invocation, arguments: dict[str, Any]):
        self.plan = plan
        self.inv = inv
        self.arguments = arguments
        self.batch: _Batch | None = None
        self.deferred: dict[int, dict[str, Any]] = {}  # node -> arguments, waiting for a vector call
        self.vectored: set[int] = set()  # nodes whose calls are in flight in a vector call
        self.closed = False
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

    @property
    def finished(self) -> bool:
        # Every call issued must complete, even if nothing uses its result:
        # structured concurrency (DESIGN §1.4, rule 3).
        return (
            self.deopt is None
            and self.result is not None
            and not self.tasks
            and not self.deferred
            and not self.vectored
            and len(self.started) == len(self.nodes)
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for scope, sub in self.scopes.items():
            if scope:
                # Calls inlined callees left in flight complete before the
                # invocation does, like the invocation's own.
                sub.finished = True
                self.inv.children.extend(sub.children)

    def task_done(self, task: asyncio.Task[Any]) -> None:
        nid = self.tasks.pop(task)
        if task.cancelled():
            self._fail(f"%{nid} was cancelled")
        elif (e := task.exception()) is not None:
            self._fail(f"%{nid} {_short(self.nodes[nid].attrs['cell'])} raised {type(e).__name__}")
        else:
            # The dataflow consumed it: "awaited" in the journal's sense.
            self.issued[nid]._entry.awaited = True
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
            pending = [*self.tasks.values(), *self.deferred, *self.vectored]
            if any(self.nodes[t].scope[: len(scope)] == scope for t in pending):
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
        if self.batch is not None and self.batch.vector_of(self, n, arguments) is not None:
            self.deferred[n.id] = arguments
            return
        self.issue(n, arguments)

    def issue(self, n: Node, arguments: dict[str, Any]) -> None:
        a = n.attrs
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


type _Element = tuple[_Run, int, dict[str, Any], str, int]  # lane, node, arguments, "call" or "map", count


class _Batch:
    """Runs lanes together, and sends their vector-capable calls as vector calls."""

    def __init__(
        self, plan: Plan, lanes: list[_Run], on_outcome: Callable[[int, Ok | Deopt], None] | None = None
    ):
        self.plan = plan
        self.lanes = lanes
        self.on_outcome = on_outcome
        self.runtime = lanes[0].inv.runtime
        for lane in lanes:
            lane.batch = self
        self.outcomes: dict[int, Ok | Deopt] = {}
        self.vector_tasks: dict[asyncio.Task[Any], list[_Element]] = {}

    def vector_of(self, lane: _Run, n: Node, arguments: dict[str, Any]) -> tuple[Cell, str] | None:
        """The vector form a call can go out as, and how: as one element ("call"), or one per item ("map")."""
        if n.kind != "call" or n.effectful:
            return None
        if lane.scopes[n.scope]._replay is not None:
            return None  # replaying (validation): calls are answered from the journal
        leaves, _ = data.flatten(arguments, is_leaf=lambda v: isinstance(v, Handle))
        if any(isinstance(v, Handle) for v in leaves):
            return None  # a handle still resolving: issue the call, and let it resolve
        cell = self.plan.cells[n.id]
        if cell.id in MAP_CELLS:
            target = map_target(arguments)
            if target is None or target.vector is None or self.runtime.caches(target):
                return None
            return target.vector, "map"
        if cell.vector is None or self.runtime.caches(cell):
            return None  # let the cache answer instead
        return cell.vector, "call"

    def _active(self) -> list[_Run]:
        return [lane for i, lane in enumerate(self.lanes) if i not in self.outcomes]

    def _resolve(self, i: int, outcome: Ok | Deopt) -> None:
        """A lane is done: report it now, not when the whole batch is.

        Otherwise every request in a batch would wait for its slowest lane,
        and a lane that deopted would wait to start its replay.
        """
        self.outcomes[i] = outcome
        self.lanes[i].close()
        if self.on_outcome is not None:
            self.on_outcome(i, outcome)

    async def run(self) -> list[Ok | Deopt]:
        try:
            while True:
                for i, lane in enumerate(self.lanes):
                    if i in self.outcomes:
                        continue
                    lane._start_ready()
                    if lane.deopt is not None:
                        self._resolve(i, Deopt(lane.deopt))
                if self._flush():
                    continue  # it completed nodes (maps over nothing): start what they unblock
                for i, lane in enumerate(self.lanes):
                    if i not in self.outcomes and lane.finished:
                        assert lane.result is not None
                        self._resolve(i, lane.result)
                active = self._active()
                if not active:
                    return [self.outcomes[i] for i in range(len(self.lanes))]
                waiting = {t: lane for lane in active for t in lane.tasks}
                if not waiting and not self.vector_tasks:
                    for i in range(len(self.lanes)):
                        if i not in self.outcomes:
                            self._resolve(i, Deopt("stalled: nodes that can never start"))
                    continue
                finished, _ = await asyncio.wait([*waiting, *self.vector_tasks], return_when=asyncio.FIRST_COMPLETED)
                for task in finished:
                    if task in self.vector_tasks:
                        self._vector_done(task)
                    elif task in waiting:
                        waiting[task].task_done(task)
        finally:
            for lane in self.lanes:
                lane.close()

    def _flush(self) -> bool:
        """Send the deferred calls every active lane has reached, grouped by vector cell.

        Returns whether any completed right away.
        """
        completed = False
        active = self._active()
        groups: dict[str, tuple[Cell, list[_Element]]] = {}
        for lane in active:
            for nid, arguments in list(lane.deferred.items()):
                if not all(nid in other.started for other in active):
                    continue  # wait for the other lanes to get here
                found = self.vector_of(lane, self.plan.graph.nodes[nid], arguments)
                assert found is not None
                vector, how = found
                count = len(arguments["items"]) if how == "map" else 1
                del lane.deferred[nid]
                lane.vectored.add(nid)
                groups.setdefault(vector.id, (vector, []))[1].append((lane, nid, arguments, how, count))
        for vector, elements in groups.values():
            n = sum(e[4] for e in elements)
            if len(elements) == 1 and n > 0:
                # Nothing to share: issue the call as it is. (A map still uses
                # the vector form, inside the map cell.)
                lane, nid, arguments, _, _ = elements[0]
                lane.vectored.discard(nid)
                lane.issue(self.plan.graph.nodes[nid], arguments)
                continue
            if n == 0:  # maps over nothing
                for lane, nid, arguments, _, _ in elements:
                    self._deliver(lane, nid, arguments, [])
                completed = True
                continue
            columns: dict[str, list[Any]] = {p: [] for p in vector.params}
            for _, _, arguments, how, count in elements:
                for p in vector.params:
                    if how == "call":
                        columns[p].append(arguments[p])
                    elif p == arguments["param"]:
                        columns[p].extend(arguments["items"])
                    else:
                        columns[p].extend([arguments["fixed"][p]] * count)
            self.runtime.vector_stats["calls"] += 1
            self.runtime.vector_stats["elements"] += n
            request_id = f"vector-{self.runtime.vector_stats['calls']}"
            task = asyncio.create_task(self.runtime._invoke(vector, columns, (), request_id))
            self.vector_tasks[task] = elements
        return completed

    def _vector_done(self, task: asyncio.Task[Any]) -> None:
        elements = self.vector_tasks.pop(task)
        outcome = task.result().journal.outcome
        n = sum(e[4] for e in elements)
        results = outcome.value if isinstance(outcome, Ok) else None
        if not isinstance(results, (list, tuple)) or len(results) != n:
            # Failed, or not one result per element: send the calls one by one.
            self.runtime.vector_stats["fallbacks"] += 1
            for lane, nid, arguments, _, _ in elements:
                lane.vectored.discard(nid)
                if lane.deopt is None:
                    lane.issue(self.plan.graph.nodes[nid], arguments)
            return
        at = 0
        for lane, nid, arguments, how, count in elements:
            value = results[at] if how == "call" else list(results[at : at + count])
            at += count
            self._deliver(lane, nid, arguments, value)

    def _deliver(self, lane: _Run, nid: int, arguments: dict[str, Any], value: Any) -> None:
        lane.vectored.discard(nid)
        if lane.deopt is not None:
            return  # a pure result the lane no longer needs; replay recomputes it
        n = self.plan.graph.nodes[nid]
        entry = Entry(seq=n.attrs["seq"], kind="call", target=n.attrs["cell"], effectful=False)
        entry.args = arguments
        entry.args_digest = data.digest(arguments)
        entry.started = entry.awaited = entry.batched = True
        entry.outcome = Ok(value)
        lane.scopes[n.scope].journal.entries.append(entry)
        lane._complete(nid, value)
