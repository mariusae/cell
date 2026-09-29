"""The runtime: eager execution (M0), and dispatch to compiled graphs (M2).

Cell bodies run as ordinary Python. Each call is issued when it is made,
runs in its own task, and is resolved in-process. Every invocation keeps
a journal (DESIGN §3), and an invocation can be replayed from a journal:
journaled calls return their recorded outcomes instead of running again.

A cell with an installed graph runs from the graph instead (compiled.py).
If the graph's assumptions fail, the invocation deopts: it is replayed
eagerly from the journal of what the compiled run already issued.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from . import data
from .context import Ctx, Handle
from .core import Cell
from .errors import ContextError, DataError, DeterminismError
from .compiled import Plan, compile_graph, run_plan
from .graph import Graph
from .journal import Entry, Err, Journal, Ok, Outcome
from .semantics import Domain, check_domain


type Body = Callable[[Ctx, dict[str, Any]], Awaitable[Any]]


class _Interrupt(BaseException):
    """Stops a root body before it issues a given seq (Runtime.execute(interrupt_at=...))."""


class Invocation:
    """One running cell body: its ctx state and its journal."""

    def __init__(
        self,
        runtime: Runtime,
        cell: Cell,
        arguments: dict[str, Any],
        path: tuple[int, ...],
        request_id: str,
        replay: Journal | None,
        interrupt_at: int | None,
    ):
        self.runtime = runtime
        self.cell = cell
        self.journal = Journal(request_id=request_id, path=path, cell=cell.id, args=arguments)
        self.task: asyncio.Task[Any] | None = None
        self.finished = False
        self.interrupted = False
        self.violation: DeterminismError | None = None
        self.children: list[Handle[Any]] = []
        self.consumed: set[int] = set()
        self._next_seq = 0
        self._interrupt_at = interrupt_at
        self._domains: list[Domain] = []
        self.replay = replay
        self._replay = None if replay is None else {e.seq: e for e in replay.entries}

    # Issuing

    def _check_ctx(self) -> None:
        if self.finished:
            raise ContextError(f"this ctx belongs to an invocation of {self.cell.id} that has finished")
        if asyncio.current_task() is not self.task:
            raise ContextError(
                "calls and ctx sources must be made from the cell body's own task, so that "
                "program order is deterministic; to run calls concurrently, issue them and "
                "await their handles"
            )

    def stop(self) -> None:
        """Refuse all further calls and sources: each raises the interrupt."""
        self._interrupt_at = self._next_seq

    def _issue(self) -> int:
        seq = self._next_seq
        if self._interrupt_at is not None and seq >= self._interrupt_at:
            raise _Interrupt()
        self._next_seq += 1
        return seq

    def call(
        self, cell: Cell, arguments: dict[str, Any], *, seq: int | None = None, domain: Domain | None = None
    ) -> Handle[Any]:
        """Issue a call. Compiled execution passes the seq and domain recorded in the graph."""
        self._check_ctx()
        try:
            data.flatten(arguments, is_leaf=_is_handle)
        except DataError as e:
            raise DataError(f"argument to {cell.id} is not data: {e}") from None
        if seq is None:
            seq = self._issue()
            if cell.effectful:
                domain = self._domains[-1] if self._domains else cell.domain
        elif not cell.effectful:
            domain = None
        entry = Entry(seq=seq, kind="call", target=cell.id, effectful=cell.effectful, domain=domain)
        self.journal.entries.append(entry)
        task = asyncio.create_task(
            self.runtime._run_call(self, entry, cell, arguments),
            name=f"{cell.id}@{self.journal.path + (seq,)}",
        )
        handle: Handle[Any] = Handle(task, entry)
        entry.handle = handle
        self.children.append(handle)
        return handle

    def source(self, name: str, args: dict[str, Any], *, seq: int | None = None) -> Any:
        """A ctx source (now, random, config), journaled at the next seq or the given one."""
        self._check_ctx()
        if seq is None:
            seq = self._issue()
        entry = Entry(seq=seq, kind="source", target=name, args=args, args_digest=data.digest(args))
        self.journal.entries.append(entry)
        hit = self.lookup(entry)
        if hit is not None and isinstance(hit.outcome, Ok):
            value = hit.outcome.value
            entry.replayed = True
        else:
            value = self._compute_source(name, args, seq)
            data.check(value)
        entry.started = True
        entry.awaited = True
        entry.outcome = Ok(value)
        return value

    def _compute_source(self, name: str, args: dict[str, Any], seq: int) -> Any:
        if name == "now":
            return float(self.runtime.clock())
        if name == "random":
            return self.derived_random(seq)
        if name == "config":
            return self.runtime.config.get(args["key"], args["default"])
        raise ValueError(f"unknown ctx source {name!r}")

    def derived_random(self, seq: int) -> float:
        key = f"{self.runtime.seed}|{self.journal.request_id}|{self.journal.path}|{seq}"
        return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64

    def push_domain(self, domain: Domain) -> None:
        self._check_ctx()
        self._domains.append(check_domain(domain))

    def pop_domain(self) -> None:
        self._domains.pop()

    # Replay

    def lookup(self, entry: Entry) -> Entry | None:
        """The journaled entry to replay for `entry`, if there is one (DESIGN §3)."""
        if self._replay is None:
            return None
        old = self._replay.get(entry.seq)
        if old is None:
            return None
        if old.key == entry.key:
            self.consumed.add(entry.seq)
            return old
        if old.effectful and old.started:  # one that never started had no effect
            err = DeterminismError(
                f"replay of {self.cell.id} at {self.journal.path + (entry.seq,)}: the journal has "
                f"effectful {old.target}({old.args_digest}) but the body issued "
                f"{entry.target}({entry.args_digest})"
            )
            self.violation = self.violation or err
            raise err
        return None  # a pure call or source that doesn't match: just execute it


class Runtime:
    """Runs cells eagerly, in-process."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any] | None = None,
        resources: Mapping[Any, Any] | None = None,
        clock: Callable[[], float] | None = None,
        seed: int = 0,
        cache: Mapping[Cell, float] | None = None,
    ):
        self.config = dict(config or {})
        data.check(self.config)
        self._resources = dict(resources or {})
        self.clock = clock or time.time
        self.seed = seed
        self._request_ids = itertools.count(1)
        self._plans: dict[str, Plan] = {}
        self._ttls: dict[str, float] = {}
        self._cache: dict[tuple[str, str, str], tuple[Any, float]] = {}
        self.cache_stats: Counter[str] = Counter()
        for c, ttl in (cache or {}).items():
            self.cache(c, ttl)

    def cache(self, cell: Cell, ttl: float) -> None:
        """Cache the results of calls to a pure cell for `ttl` seconds (NOTES §2).

        Only pure cells can be cached: the policy is checked against the
        cell's declared semantics. Errors are not cached.
        """
        if cell.effectful:
            raise ValueError(f"{cell.id} is {cell.semantics!r}; only pure cells can be cached")
        self._ttls[cell.id] = ttl

    def _cached(self, cell: Cell, digest: str) -> tuple[bool, Any]:
        ttl = self._ttls.get(cell.id)
        if ttl is None:
            return False, None
        found = self._cache.get((cell.id, cell.code_hash, digest))
        if found is not None and self.clock() - found[1] < ttl:
            self.cache_stats["hit"] += 1
            return True, found[0]
        self.cache_stats["miss"] += 1
        return False, None

    def _store(self, cell: Cell, digest: str, outcome: Outcome) -> None:
        if cell.id in self._ttls and isinstance(outcome, Ok):
            self._cache[(cell.id, cell.code_hash, digest)] = (outcome.value, self.clock())

    def resource(self, key: Any) -> Any:
        try:
            return self._resources[key]
        except KeyError:
            raise ContextError(f"no resource {key!r} in this runtime") from None

    async def run(self, cell: Cell, /, *args: Any, **kwargs: Any) -> Run:
        """Run a cell as a new request."""
        return await self.execute(cell, args, kwargs)

    def run_sync(self, cell: Cell, /, *args: Any, **kwargs: Any) -> Run:
        return asyncio.run(self.run(cell, *args, **kwargs))

    async def replay(self, cell: Cell, journal: Journal) -> Run:
        """Run a cell again, returning journaled outcomes for the calls the journal has."""
        if journal.cell != cell.id:
            raise ValueError(f"journal is for {journal.cell}, not {cell.id}")
        return await self.execute(cell, (), journal.args, request_id=journal.request_id, replay=journal)

    async def execute(
        self,
        cell: Cell,
        args: tuple[Any, ...] = (),
        kwargs: Mapping[str, Any] | None = None,
        *,
        request_id: str | None = None,
        replay: Journal | None = None,
        interrupt_at: int | None = None,
    ) -> Run:
        """Run a cell as the root of a request.

        `replay` supplies a journal for the root invocation. `interrupt_at`
        stops the root body just before it issues that seq, leaving a
        partial journal; this is how tests simulate the deopt of M2.
        """
        if not isinstance(cell, Cell):
            raise TypeError(f"{cell!r} is not a cell")
        arguments = cell.bind(tuple(args), dict(kwargs or {}))
        data.check(arguments)
        if request_id is None:
            request_id = replay.request_id if replay is not None else self.new_request_id()
        inv = await self.root(cell, arguments, request_id, replay=replay, interrupt_at=interrupt_at)
        return self.run_of(inv)

    def new_request_id(self) -> str:
        return f"r{next(self._request_ids)}"

    async def root(
        self,
        cell: Cell,
        arguments: dict[str, Any],
        request_id: str,
        *,
        replay: Journal | None = None,
        interrupt_at: int | None = None,
        make_ctx: Callable[[Invocation], Ctx] = Ctx,
        body: Body | None = None,
    ) -> Invocation:
        """Run a root invocation in its own task. The tracer supplies its own ctx and body."""
        return await asyncio.create_task(
            self._invoke(cell, arguments, (), request_id, replay, interrupt_at, make_ctx, body)
        )

    def run_of(self, inv: Invocation) -> Run:
        unconsumed: tuple[Entry, ...] = ()
        if inv.replay is not None:
            unconsumed = tuple(
                e
                for e in inv.replay.entries
                if e.seq not in inv.consumed and e.kind == "call" and e.effectful and e.started
            )
        return Run(inv.journal, inv.interrupted, unconsumed)

    # Compiled execution (DESIGN §2, §6)

    def install(self, graph: Graph) -> Plan:
        """Run invocations of the graph's cell from the graph, whether at the root or nested.

        The graph must be for the current code of its cell. Installing a
        second graph for a cell replaces the first.
        """
        plan = compile_graph(graph)
        self._plans[graph.cell] = plan
        return plan

    def uninstall(self, cell: Cell) -> None:
        self._plans.pop(cell.id, None)

    def plan(self, cell: Cell) -> Plan | None:
        plan = self._plans.get(cell.id)
        return plan if plan is not None and plan.cell is cell else None

    async def _invoke_compiled(
        self,
        plan: Plan,
        cell: Cell,
        arguments: dict[str, Any],
        path: tuple[int, ...],
        request_id: str,
        replay: Journal | None,
    ) -> Invocation:
        inv = Invocation(self, cell, arguments, path, request_id, replay, None)
        inv.task = asyncio.current_task()
        inv.journal.mode = "compiled"
        result = await run_plan(plan, inv, arguments)
        inv.finished = True
        if isinstance(result, Ok):
            inv.journal.entries.sort(key=lambda e: e.seq)
            inv.journal.outcome = result
            return inv
        # Deopt (DESIGN §6.3): replay eagerly from what the compiled attempt
        # issued, including calls still in flight, which replay waits for.
        merged = inv.journal.merged_over(replay)
        eager = await self._invoke(cell, arguments, path, request_id, merged, compiled=False)
        if inv.children:
            await asyncio.gather(*(h._task for h in inv.children), return_exceptions=True)
        eager.journal.mode = "deopt"
        eager.journal.deopt = result.reason
        return eager

    async def _invoke(
        self,
        cell: Cell,
        arguments: dict[str, Any],
        path: tuple[int, ...],
        request_id: str,
        replay: Journal | None = None,
        interrupt_at: int | None = None,
        make_ctx: Callable[[Invocation], Ctx] = Ctx,
        body: Body | None = None,
        compiled: bool = True,
    ) -> Invocation:
        if compiled and body is None and interrupt_at is None:
            plan = self.plan(cell)
            if plan is not None:
                return await self._invoke_compiled(plan, cell, arguments, path, request_id, replay)
        inv = Invocation(self, cell, arguments, path, request_id, replay, interrupt_at)
        inv.task = asyncio.current_task()
        result: Any = None
        body_error: Exception | None = None
        try:
            ctx = make_ctx(inv)
            result = await (body(ctx, arguments) if body is not None else cell.fn(ctx, **arguments))
            try:
                data.check(result)
            except DataError as e:
                raise DataError(f"{cell.id} returned a value that is not data: {e}") from None
            if cell.returns_none and result is not None:
                raise DataError(f"{cell.id} is annotated to return None but returned {type(result).__name__}")
        except _Interrupt:
            inv.interrupted = True
        except Exception as e:
            body_error = e
        except BaseException:
            inv.finished = True
            for h in inv.children:
                h._task.cancel()
            raise
        inv.finished = True
        # Structured concurrency: a cell completes only after every call it
        # issued has completed (DESIGN §1.4, rule 3).
        if inv.children:
            await asyncio.gather(*(h._task for h in inv.children), return_exceptions=True)
        if not inv.interrupted:
            inv.journal.outcome = _outcome(inv, result, body_error)
        if inv.replay is not None:
            inv.journal.unconsumed = [
                e.seq
                for e in inv.replay.entries
                if e.seq not in inv.consumed and e.kind == "call" and e.effectful and e.started
            ]
        return inv

    async def _run_call(self, parent: Invocation, entry: Entry, cell: Cell, arguments: dict[str, Any]) -> Any:
        try:
            outcome = await self._call_outcome(parent, entry, cell, arguments)
        except Exception as e:  # e.g. a DeterminismError from the journal lookup
            outcome = Err(e)
        entry.outcome = outcome
        return outcome.unwrap()

    async def _call_outcome(
        self, parent: Invocation, entry: Entry, cell: Cell, arguments: dict[str, Any]
    ) -> Outcome:
        # Handles passed as arguments resolve before the callee starts
        # (DESIGN §1.5). If one fails, the call fails without starting.
        try:
            arguments = await _resolve_handles(arguments)
        except Exception as e:
            return Err(e)
        entry.args = arguments
        entry.args_digest = data.digest(arguments)

        hit = parent.lookup(entry)
        if hit is not None:
            entry.replayed = True
            entry.started = hit.started
            entry.cached = hit.cached
            entry.child = hit.child
            if hit.outcome is not None:
                return hit.outcome
            if hit.handle is None:
                # A call inlined into a compiled run that deopted before it
                # finished: resume the callee from its partial journal.
                entry.replayed = False
                entry.started = True
                inv = await self._invoke(
                    cell, arguments, parent.journal.path + (entry.seq,), parent.journal.request_id, hit.child
                )
                entry.child = inv.journal
                assert inv.journal.outcome is not None
                return inv.journal.outcome
            # The call was still in flight when the journal was taken: wait for
            # it, then take its outcome and the callee's journal.
            try:
                outcome: Outcome = Ok(await hit.handle._task)
            except Exception as e:
                outcome = Err(e)
            entry.started = hit.started
            entry.child = hit.child
            return hit.outcome if hit.outcome is not None else outcome

        hit_cache, value = self._cached(cell, entry.args_digest)
        if hit_cache:
            entry.started = True
            entry.cached = True
            return Ok(value)

        entry.started = True
        inv = await self._invoke(cell, arguments, parent.journal.path + (entry.seq,), parent.journal.request_id)
        entry.child = inv.journal
        assert inv.journal.outcome is not None
        self._store(cell, entry.args_digest, inv.journal.outcome)
        return inv.journal.outcome


def _outcome(inv: Invocation, result: Any, body_error: Exception | None) -> Outcome:
    if inv.violation is not None:
        return Err(inv.violation)
    if body_error is not None:
        return Err(body_error)
    # An effectful call that was never awaited fails its parent if it failed
    # (DESIGN §1.4, rule 3). Unawaited pure calls are discarded.
    for h in inv.children:
        e = h._entry
        if e.effectful and not e.awaited and isinstance(e.outcome, Err):
            return Err(e.outcome.error)
    return Ok(result)


def _is_handle(value: Any) -> bool:
    return isinstance(value, Handle)


async def _resolve_handles(arguments: dict[str, Any]) -> dict[str, Any]:
    leaves, tree = data.flatten(arguments, is_leaf=_is_handle)
    if not any(isinstance(leaf, Handle) for leaf in leaves):
        return arguments
    # Await the underlying tasks, not the handles: passing a handle does
    # not count as awaiting it.
    values = [await leaf._task if isinstance(leaf, Handle) else leaf for leaf in leaves]
    return data.unflatten(tree, values)


@dataclass(frozen=True)
class Run:
    """The result of running (or replaying) a request."""

    journal: Journal
    interrupted: bool = False
    unconsumed: tuple[Entry, ...] = ()  # effectful journal entries replay never reached (DESIGN §3)

    @property
    def outcome(self) -> Outcome | None:
        return self.journal.outcome

    @property
    def value(self) -> Any:
        """The result, or raise the cell's error."""
        if self.outcome is None:
            raise RuntimeError("the run was interrupted")
        return self.outcome.unwrap()

    @property
    def error(self) -> BaseException | None:
        return self.outcome.error if isinstance(self.outcome, Err) else None
