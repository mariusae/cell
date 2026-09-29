"""The eager runtime (DESIGN §2, milestone M0).

Cell bodies run as ordinary Python. Each call is issued when it is made,
runs in its own task, and is resolved in-process. Every invocation keeps
a journal (DESIGN §3), and an invocation can be replayed from a journal:
journaled calls return their recorded outcomes instead of running again.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from . import data
from .context import Ctx, Handle
from .core import Cell
from .errors import ContextError, DataError, DeterminismError
from .journal import Entry, Err, Journal, Ok, Outcome
from .semantics import Domain, check_domain


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

    def _issue(self) -> int:
        seq = self._next_seq
        if self._interrupt_at is not None and seq >= self._interrupt_at:
            raise _Interrupt()
        self._next_seq += 1
        return seq

    def call(self, cell: Cell, arguments: dict[str, Any]) -> Handle[Any]:
        self._check_ctx()
        try:
            data.flatten(arguments, is_leaf=_is_handle)
        except DataError as e:
            raise DataError(f"argument to {cell.id} is not data: {e}") from None
        seq = self._issue()
        domain = None
        if cell.effectful:
            domain = self._domains[-1] if self._domains else cell.domain
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

    def source(self, name: str, args: dict[str, Any], compute: Callable[[int], Any]) -> Any:
        self._check_ctx()
        seq = self._issue()
        entry = Entry(seq=seq, kind="source", target=name, args=args, args_digest=data.digest(args))
        self.journal.entries.append(entry)
        hit = self.lookup(entry)
        if hit is not None and isinstance(hit.outcome, Ok):
            value = hit.outcome.value
            entry.replayed = True
        else:
            value = compute(seq)
            data.check(value)
        entry.started = True
        entry.awaited = True
        entry.outcome = Ok(value)
        return value

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
        if old.effectful:
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
    ):
        self.config = dict(config or {})
        data.check(self.config)
        self._resources = dict(resources or {})
        self.clock = clock or time.time
        self.seed = seed
        self._request_ids = itertools.count(1)

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
            request_id = replay.request_id if replay is not None else f"r{next(self._request_ids)}"
        inv = await asyncio.create_task(
            self._invoke(cell, arguments, (), request_id, replay, interrupt_at)
        )
        unconsumed: tuple[Entry, ...] = ()
        if replay is not None:
            unconsumed = tuple(
                e
                for e in replay.entries
                if e.seq not in inv.consumed and e.kind == "call" and e.effectful and e.started
            )
        return Run(inv.journal, inv.interrupted, unconsumed)

    async def _invoke(
        self,
        cell: Cell,
        arguments: dict[str, Any],
        path: tuple[int, ...],
        request_id: str,
        replay: Journal | None = None,
        interrupt_at: int | None = None,
    ) -> Invocation:
        inv = Invocation(self, cell, arguments, path, request_id, replay, interrupt_at)
        inv.task = asyncio.current_task()
        result: Any = None
        body_error: Exception | None = None
        try:
            result = await cell.fn(Ctx(inv), **arguments)
            try:
                data.check(result)
            except DataError as e:
                raise DataError(f"{cell.id} returned a value that is not data: {e}") from None
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
            entry.child = hit.child
            if hit.outcome is not None:
                return hit.outcome
            if hit.handle is None:
                raise RuntimeError(f"journal entry {hit.key} has neither an outcome nor a handle")
            try:  # the call was still in flight when the journal was taken
                return Ok(await hit.handle._task)
            except Exception as e:
                return Err(e)

        entry.started = True
        inv = await self._invoke(cell, arguments, parent.journal.path + (entry.seq,), parent.journal.request_id)
        entry.child = inv.journal
        assert inv.journal.outcome is not None
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
