"""The context a cell body runs in, and handles to issued calls."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

from .semantics import Domain

if TYPE_CHECKING:
    from .core import Cell
    from .journal import Entry
    from .runtime import Invocation


class Handle[T]:
    """A call that has been issued (DESIGN §1.4, rule 2).

    Await it for the result. A handle can also be passed, unawaited, as an
    argument to another call (DESIGN §1.5): the callee starts once it has
    resolved, and passing it does not count as awaiting it.
    """

    __slots__ = ("_task", "_entry")

    def __init__(self, task: asyncio.Task[T], entry: Entry):
        self._task = task
        self._entry = entry

    def __await__(self) -> Generator[Any, None, T]:
        self._entry.awaited = True
        return self._task.__await__()

    @property
    def done(self) -> bool:
        return self._task.done()

    def __bool__(self) -> bool:
        raise TypeError("a handle has no truth value; await it first")

    def __repr__(self) -> str:
        state = "done" if self._task.done() else "pending"
        return f"<handle #{self._entry.seq} {self._entry.target} {state}>"


class Ctx:
    """The only door from a cell body to the outside world (NOTES §4).

    Calls to other cells, time, randomness and configuration all go through
    the ctx, so a body is deterministic given its inputs and the outcomes of
    its calls, and its journal can replay it.
    """

    __slots__ = ("_inv",)

    def __init__(self, invocation: Invocation):
        self._inv = invocation

    @property
    def request_id(self) -> str:
        return self._inv.journal.request_id

    @property
    def path(self) -> tuple[int, ...]:
        """This invocation's call path: the seq of each call from the root."""
        return self._inv.journal.path

    def now(self) -> float:
        """The current time, in seconds. Journaled."""
        return self._inv.source("now", {})

    def random(self) -> float:
        """A number in [0, 1). Journaled; derived from the request and call path."""
        return self._inv.source("random", {})

    def config(self, key: str, default: Any = None) -> Any:
        """A configuration value. Journaled, so replay sees the value the run saw."""
        return self._inv.source("config", {"key": key, "default": default})

    def resource(self, key: Any) -> Any:
        """A live object provided by the runtime, such as a client or a fake service.

        For leaf cells that implement I/O. Resource access is not journaled;
        composite cells must not use it, or they are no longer deterministic.
        """
        return self._inv.runtime.resource(key)

    @contextlib.contextmanager
    def domain(self, name: Domain) -> Generator[None]:
        """Issue effectful calls in this block in the given effect domain (DESIGN §4.4)."""
        self._inv.push_domain(name)
        try:
            yield
        finally:
            self._inv.pop_domain()

    def _call(self, cell: Cell, arguments: dict[str, Any]) -> Handle[Any]:
        return self._inv.call(cell, arguments)

    def __repr__(self) -> str:
        return f"<ctx {self._inv.journal.cell} path={self.path}>"
