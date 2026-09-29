"""Simulated time: an event loop whose clock advances only when it would wait.

Leaf cells simulate latency with asyncio.sleep. On this loop a sleep takes
no real time: whenever nothing is ready to run, the clock jumps to the next
timer. Latencies measured with loop.time() are therefore exact and
repeatable, which makes latency comparisons between execution strategies
deterministic.
"""

from __future__ import annotations

import asyncio
import selectors
from collections.abc import Coroutine
from typing import Any


class _Autojump:
    """Wraps the loop's selector: never blocks, and advances the clock instead."""

    def __init__(self, loop: VirtualTimeLoop, inner: selectors.BaseSelector):
        self._loop = loop
        self._inner = inner

    def select(self, timeout: float | None = None) -> list[Any]:
        events = self._inner.select(0)
        if not events and timeout is not None and timeout > 0:
            self._loop._now += timeout
        return events

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class VirtualTimeLoop(asyncio.SelectorEventLoop):
    def __init__(self) -> None:
        super().__init__()
        self._now = 0.0
        self._selector = _Autojump(self, self._selector)  # type: ignore[assignment]

    def time(self) -> float:
        return self._now


def run(coro: Coroutine[Any, Any, Any]) -> tuple[Any, float]:
    """Run a coroutine in simulated time. Returns its result and the elapsed virtual seconds."""
    loop = VirtualTimeLoop()
    try:
        start = loop.time()
        result = loop.run_until_complete(coro)
        return result, loop.time() - start
    finally:
        loop.close()
