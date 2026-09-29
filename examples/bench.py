"""Latency of the example scenarios: eager, compiled, and compiled with caching.

Run with `uv run python -m examples --bench`. Leaf calls take 10ms, except
where a scenario's profile says otherwise. Times are simulated
(simtime.py), so they are exact.

- eager: the cell run as written.
- compiled: its graph, traced from the same scenario and optimized
  (inlined, folded, deduplicated), run as dataflow.
- cached: compiled, with pure leaf cells cached, on a second request.
"""

from __future__ import annotations

import asyncio
from typing import Any

from cell import Runtime
from cell.passes import optimize
from cell.static import call_graph

from . import SCENARIOS, simtime
from .harness import Scenario, World

DEFAULT_LATENCY = 0.010
PROFILES: dict[str, dict[str, float]] = {
    "examples.home": {"get_items": 0.030, "get_items_many": 0.030},  # a slow candidate source
    # Scoring features is the expensive service (model inference); marking
    # posts seen is a cheap write.
    "examples.feed": {"features": 0.030, "features_many": 0.030, "mark_seen": 0.005},
}


def _profile(s: Scenario) -> dict[str, float]:
    return PROFILES.get(s.cell.fn.__module__, {})


async def callee_graphs(s: Scenario, journal: Any) -> dict[str, Any]:
    """Graphs for the composite cells a request called, traced from those calls."""
    callees: dict[str, Any] = {}
    cells = {c.id: c for c in call_graph([s.cell])}
    for entry in journal.entries:
        child = entry.child
        if child is not None and child.cell in cells and any(e.kind == "call" for e in child.entries):
            traced, _ = await Scenario(s.name, cells[child.cell], (), dict(child.args), s.tables).trace()
            callees.setdefault(child.cell, traced.graph)
    return callees


def _graph(s: Scenario) -> Any:
    traced, _ = asyncio.run(s.trace())
    return optimize(traced.graph, asyncio.run(callee_graphs(s, traced.run.journal)))


def _latency(s: Scenario, graph: Any = None, cache: bool = False) -> tuple[str, float]:
    async def go() -> tuple[str, float]:
        world = World(s.tables(), latency=_profile(s), default_latency=DEFAULT_LATENCY)
        pure = [c for c in call_graph([s.cell]) if not c.effectful] if cache else []
        rt = Runtime(config=s.config, resources={World: world}, cache={c: 60.0 for c in pure})
        if graph is not None:
            rt.install(graph)
        if cache:
            await rt.execute(s.cell, s.args, s.kwargs, request_id="warm")
        loop = asyncio.get_running_loop()
        start = loop.time()
        r = await rt.execute(s.cell, s.args, s.kwargs, request_id="req")
        return r.journal.mode, loop.time() - start

    (mode, elapsed), _ = simtime.run(go())
    return mode, elapsed * 1000


def main() -> None:
    print(f"{'scenario':32} {'eager':>8} {'compiled':>9} {'cached':>8}  mode")
    for s in SCENARIOS:
        graph = _graph(s)
        _, eager = _latency(s)
        mode, compiled = _latency(s, graph)
        _, cached = _latency(s, graph, cache=True)
        print(f"{s.name:32} {eager:6.0f}ms {compiled:7.0f}ms {cached:6.0f}ms  {mode}")
