"""Tracer strictness with sys.monitoring (DESIGN §5.4, defense 3).

Each program here takes a path in trace mode that it wouldn't take eagerly
unless the tracer notices what C code does with a traced value. With
strictness on, each breaks, and replay gives the eager result. With it off
(strict=False), the trace silently records the wrong path: these are the
holes defense 3 closes.
"""

import asyncio
import json

import pytest

from cell import Runtime, cell, outcome_digest
from cell.monitor import MONITOR
from cell.trace import trace
from examples.features import kv_get
from examples.harness import World


def run(coro):
    return asyncio.run(coro)


def traced(f, strict=True):
    rt = Runtime(resources={World: World({"kv": {"a": 1, "s": "x"}})})
    t = run(trace(rt, f, request_id="t", strict=strict))
    eager = run(Runtime(resources={World: World({"kv": {"a": 1, "s": "x"}})}).execute(f, request_id="t"))
    return t, eager


@cell
async def uses_type(ctx) -> str:
    a = await kv_get(ctx, "a")
    return "int" if type(a) is int else "other"


@cell
async def uses_id(ctx) -> bool:
    a = await kv_get(ctx, "a")
    return id(a) == id(a) and True


@cell
async def catches_join(ctx) -> str:
    s = await kv_get(ctx, "s")
    try:
        return ", ".join([s, s])
    except TypeError:
        return "fallback"


@cell
async def catches_json(ctx) -> str:
    a = await kv_get(ctx, "a")
    try:
        return json.dumps({"a": a})
    except TypeError:
        return "fallback"


@cell
async def uses_callable(ctx) -> bool:
    a = await kv_get(ctx, "a")
    return callable(a)


HOLES = [
    (uses_type, "passed to type"),
    (catches_join, "TypeError was caught"),
    (uses_callable, "passed to callable"),
]


@pytest.mark.parametrize("f, reason", HOLES, ids=[f.__name__ for f, _ in HOLES])
def test_strict_tracing_breaks_and_gets_the_eager_result(f, reason):
    t, eager = traced(f)
    assert t.graph.nodes[-1].kind == "deopt", t.graph.format()
    assert reason in t.graph.nodes[-1].attrs["reason"]
    assert outcome_digest(t.run.outcome) == outcome_digest(eager.outcome)


@pytest.mark.parametrize("f, reason", HOLES, ids=[f.__name__ for f, _ in HOLES])
def test_without_strictness_the_trace_takes_the_wrong_path(f, reason):
    t, eager = traced(f, strict=False)
    assert outcome_digest(t.run.outcome) != outcome_digest(eager.outcome)


def test_json_breaks_either_way():
    """The JSON encoder calls isinstance, which breaks through __class__ (defense 2)."""
    for strict in (True, False):
        t, eager = traced(catches_json, strict=strict)
        assert t.graph.nodes[-1].kind == "deopt"
        assert t.run.value == eager.value == '{"a": 1}'


def test_id_breaks():
    t, eager = traced(uses_id)
    assert "passed to id" in t.graph.nodes[-1].attrs["reason"]
    assert t.run.value == eager.value


def test_safe_uses_do_not_break():
    @cell
    async def f(ctx) -> tuple:
        a = await kv_get(ctx, "a")
        xs = []
        xs.append(a)
        xs.extend([a, a])
        d = {"k": a}
        return (len(xs), sum(xs), max(xs), tuple(sorted(xs)), d.get("k"), list(zip(xs, xs))[0])

    t, eager = traced(f)
    assert t.graph.complete, t.graph.format()
    assert t.run.value == eager.value


def test_monitoring_stops_after_the_trace():
    t, _ = traced(uses_type)
    assert not MONITOR.available
    assert MONITOR.active == {}


def test_concurrent_eager_work_is_not_affected():
    """Only the traced body's task is watched; other requests may use type() freely."""

    @cell
    async def other(ctx) -> str:
        await asyncio.sleep(0.001)
        return type(await kv_get(ctx, "a")).__name__

    async def both():
        world = World({"kv": {"a": 1}})
        rt = Runtime(resources={World: world})
        eager_task = asyncio.create_task(rt.run(other))
        t = await trace(rt, uses_type)
        return t, await eager_task

    t, r = run(both())
    assert r.value == "int"
    assert t.graph.nodes[-1].kind == "deopt"
