"""Compiled execution, checked against eager execution (DESIGN §2's central invariant).

For the example scenarios:

- every graph traced from a scenario is run on every scenario of the same
  cell: the outcome, journal and effects match eager execution, whether
  the graph's guards hold (it runs compiled) or not (it deopts);
- a deopt is forced at each node of each graph, and the result still
  matches eager execution;
- the validator accepts every graph against recorded runs, and catches a
  graph that computes the wrong result.
"""

import asyncio
import dataclasses
import json
from collections import Counter

import pytest

from cell import Ok, Runtime, cell, effects, outcome_digest, pure
from cell.compiled import CompileError
from cell.graph import Graph
from cell.trace import trace
from cell.validate import promote, validate
from examples import SCENARIOS
from examples.features import kv_get, kv_put, nested, sum_keys
from examples.harness import Scenario, World


def run(coro):
    return asyncio.run(coro)


BY_CELL: dict[str, list[Scenario]] = {}
for _s in SCENARIOS:
    BY_CELL.setdefault(_s.cell.id, []).append(_s)
PAIRS = [(src, tgt) for ss in BY_CELL.values() for src in ss for tgt in ss]


def compiled_run(graph: Graph, s: Scenario, fail_at: int | None = None):
    world = s.world()
    rt = s.runtime(world)
    plan = rt.install(graph)
    plan.fail_at = fail_at
    r = run(rt.execute(s.cell, s.args, s.kwargs, request_id="req"))
    return r, world, plan


def assert_matches_eager(s: Scenario, r, world) -> None:
    eager, eager_world = run(s.run())
    assert outcome_digest(r.outcome) == outcome_digest(eager.outcome), r.journal.format()
    # Pure calls may be speculated, or fused into vector calls (M5), so
    # compare what matters outside: the outcome and effectful calls (§2).
    assert r.journal.effects() == eager.journal.effects(), r.journal.format()
    if not any(e.batched for j in r.journal.walk() for e in j.entries):
        assert r.journal.summary() == eager.journal.summary(), r.journal.format()
        # Every call eager makes happens; compiled may add speculative pure calls.
        assert not Counter(eager_world.calls) - Counter(world.calls)
    assert Counter(e.key() for e in world.effects) == Counter(e.key() for e in eager_world.effects)
    assert r.unconsumed == ()


@pytest.mark.parametrize("src, tgt", PAIRS, ids=[f"{a.name}@{b.name}" for a, b in PAIRS])
def test_graph_on_every_scenario_of_its_cell(src: Scenario, tgt: Scenario):
    graph = run(src.trace())[0].graph
    r, world, _ = compiled_run(graph, tgt)
    assert_matches_eager(tgt, r, world)
    eager = run(tgt.run())[0]
    if src is tgt and graph.complete and all(isinstance(e.outcome, Ok) for e in eager.journal.entries):
        # Its own scenario, which traced without breaking and where no call
        # failed: no deopt. (A failed call deopts even if nothing used it.)
        assert r.journal.mode == "compiled", r.journal.deopt


@pytest.mark.parametrize("s", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_forced_deopt_at_every_node(s: Scenario):
    graph = run(s.trace())[0].graph
    for k in range(len(graph.nodes)):
        r, world, plan = compiled_run(graph, s, fail_at=k)
        assert_matches_eager(s, r, world)
        if r.journal.mode == "deopt" and r.journal.deopt == f"forced deopt at %{k}":
            assert plan.deopts[f"forced deopt at %{k}"] == 1


@pytest.mark.parametrize("s", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_validation(s: Scenario):
    graph = run(s.trace())[0].graph
    recorded = [run(t.run())[0] for t in BY_CELL[s.cell.id]]
    v = run(validate(graph, recorded))
    assert v.passed, [(m.reason, m.actual.format()) for m in v.mismatches]
    assert v.compiled + v.deopted == len(recorded)


def test_validation_catches_a_wrong_graph():
    s = next(s for s in SCENARIOS if s.name == "home/premium")
    graph = run(s.trace())[0].graph
    j = graph.to_json()
    j["nodes"][-1]["inputs"] = [8]  # return the ranked items instead of the page
    wrong = Graph.from_json(j)
    recorded = [run(s.run())[0]]
    v = run(validate(wrong, recorded))
    assert [m.reason for m in v.mismatches] == ["outcome"]
    rt = Runtime()
    assert not run(promote(rt, wrong, recorded)).passed
    assert rt.plan(s.cell) is None
    assert run(promote(rt, graph, recorded)).passed
    assert rt.plan(s.cell) is not None


def test_graphs_load_from_json():
    s = next(s for s in SCENARIOS if s.name == "feed/top2")
    graph = run(s.trace())[0].graph
    loaded = Graph.from_json(json.loads(json.dumps(graph.to_json())))
    r, world, _ = compiled_run(loaded, s)
    assert r.journal.mode == "compiled"
    assert_matches_eager(s, r, world)


def test_a_graph_for_other_code_does_not_compile():
    s = next(s for s in SCENARIOS if s.name == "home/premium")
    graph = run(s.trace())[0].graph
    with pytest.raises(CompileError):
        Runtime().install(dataclasses.replace(graph, code="0" * 16))


def test_nested_cells_run_compiled():
    w = World({"kv": {"a": 1, "b": 2}})
    tracer_rt = Runtime(resources={World: w})
    outer = run(trace(tracer_rt, nested)).graph
    inner = run(trace(tracer_rt, sum_keys, (("a", "b"),))).graph
    rt = Runtime(resources={World: World({"kv": {"a": 1, "b": 2}})})
    rt.install(outer)
    rt.install(inner)
    r = run(rt.run(nested))
    assert r.value == 4
    assert r.journal.mode == "compiled"
    # The inner graph was traced on ("a", "b"); the second call deopts on its length guard.
    modes = [e.child.mode for e in r.journal.entries]
    assert modes == ["compiled", "deopt"]


# Dataflow, speculation and effects


def events_program():
    events = []

    @cell(pure)
    async def slow(ctx, name: str) -> str:
        events.append(("start", name))
        await asyncio.sleep(0.01)
        events.append(("end", name))
        return name

    @cell
    async def two(ctx) -> tuple[str, str]:
        a = await slow(ctx, "a")
        b = await slow(ctx, "b")  # doesn't depend on a
        return (a, b)

    return events, two


def test_independent_calls_run_in_parallel():
    events, two = events_program()
    rt = Runtime()
    graph = run(trace(rt, two)).graph
    assert events == [("start", "a"), ("end", "a"), ("start", "b"), ("end", "b")]
    events.clear()
    rt.install(graph)
    r = run(rt.run(two))
    assert r.value == ("a", "b") and r.journal.mode == "compiled"
    assert events[:2] == [("start", "a"), ("start", "b")]


def test_pure_calls_are_speculated_past_a_failing_guard():
    calls = []

    @cell(pure)
    async def first(ctx) -> int:
        await asyncio.sleep(0.01)
        calls.append("first")
        return 1

    @cell(pure)
    async def second(ctx) -> int:
        calls.append("second")
        return 2

    @cell
    async def f(ctx, x: int) -> int:
        v = await first(ctx)
        if v > x:
            return await second(ctx)  # doesn't depend on v: ready before the guard
        return v

    rt = Runtime()
    graph = run(trace(rt, f, (0,))).graph
    calls.clear()
    rt.install(graph)
    r = run(rt.run(f, 5))
    assert r.value == 1
    assert r.journal.mode == "deopt" and r.journal.deopt.startswith("guard")
    assert calls == ["second", "first"]  # speculated, then discarded by the deopt


def test_effects_are_never_speculated():
    @cell
    async def f(ctx, x: int) -> int:
        v = await kv_get(ctx, "a")
        if x > v:
            await kv_put(ctx, "big", x)
        return v

    w = World({"kv": {"a": 1}})
    graph = run(trace(Runtime(resources={World: w}), f, (5,))).graph
    w = World({"kv": {"a": 1}})
    rt = Runtime(resources={World: w})
    rt.install(graph)
    r = run(rt.run(f, 0))
    assert r.value == 1 and r.journal.mode == "deopt"
    assert w.effects == []


def test_an_effect_waits_for_a_failing_input():
    """Regression: an effectful call must not start on an input call's handle."""

    @cell(effects("kv"))
    async def note(ctx, v: int) -> None:
        ctx.resource(World).effect(ctx, "note", v=v)

    @cell(pure)
    async def maybe_fail(ctx, x: int) -> int:
        if x < 0:
            raise ValueError(x)
        return x

    @cell
    async def f(ctx, x: int) -> int:
        v = await maybe_fail(ctx, x)
        await note(ctx, v)
        return v

    graph = run(trace(Runtime(resources={World: World()}), f, (1,))).graph
    w = World()
    rt = Runtime(resources={World: w})
    rt.install(graph)
    r = run(rt.run(f, -1))
    assert isinstance(r.error, ValueError) and r.journal.mode == "deopt"
    assert w.effects == []


def test_deopt_statistics():
    s = next(s for s in SCENARIOS if s.name == "home/premium")
    graph = run(s.trace())[0].graph
    basic = next(s for s in SCENARIOS if s.name == "home/basic")
    world = basic.world()
    rt = basic.runtime(world)
    plan = rt.install(graph)
    for _ in range(3):
        run(rt.execute(basic.cell, basic.args))
    run(rt.execute(s.cell, s.args))
    assert plan.runs == 4
    assert plan.deopts == Counter({"guard %7 failed": 3})
