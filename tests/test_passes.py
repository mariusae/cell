"""Graph passes, caching, and the latency they buy (milestone M4).

An optimized graph issues fewer pure calls than eager execution (shared,
folded or cached), so it must match eager execution in outcome and effects
(Journal.effects), not call for call. That is checked for:

- every example graph, optimized and run on every scenario of its cell,
  and with a deopt forced at each node;
- random programs that call a generated composite cell, whose graph is
  inlined, with a deopt forced at each node: a deopt inside an inlined
  callee resumes it from its partial journal, exactly once.

Latency is measured in simulated time (examples/simtime.py), so the
numbers are exact.
"""

import asyncio
import linecache
import random
from collections import Counter

import pytest

from cell import Runtime, cell, effects, outcome_digest, pure
from cell.graph import Const
from cell.passes import dedup, fold, inline, optimize
from cell.trace import trace
from examples import SCENARIOS, simtime
from examples.features import concurrent, kv_get, kv_put, nested, sum_keys
from examples.harness import Scenario, World
from test_random_programs import NAMESPACE, Generator, assert_effects_match

BY_CELL: dict[str, list[Scenario]] = {}
for _s in SCENARIOS:
    BY_CELL.setdefault(_s.cell.id, []).append(_s)
PAIRS = [(src, tgt) for ss in BY_CELL.values() for src in ss for tgt in ss]


def run(coro):
    return asyncio.run(coro)


def assert_equivalent(r, eager, world, eager_world, context=""):
    assert outcome_digest(r.outcome) == outcome_digest(eager.outcome), context + r.journal.format()
    assert r.journal.effects() == eager.journal.effects(), context + r.journal.format()
    assert_effects_match(r, eager, world, eager_world, context)


def optimized(s: Scenario):
    graph = run(s.trace())[0].graph
    callees = {}
    if s.cell is nested:  # the one example with a composite callee
        w = World({"kv": {"a": 1, "b": 2}})
        callees[sum_keys.id] = run(trace(Runtime(resources={World: w}), sum_keys, (("a", "b"),))).graph
    return optimize(graph, callees)


def run_graph(graph, s: Scenario, fail_at=None):
    world = s.world()
    rt = s.runtime(world)
    rt.install(graph).fail_at = fail_at
    return run(rt.execute(s.cell, s.args, s.kwargs, request_id="req")), world


@pytest.mark.parametrize("src, tgt", PAIRS, ids=[f"{a.name}@{b.name}" for a, b in PAIRS])
def test_optimized_graphs_match_eager(src, tgt):
    graph = optimized(src)
    r, world = run_graph(graph, tgt)
    eager, eager_world = run(tgt.run())
    assert_equivalent(r, eager, world, eager_world, graph.format() + "\n")


@pytest.mark.parametrize("s", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_optimized_graphs_with_forced_deopts(s):
    graph = optimized(s)
    eager, eager_world = run(s.run())
    for k in range(len(graph.nodes)):
        r, world = run_graph(graph, s, fail_at=k)
        assert_equivalent(r, eager, world, eager_world, f"fail_at={k}\n{graph.format()}\n")


# The passes, one at a time


def test_dedup_shares_identical_pure_calls():
    s = next(s for s in SCENARIOS if s.name == "features/concurrent")
    graph = run(s.trace())[0].graph
    shared = dedup(graph)
    calls = lambda g: [n for n in g.nodes if n.kind == "call"]  # noqa: E731
    assert len(calls(graph)) == 3 and len(calls(shared)) == 2
    r, world = run_graph(shared, s)
    assert r.value == 4 and r.journal.mode == "compiled"
    assert [c.name for c in world.calls] == ["kv_get", "kv_get"]


def test_dedup_never_shares_effects():
    @cell
    async def twice(ctx) -> None:
        await kv_put(ctx, "k", 1)
        await kv_put(ctx, "k", 1)

    graph = run(trace(Runtime(resources={World: World()}), twice)).graph
    assert len(dedup(graph).nodes) == len(graph.nodes)


def test_fold():
    @cell
    async def f(ctx) -> int:
        k = ("a", "b")[len(("x",)) - 1]
        return await kv_get(ctx, k)

    graph = run(trace(Runtime(resources={World: World({"kv": {"a": 1}})}), f)).graph
    folded = fold(graph)
    assert [n.kind for n in folded.nodes] == ["call", "return"]
    assert folded.nodes[0].inputs == [Const("a")]


def test_inlined_callee_runs_compiled():
    graph = optimized(next(s for s in SCENARIOS if s.cell is nested))
    kinds = [n.kind for n in graph.nodes]
    # The first call site is inlined; the second can't match the callee's
    # specialization (its length guard folds to false), so it stays a call.
    assert kinds.count("enter") == 1 and kinds.count("exit") == 1
    r, _ = run_graph(graph, next(s for s in SCENARIOS if s.cell is nested))
    assert r.value == 4 and r.journal.mode == "compiled"
    assert r.journal.entries[0].child.mode == "compiled"


def test_inlining_stops_at_recursion():
    @cell
    async def countdown(ctx, n: int) -> int:
        if n <= 0:
            return 0
        return 1 + await countdown(ctx, n - 1)

    graph = run(trace(Runtime(), countdown, (1,))).graph
    inlined = inline(graph, {countdown.id: graph})
    assert [n.kind for n in inlined.nodes].count("enter") == 0


# An inlined effectful callee, interrupted by deopts at every node


@cell(pure)
async def lookup(ctx, k: str) -> int:
    await ctx.resource(World).enter(ctx, "lookup")
    return {"a": 1, "b": 2}.get(k, 0)


@cell(effects("kv"))
async def write(ctx, k: str, v: int) -> None:
    w = await ctx.resource(World).enter(ctx, "write")
    w.effect(ctx, "write", k=k, v=v)


@cell
async def step(ctx, k: str) -> int:
    await write(ctx, k, 1)
    v = await lookup(ctx, k)
    await write(ctx, k + "!", v)
    return v


@cell
async def two_steps(ctx, k: str) -> int:
    a = await step(ctx, k)
    b = await step(ctx, "b")
    return a + b


def test_deopts_inside_inlined_callees_resume_them_exactly_once():
    inner = run(trace(Runtime(resources={World: World()}), step, ("a",))).graph
    outer = run(trace(Runtime(resources={World: World()}), two_steps, ("a",))).graph
    graph = optimize(outer, {step.id: inner})
    assert [n.kind for n in graph.nodes].count("enter") == 2
    eager_world = World()
    eager = run(Runtime(resources={World: eager_world}).run(two_steps, "a"))
    for k in [None, *range(len(graph.nodes))]:
        world = World()
        rt = Runtime(resources={World: world})
        rt.install(graph).fail_at = k
        r = run(rt.run(two_steps, "a"))
        assert_equivalent(r, eager, world, eager_world, f"fail_at={k}\n")
        if k is None:
            assert r.journal.mode == "compiled"


# Random programs calling a generated composite cell


PROGRAMS = 40
SEED = 777


def make_nested(i: int):
    rng = random.Random(SEED + i)
    namespace = dict(NAMESPACE, __name__=f"nested_{i}")
    sources = []
    for name in ("sub", "prog"):
        g = Generator(rng)
        if name == "prog":
            g.callee = "sub"
        text = g.program().replace("async def prog(", f"async def {name}(")
        filename = f"<nested {i} {name}>"
        linecache.cache[filename] = (len(text), None, text.splitlines(True), filename)
        exec(compile(text, filename, "exec"), namespace)
        sources.append(text)
    inputs = [(rng.randrange(20), rng.randrange(20)) for _ in range(3)]
    return "\n".join(sources), namespace["sub"], namespace["prog"], inputs


@pytest.mark.parametrize("i", range(PROGRAMS))
def test_random_nested_programs(i):
    source, sub, prog, inputs = make_nested(i)
    sub_graph = run(trace(Runtime(resources={World: World()}), sub, inputs[0])).graph
    for targs in inputs:
        graph = optimize(run(trace(Runtime(resources={World: World()}), prog, targs)).graph, {sub.id: sub_graph})
        for args in inputs:
            eager_world = World()
            eager = run(Runtime(resources={World: eager_world}).execute(prog, args, request_id="r"))
            for k in [None, *range(0, len(graph.nodes), 3)] if args == targs else [None]:
                world = World()
                rt = Runtime(resources={World: world})
                rt.install(graph).fail_at = k
                r = run(rt.execute(prog, args, request_id="r"))
                assert_equivalent(r, eager, world, eager_world, f"{source}\nargs={args} fail_at={k}\n{graph.format()}\n")


def test_nested_programs_inline_and_run_compiled():
    inlined = compiled = 0
    for i in range(PROGRAMS):
        _, sub, prog, inputs = make_nested(i)
        sub_graph = run(trace(Runtime(resources={World: World()}), sub, inputs[0])).graph
        graph = optimize(run(trace(Runtime(resources={World: World()}), prog, inputs[0])).graph, {sub.id: sub_graph})
        inlined += any(n.kind == "enter" for n in graph.nodes)
        rt = Runtime(resources={World: World()})
        rt.install(graph)
        compiled += run(rt.execute(prog, inputs[0], request_id="r")).journal.mode == "compiled"
    assert inlined > PROGRAMS // 3 and compiled > PROGRAMS // 3, (inlined, compiled)


# Caching


def test_only_pure_cells_can_be_cached():
    with pytest.raises(ValueError):
        Runtime(cache={kv_put: 60})


def test_cache_hits_across_requests_and_expires():
    now = [0.0]
    world = World({"kv": {"a": 1, "b": 2}})
    rt = Runtime(resources={World: world}, clock=lambda: now[0], cache={kv_get: 10})
    first = run(rt.run(concurrent))
    second = run(rt.run(concurrent))
    assert first.value == second.value == 4
    assert [c.name for c in world.calls] == ["kv_get"] * 2  # 'a' and 'b', once each
    assert [e.cached for e in second.journal.entries] == [True, True, True]
    now[0] = 11.0
    run(rt.run(concurrent))
    assert len(world.calls) == 4
    # Hits: the repeated 'a' in the first and third runs, and all three calls in the second.
    assert rt.cache_stats["hit"] == 5


def test_errors_are_not_cached():
    from examples.features import Boom, boom, fallback

    world = World({"kv": {"b": 2}})
    rt = Runtime(resources={World: world}, cache={boom: 60, kv_get: 60})
    assert run(rt.run(fallback)).value == 2
    assert run(rt.run(fallback)).value == 2
    assert [c.name for c in world.calls] == ["boom", "kv_get", "boom"]
    assert Boom


def test_replay_reproduces_cached_calls():
    world = World({"kv": {"a": 1, "b": 2}})
    rt = Runtime(resources={World: world}, cache={kv_get: 60})
    run(rt.run(concurrent))
    second = run(rt.run(concurrent))
    replayed = run(Runtime().replay(concurrent, second.journal))
    assert replayed.value == 4 and replayed.journal.effects() == second.journal.effects()


# Latency, in simulated time


def latency(s: Scenario, graph=None, latency=None, **runtime_options):
    async def go():
        world = World(s.tables(), latency=latency or {}, default_latency=0.010)
        rt = Runtime(config=s.config, resources={World: world}, **runtime_options)
        if graph is not None:
            rt.install(graph)
        return await rt.execute(s.cell, s.args, s.kwargs, request_id="req")

    r, elapsed = simtime.run(go())
    return r, round(elapsed * 1000, 3)


def test_dataflow_overlaps_calls_program_order_serialized():
    s = next(s for s in SCENARIOS if s.name == "home/premium")
    slow_items = {"get_items": 0.030}  # a slow candidate source
    graph = optimized(s)
    eager, eager_ms = latency(s, latency=slow_items)
    compiled, compiled_ms = latency(s, graph, latency=slow_items)
    assert compiled.journal.mode == "compiled" and compiled.value == eager.value
    # Eager: get_user, then get_items, then rank. Compiled: get_items starts at once.
    assert (eager_ms, compiled_ms) == (50.0, 40.0)


def test_effect_domains_take_audit_writes_off_the_critical_path():
    s = next(s for s in SCENARIOS if s.name == "checkout/ok")
    graph = optimized(s)
    eager, eager_ms = latency(s)
    compiled, compiled_ms = latency(s, graph)
    assert compiled.journal.mode == "compiled" and compiled.value == eager.value
    # Five sequential writes eagerly; compiled, both audit writes overlap the main domain.
    assert (eager_ms, compiled_ms) == (50.0, 30.0)


def test_caching_across_requests():
    s = next(s for s in SCENARIOS if s.name == "home/premium")
    graph = optimized(s)

    async def two():
        world = World(s.tables(), default_latency=0.010)
        rt = Runtime(resources={World: world}, cache={c: 60 for c in _home_pure()})
        rt.install(graph)
        await rt.execute(s.cell, s.args, request_id="r1")
        loop = asyncio.get_running_loop()
        start = loop.time()
        r = await rt.execute(s.cell, s.args, request_id="r2")
        return r, loop.time() - start

    (r, second), _ = simtime.run(two())
    assert r.journal.mode == "compiled"
    assert round(second * 1000, 3) == 10.0  # only get_prefs, which is effectful, goes out


def _home_pure():
    from examples.home import get_items, get_user, rank

    return [get_user, get_items, rank]
