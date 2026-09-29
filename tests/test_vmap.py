"""Vector forms, ctx.map, and batches of requests (milestone M5).

- ctx.map fans out as one call, and one traced node, whatever its size.
- Calls to cells with vector forms go out as vector calls: within a request
  (calls ready at the same time), and across the requests of a batch.
- In a batch, each request runs in its own lane, deopts on its own (a
  failing guard partitions it out), and gets exactly the outcome and
  effects eager execution gives it. That is checked on the feed scenarios
  and on the random programs, whose `get` leaf has a vector form that
  fails on some keys, and with a deopt forced at every node.
"""

import asyncio
from collections import Counter

import pytest

from cell import Runtime, cell, effects, outcome_digest, pure
from cell.trace import trace
from examples import scenario, simtime
from examples.feed import BUSY, Feed, feed, feed_mapped, features, following, recent_posts, tables
from examples.harness import World
from test_random_programs import PROGRAMS, assert_effects_match, make


def run(coro):
    return asyncio.run(coro)


def rt(world=None, **options):
    return Runtime(resources={World: world or World(tables())}, **options)


# Vector forms


def test_only_pure_cells_have_vector_forms():
    @cell(effects("x"))
    async def write(ctx, k: int) -> None: ...

    with pytest.raises(ValueError):

        @write.vectorized
        async def write_many(ctx, k: list[int]) -> None: ...


def test_vector_forms_take_the_same_parameters():
    @cell(pure)
    async def one(ctx, k: int) -> int:
        return k

    with pytest.raises(TypeError):

        @one.vectorized
        async def many(ctx, keys: list[int]) -> list[int]:
            return keys


# ctx.map


def test_map_is_one_journaled_call_with_one_vector_call_under_it():
    w = World(tables())
    r = run(rt(w).run(feed_mapped, BUSY))
    assert r.value == scenario("feed_mapped/busy").expect
    maps = [e for e in r.journal.entries if e.target.startswith("cell.mapping")]
    assert len(maps) == 2
    assert [[c.target.rsplit(".", 1)[-1] for c in e.child.entries] for e in maps] == [
        ["recent_posts_many"],
        ["features_many"],
    ]
    assert Counter(c.name for c in w.calls) == Counter(
        {"following": 1, "recent_posts_many": 1, "features_many": 1, "mark_seen": 1}
    )


def test_map_without_a_vector_form_calls_each_item():
    @cell(pure)
    async def double(ctx, x: int) -> int:
        return 2 * x

    @cell
    async def f(ctx, xs: tuple[int, ...]) -> list[int]:
        return await ctx.map(double, xs)

    r = run(Runtime().run(f, (1, 2, 3)))
    assert r.value == [2, 4, 6]
    assert [e.target.rsplit(".", 1)[-1] for e in r.journal.entries[0].child.entries] == ["double"] * 3


def test_map_errors_are_attributed_to_items():
    class Bad(Exception):
        pass

    @cell(pure)
    async def check(ctx, x: int) -> int:
        if x < 0:
            raise Bad(x)
        return x

    @check.vectorized
    async def check_many(ctx, x: list[int]) -> list[int]:
        if any(v < 0 for v in x):
            raise Bad(x)
        return x

    @cell
    async def f(ctx, xs: tuple[int, ...]) -> list[int]:
        return await ctx.map(check, xs)

    r = run(Runtime().run(f, (1, -2, 3)))
    assert isinstance(r.error, Bad) and r.error.args == (-2,)
    children = [e.target.rsplit(".", 1)[-1] for e in r.journal.entries[0].child.entries]
    assert children == ["check_many", "check", "check", "check"]  # the vector call, then one by one


def test_map_traces_as_one_node_for_every_user():
    graphs = [run(scenario(n).trace())[0].graph for n in ("feed_mapped/top2", "feed_mapped/busy")]
    assert graphs[0].hash == graphs[1].hash
    unrolled = [run(scenario(n).trace())[0].graph for n in ("feed/top2", "feed/busy")]
    assert unrolled[0].hash != unrolled[1].hash
    assert len(unrolled[1].nodes) > 3 * len(graphs[1].nodes)


# Fusion within a request


def test_unrolled_calls_fuse_into_vector_calls():
    graph = run(scenario("feed/busy").trace())[0].graph
    w = World(tables())
    runtime = rt(w)
    runtime.install(graph)
    r = run(runtime.run(feed, BUSY))
    assert r.value == scenario("feed/busy").expect and r.journal.mode == "compiled"
    names = Counter(c.name for c in w.calls)
    # The 15 calls to features become one; the 6 to recent_posts become one.
    assert names["features_many"] == 1 and names["features"] == 0
    assert names["recent_posts_many"] == 1 and names["recent_posts"] == 0
    # recent_posts and features; following is a single call, with nothing to share, so it goes out as is.
    assert sum(e.batched for e in r.journal.entries) == 6 + 15


# Batches


FEED_REQUESTS = [1, BUSY, 4, BUSY, 1, 4, BUSY, 1]  # 4 follows nobody: its guard fails


def eager_runs(uids, cell_=feed_mapped):
    out = []
    for uid in uids:
        w = World(tables())
        out.append((run(rt(w).run(cell_, uid)), w))
    return out


def batched(uids, cell_=feed_mapped, fail_at=None, trace_uid=1):
    graph = run(trace(rt(), cell_, (trace_uid,))).graph
    w = World(tables())
    runtime = rt(w)
    runtime.install(graph).fail_at = fail_at
    runs = run(runtime.execute_batch(cell_, [((uid,), {}, f"r{i}") for i, uid in enumerate(uids)]))
    return runs, w, runtime


def test_a_batch_matches_eager_request_by_request():
    runs, w, runtime = batched(FEED_REQUESTS)
    for r, (e, _) in zip(runs, eager_runs(FEED_REQUESTS)):
        assert outcome_digest(r.outcome) == outcome_digest(e.outcome)
        assert r.journal.effects() == e.journal.effects()
    # A failing guard partitions a lane out; the others finish compiled.
    assert [r.journal.mode for r in runs] == ["deopt" if uid == 4 else "compiled" for uid in FEED_REQUESTS]
    # One vector call per leaf for the whole batch.
    assert Counter(c.name for c in w.calls) == Counter(
        {"following_many": 1, "recent_posts_many": 1, "features_many": 1, "mark_seen": 6}
    )
    assert runtime.vector_stats["calls"] == 3


def test_effects_in_a_batch_happen_once_per_request():
    runs, w, _ = batched(FEED_REQUESTS)
    seen = Counter((e.path, e.args["uid"]) for e in w.effects)
    assert sum(seen.values()) == sum(uid != 4 for uid in FEED_REQUESTS)


def test_a_batch_with_a_deopt_forced_at_every_node():
    graph = run(trace(rt(), feed_mapped, (1,))).graph
    expected = eager_runs(FEED_REQUESTS)
    for k in range(len(graph.nodes)):
        runs, _, _ = batched(FEED_REQUESTS, fail_at=k)
        for r, (e, _) in zip(runs, expected):
            assert outcome_digest(r.outcome) == outcome_digest(e.outcome), k
            assert r.journal.effects() == e.journal.effects(), k


def test_a_batch_of_the_unrolled_feed_partitions_on_length_guards():
    uids = [1, BUSY, 4, 1]
    runs, _, _ = batched(uids, cell_=feed, trace_uid=1)
    for r, (e, _) in zip(runs, eager_runs(uids, feed)):
        assert outcome_digest(r.outcome) == outcome_digest(e.outcome)
    assert [r.journal.mode for r in runs] == ["compiled", "deopt", "deopt", "compiled"]


@pytest.mark.parametrize("i", range(PROGRAMS))
def test_random_programs_in_batches(i):
    source, prog, inputs = make(i)
    graph = run(trace(Runtime(resources={World: World()}), prog, inputs[0])).graph
    expected = []
    for args in inputs:
        w = World()
        expected.append((run(Runtime(resources={World: w}).execute(prog, args, request_id=f"r{len(expected)}")), w))
    for k in [None, *range(0, len(graph.nodes), 2)]:
        # Each lane runs against its own world, so effects can be compared per request.
        worlds = [World() for _ in inputs]
        runtime = Runtime(resources={World: _PerRequest(worlds)})
        runtime.install(graph).fail_at = k
        runs = run(runtime.execute_batch(prog, [(args, {}, f"r{j}") for j, args in enumerate(inputs)]))
        for j, (r, (e, ew)) in enumerate(zip(runs, expected)):
            context = f"\n{source}\nargs={inputs[j]} fail_at={k}\n{graph.format()}\n{r.journal.format()}"
            assert outcome_digest(r.outcome) == outcome_digest(e.outcome), context
            assert r.journal.effects() == e.journal.effects(), context
            assert_effects_match(r, e, worlds[j], ew, context)


class _PerRequest:
    """A World per request, chosen by request id; vector calls go to a spare one."""

    def __init__(self, worlds):
        self.worlds = worlds
        self.spare = World()

    async def enter(self, ctx, name, n=1):
        rid = ctx.request_id
        world = self.worlds[int(rid[1:])] if rid.startswith("r") else self.spare
        return await world.enter(ctx, name, n)


def test_random_programs_use_vector_calls():
    fused = 0
    for i in range(PROGRAMS):
        _, prog, inputs = make(i)
        graph = run(trace(Runtime(resources={World: World()}), prog, inputs[0])).graph
        runtime = Runtime(resources={World: World()})
        runtime.install(graph)
        run(runtime.execute_batch(prog, [(args, {}, None) for args in inputs]))
        fused += runtime.vector_stats["calls"] > 0
    assert fused > PROGRAMS // 3, fused


# The batching policy


def test_requests_arriving_together_run_as_a_batch():
    graph = run(trace(rt(), feed_mapped, (1,))).graph

    async def go():
        w = World(tables(), default_latency=0.010)
        runtime = rt(w)
        runtime.install(graph)
        runtime.batch(feed_mapped, max_size=8, window=0.002)
        runs = await asyncio.gather(*(runtime.run(feed_mapped, uid) for uid in FEED_REQUESTS * 2))
        return runs, runtime, w

    (runs, runtime, w), elapsed = simtime.run(go())
    assert [r.value for r in runs] == [e.value for e, _ in eager_runs(FEED_REQUESTS * 2)]
    assert runtime.batch_stats == Counter({"batches": 2, "requests": 16})
    assert Counter(c.name for c in w.calls)["features_many"] == 2
    assert elapsed > 0.002  # the first batch waited out its window


def test_batching_without_a_graph_runs_requests_one_by_one():
    runtime = rt()
    runtime.batch(feed_mapped)
    r = run(runtime.run(feed_mapped, 1))
    assert r.value.posts and runtime.batch_stats["batches"] == 0


def test_cached_cells_are_not_fused():
    graph = run(trace(rt(), feed_mapped, (1,))).graph
    w = World(tables())
    runtime = rt(w, cache={following: 60})
    runtime.install(graph)
    run(runtime.execute_batch(feed_mapped, [((1,), {}, None), ((1,), {}, None)]))
    names = Counter(c.name for c in w.calls)
    assert names["following"] == 1 and names["following_many"] == 0  # the second hit the cache
    assert Feed and features and recent_posts


def test_lanes_finish_as_soon_as_they_are_done():
    """A request doesn't wait for the rest of its batch (regression: it used to,
    so every request paid for its batch's slowest lane)."""
    graph = run(trace(rt(), feed_mapped, (1,))).graph

    async def go():
        runtime = rt(World(tables(), default_latency=0.010))
        runtime.install(graph)
        runtime.batch(feed_mapped, max_size=2, window=0.001)
        loop = asyncio.get_running_loop()
        done = {}

        async def one(uid):
            r = await runtime.run(feed_mapped, uid)
            done[uid] = (loop.time(), r.journal.mode)

        await asyncio.gather(one(4), one(1))
        return done, runtime.batch_stats["batches"]

    (done, batches), _ = simtime.run(go())
    assert batches == 1
    # User 4 follows nobody: its lane deopts at the last guard and replays at
    # once; user 1's lane still has mark_seen (10ms) to do.
    assert done[4][1] == "deopt" and done[1][1] == "compiled"
    assert done[1][0] - done[4][0] > 0.005  # it was 0, when both waited for the batch
