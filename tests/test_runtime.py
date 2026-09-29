"""Semantics of the eager runtime that the example scenarios don't pin down."""

import asyncio
import dataclasses

import pytest

from cell import (
    MAIN,
    UNIQUE,
    ContextError,
    DataError,
    DeterminismError,
    Entry,
    Err,
    Handle,
    Journal,
    Ok,
    Runtime,
    cell,
    effects,
    pure,
)
from cell.data import digest
from examples.features import kv_get, kv_put, sum_keys
from examples.harness import World


def run(coro):
    return asyncio.run(coro)


def runtime(**tables) -> tuple[Runtime, World]:
    world = World({"kv": {"a": 1, "b": 2}, **tables})
    return Runtime(resources={World: world}, clock=lambda: 42.0), world


# Declarations


def test_cell_must_be_async():
    with pytest.raises(TypeError):
        cell(lambda ctx: 0)


def test_cell_must_take_ctx():
    with pytest.raises(TypeError):

        @cell
        async def no_params() -> None: ...


def test_cell_rejects_varargs():
    with pytest.raises(TypeError):

        @cell
        async def varargs(ctx, *xs) -> None: ...


def test_pure_cell_has_no_domain():
    with pytest.raises(ValueError):

        @cell(pure, domain="audit")
        async def f(ctx) -> None: ...


def test_default_semantics():
    @cell
    async def f(ctx) -> None: ...

    assert f.effectful and f.domain == MAIN
    assert repr(f.semantics) == "external"
    assert not kv_get.effectful and kv_get.domain is None
    assert kv_put.semantics == effects("kv")


# Calling


def test_call_requires_ctx():
    with pytest.raises(ContextError):
        kv_get(None, "a")


def test_bad_arguments_raise_at_call_site():
    @cell
    async def f(ctx) -> None:
        kv_get(ctx, "a", "extra")

    rt, _ = runtime()
    r = run(rt.run(f))
    assert isinstance(r.error, TypeError)
    assert r.journal.entries == []


def test_non_data_argument_raises_at_call_site():
    @dataclasses.dataclass
    class Mutable:
        x: int

    @cell
    async def f(ctx) -> None:
        kv_get(ctx, Mutable(1))

    rt, _ = runtime()
    r = run(rt.run(f))
    assert isinstance(r.error, DataError)
    assert r.journal.entries == []


def test_non_data_result_fails_the_cell():
    @cell(pure)
    async def f(ctx) -> object:
        return {1, 2}

    rt, _ = runtime()
    assert isinstance(run(rt.run(f)).error, DataError)


def test_calls_are_issued_when_made():
    order = []

    @cell(pure)
    async def leaf(ctx, n: int) -> int:
        order.append(("start", n))
        return n

    @cell
    async def f(ctx) -> int:
        a = leaf(ctx, 1)
        b = leaf(ctx, 2)
        order.append(("body", 0))
        await asyncio.sleep(0)
        order.append(("body", 1))
        return await b + await a

    rt, _ = runtime()
    assert run(rt.run(f)).value == 3
    # Both calls started before the body resumed, without being awaited.
    assert order == [("body", 0), ("start", 1), ("start", 2), ("body", 1)]


def test_structured_concurrency_waits_for_unawaited_calls():
    done = []

    @cell(effects("x"))
    async def slow(ctx) -> None:
        await asyncio.sleep(0.01)
        done.append(True)

    @cell
    async def f(ctx) -> int:
        slow(ctx)
        return 1

    rt, _ = runtime()
    r = run(rt.run(f))
    assert r.value == 1 and done == [True]


def test_body_error_wins_over_unawaited_failures():
    class BodyError(Exception):
        pass

    @cell(effects("x"))
    async def fails(ctx) -> None:
        raise ValueError("child")

    @cell
    async def f(ctx) -> None:
        fails(ctx)
        raise BodyError()

    rt, _ = runtime()
    assert isinstance(run(rt.run(f)).error, BodyError)


def test_calls_from_another_task_are_rejected():
    @cell
    async def f(ctx) -> int:
        async def helper():
            return await kv_get(ctx, "a")

        return await asyncio.create_task(helper())

    rt, _ = runtime()
    assert isinstance(run(rt.run(f)).error, ContextError)


def test_ctx_cannot_be_used_after_the_invocation_finishes():
    leaked = []

    @cell(pure)
    async def leak(ctx) -> None:
        leaked.append(ctx)

    @cell
    async def f(ctx) -> int:
        await leak(ctx)
        return await kv_get(leaked[0], "a")

    rt, _ = runtime()
    r = run(rt.run(f))
    assert isinstance(r.error, ContextError)
    assert "finished" in str(r.error)


def test_paths_and_request_ids():
    seen = []

    @cell(pure)
    async def leaf(ctx) -> None:
        seen.append((ctx.request_id, ctx.path))

    @cell
    async def mid(ctx) -> None:
        ctx.now()
        await leaf(ctx)

    @cell
    async def top(ctx) -> None:
        await leaf(ctx)
        await mid(ctx)

    rt, _ = runtime()
    r1 = run(rt.run(top))
    r2 = run(rt.run(top))
    assert r1.journal.request_id != r2.journal.request_id
    assert [p for _, p in seen[:2]] == [(0,), (1, 1)]


def test_resources_are_per_runtime():
    @cell(pure)
    async def f(ctx) -> None:
        ctx.resource("missing")

    rt, _ = runtime()
    assert isinstance(run(rt.run(f)).error, ContextError)


# Sources


def test_random_is_derived_from_request_and_path():
    @cell
    async def f(ctx) -> tuple[float, float]:
        return (ctx.random(), ctx.random())

    rt, _ = runtime()
    a = run(rt.execute(f, request_id="x")).value
    b = run(rt.execute(f, request_id="x")).value
    c = run(rt.execute(f, request_id="y")).value
    assert a == b
    assert a[0] != a[1]
    assert a != c


def test_config_must_be_data():
    with pytest.raises(DataError):
        Runtime(config={"k": {1, 2}})


# Domains


def test_domains_are_recorded():
    @cell(effects("email"), domain=UNIQUE)
    async def send(ctx) -> None: ...

    @cell
    async def f(ctx) -> None:
        await kv_put(ctx, "a", 1)
        await send(ctx)
        with ctx.domain("audit"):
            await kv_put(ctx, "b", 2)
            await kv_get(ctx, "a")
            with ctx.domain(UNIQUE):
                await kv_put(ctx, "c", 3)
        await kv_put(ctx, "d", 4)

    rt, _ = runtime()
    r = run(rt.run(f))
    assert [e.domain for e in r.journal.entries] == [MAIN, UNIQUE, "audit", None, UNIQUE, MAIN]


def test_bad_domain():
    @cell
    async def f(ctx) -> None:
        with ctx.domain(""):
            pass

    rt, _ = runtime()
    assert isinstance(run(rt.run(f)).error, ValueError)


# Replay


def test_replay_detects_a_different_effect():
    @cell
    async def f(ctx) -> None:
        key = ctx.resource("key")  # not journaled: a determinism violation
        await kv_put(ctx, key, 1)

    world = World({"kv": {}})
    first = run(Runtime(resources={World: world, "key": "a"}).run(f))
    assert isinstance(first.outcome, Ok)
    other = Runtime(resources={World: world, "key": "b"})
    r = run(other.replay(f, first.journal))
    assert isinstance(r.error, DeterminismError)


def test_replay_reports_a_violation_even_if_the_call_is_not_awaited():
    @cell
    async def f(ctx) -> int:
        if ctx.resource("flag"):
            kv_put(ctx, "a", 1)
        else:
            kv_get(ctx, "a")
        return 0

    world = World({"kv": {"a": 1}})
    first = run(Runtime(resources={World: world, "flag": True}).run(f))
    r = run(Runtime(resources={World: world, "flag": False}).replay(f, first.journal))
    assert isinstance(r.error, DeterminismError)


def test_replay_executes_a_different_pure_call():
    @cell
    async def f(ctx) -> int:
        return await kv_get(ctx, ctx.resource("key"))

    world = World({"kv": {"a": 1, "b": 2}})
    first = run(Runtime(resources={World: world, "key": "a"}).run(f))
    r = run(Runtime(resources={World: world, "key": "b"}).replay(f, first.journal))
    assert r.value == 2
    assert not r.journal.entries[0].replayed


def test_replay_reproduces_sources():
    @cell
    async def f(ctx) -> tuple[float, float, str]:
        return (ctx.now(), ctx.random(), ctx.config("k"))

    first = run(Runtime(config={"k": "v1"}, clock=lambda: 1.0).run(f))
    later = Runtime(config={"k": "v2"}, clock=lambda: 2.0)
    assert run(later.replay(f, first.journal)).value == first.value


def test_replay_waits_for_calls_in_flight():
    """A journal taken while a call was still in flight, as deopt will take one (M2)."""
    executed = []

    @cell(effects("x"))
    async def slow(ctx) -> int:
        executed.append(True)
        return 0

    @cell
    async def f(ctx) -> int:
        return await slow(ctx)

    async def scenario():
        release = asyncio.Event()

        async def in_flight() -> int:
            await release.wait()
            return 7

        entry = Entry(
            seq=0, kind="call", target=slow.id, effectful=True, domain=MAIN,
            args={}, args_digest=digest({}), started=True,
        )
        entry.handle = Handle(asyncio.create_task(in_flight()), entry)
        journal = Journal(request_id="r", path=(), cell=f.id, args={}, entries=[entry])
        rt, _ = runtime()
        replay = asyncio.create_task(rt.replay(f, journal))
        await asyncio.sleep(0.01)
        assert not replay.done()
        release.set()
        return await replay

    r = run(scenario())
    assert r.value == 7
    assert executed == []
    assert r.journal.entries[0].replayed


def test_unconsumed_effects_are_reported():
    @cell
    async def f(ctx) -> None:
        if ctx.resource("write"):
            await kv_put(ctx, "a", 1)

    world = World({"kv": {}})
    first = run(Runtime(resources={World: world, "write": True}).run(f))
    r = run(Runtime(resources={World: world, "write": False}).replay(f, first.journal))
    assert r.value is None
    assert [e.target for e in r.unconsumed] == [kv_put.id]


def test_replay_of_a_nested_composite_uses_its_recorded_journal():
    @cell
    async def f(ctx) -> int:
        return await sum_keys(ctx, ("a", "b"))

    rt, world = runtime()
    first = run(rt.run(f))
    world.calls.clear()
    r = run(rt.replay(f, first.journal))
    assert r.value == 3
    assert world.calls == []
    assert r.journal.entries[0].child is first.journal.entries[0].child


def test_run_value_raises_the_error():
    @cell(pure)
    async def f(ctx) -> None:
        raise KeyError("k")

    rt, _ = runtime()
    r = run(rt.run(f))
    assert isinstance(r.outcome, Err)
    with pytest.raises(KeyError):
        r.value


def test_journal_format():
    rt, _ = runtime()
    r = run(rt.run(sum_keys, ("a", "b")))
    text = r.journal.format()
    assert text.splitlines()[0] == "sum_keys(keys=('a', 'b')) -> Ok(3)"
    assert "#1 kv_get(key='b') -> Ok(2)" in text
