"""Small programs, each exercising one piece of the programming model.

Where the application examples show realistic shapes, these pin down the
semantics of DESIGN §1: sources, issue-on-call, structured concurrency,
pushdown, helpers, nesting, loops, exceptions and domains.
"""

from __future__ import annotations

import asyncio

from cell import cell, effects, pure

from .harness import NOW, Scenario, service


class Boom(Exception):
    pass


# Leaf cells over a key-value table.


@cell(pure)
async def kv_get(ctx, key: str) -> int:
    w = await service(ctx, "kv_get")
    return w.table("kv")[key]


@cell(effects("kv"))
async def kv_put(ctx, key: str, value: int) -> None:
    w = await service(ctx, "kv_put")
    w.table("kv")[key] = value
    w.effect(ctx, "kv_put", key=key, value=value)


@cell(pure)
async def boom(ctx, what: str) -> int:
    await service(ctx, "boom")
    raise Boom(what)


@cell(effects("kv"))
async def boom_write(ctx, what: str) -> None:
    await service(ctx, "boom_write")
    raise Boom(what)


def tables():
    return {"kv": {"a": 1, "b": 2}}


# Programs.


@cell
async def sources(ctx) -> tuple[float, str, bool]:
    """Time, configuration and randomness all come from the ctx, and are journaled."""
    return (ctx.now(), ctx.config("greeting", "hi"), 0.0 <= ctx.random() < 1.0)


@cell
async def fire_and_wait(ctx) -> int:
    """A call is issued when made; the cell waits for it even if never awaited."""
    kv_put(ctx, "z", 26)
    return 0


@cell
async def unawaited_write_fails(ctx) -> int:
    """An unawaited effectful call that fails fails the cell."""
    boom_write(ctx, "write")
    return 1


@cell
async def unawaited_read_fails(ctx) -> int:
    """An unawaited pure call that fails is discarded."""
    boom(ctx, "read")
    return 1


async def get_both(ctx) -> int:
    # A plain helper: not a cell, but it can call cells with the ctx.
    return await kv_get(ctx, "a") + await kv_get(ctx, "b")


@cell
async def helpers(ctx) -> int:
    return await get_both(ctx)


@cell
async def sum_keys(ctx, keys: tuple[str, ...]) -> int:
    handles = [kv_get(ctx, k) for k in keys]
    return sum([await h for h in handles])


@cell
async def nested(ctx) -> int:
    """Composite cells calling composite cells."""
    return await sum_keys(ctx, ("a", "b")) + await sum_keys(ctx, ("a",))


@cell
async def concurrent(ctx) -> int:
    """Handles work with asyncio.gather."""
    values = await asyncio.gather(kv_get(ctx, "a"), kv_get(ctx, "b"), kv_get(ctx, "a"))
    return sum(values)


@cell
async def loop(ctx, n: int) -> int:
    """A data-dependent number of calls."""
    total = 0
    for _ in range(n):
        total += await kv_get(ctx, "a")
    await kv_put(ctx, "total", total)
    return total


@cell
async def fallback(ctx) -> int:
    """Catching a call's error at its await."""
    try:
        return await boom(ctx, "primary")
    except Boom:
        return await kv_get(ctx, "b")


@cell
async def pushdown_failure(ctx) -> None:
    """A failed handle passed to an effectful call: the call never starts (DESIGN §1.5)."""
    value = boom(ctx, "value")
    await kv_put(ctx, "y", value)


@cell
async def pushdown(ctx) -> int:
    """Handles nested inside arguments resolve before the callee starts."""
    total = sum_keys(ctx, ("a", "b"))
    await kv_put(ctx, "sum", total)
    return await kv_get(ctx, "a")


@cell
async def domains(ctx) -> None:
    """Choosing an effect domain at the call site (DESIGN §4.4)."""
    with ctx.domain("audit"):
        await kv_put(ctx, "log", 1)
    await kv_put(ctx, "k", 2)


SCENARIOS = [
    Scenario("features/sources", sources, tables=tables, config={"greeting": "hello"}, expect=(NOW, "hello", True)),
    Scenario("features/sources-default", sources, tables=tables, expect=(NOW, "hi", True)),
    Scenario(
        "features/fire-and-wait",
        fire_and_wait,
        tables=tables,
        expect=0,
        effects=(("kv_put", {"key": "z", "value": 26}),),
    ),
    Scenario("features/unawaited-write-fails", unawaited_write_fails, tables=tables, raises=Boom),
    Scenario("features/unawaited-read-fails", unawaited_read_fails, tables=tables, expect=1),
    Scenario("features/helpers", helpers, tables=tables, expect=3),
    Scenario("features/nested", nested, tables=tables, expect=4),
    Scenario("features/concurrent", concurrent, tables=tables, expect=4),
    Scenario(
        "features/loop",
        loop,
        (3,),
        tables=tables,
        expect=3,
        effects=(("kv_put", {"key": "total", "value": 3}),),
    ),
    Scenario("features/loop-zero", loop, (0,), tables=tables, expect=0, effects=(("kv_put", {"key": "total", "value": 0}),)),
    Scenario("features/fallback", fallback, tables=tables, expect=2),
    Scenario("features/pushdown-failure", pushdown_failure, tables=tables, raises=Boom),
    Scenario(
        "features/pushdown",
        pushdown,
        tables=tables,
        expect=1,
        effects=(("kv_put", {"key": "sum", "value": 3}),),
    ),
    Scenario(
        "features/domains",
        domains,
        tables=tables,
        effects=(("kv_put", {"key": "log", "value": 1}), ("kv_put", {"key": "k", "value": 2})),
    ),
]
