"""Cells with seeded lint violations, for tests/test_static.py.

Each violating line ends with `# expect: <rule>`; the test checks that
lint finds exactly these, at these lines.
"""

import asyncio
import datetime
import json
import os
import random
import subprocess
import time
import uuid
from random import randint

from cell import cell, op
from examples.features import kv_get, kv_put

CACHE: dict[int, int] = {}
LIMIT = 10  # immutable: fine to read


@op
def double(x: int) -> int:
    return 2 * x


@cell
async def nondeterministic(ctx) -> float:
    a = await kv_get(ctx, "a")
    t = time.time()  # expect: nondeterminism
    r = random.random()  # expect: nondeterminism
    i = randint(0, 3)  # expect: nondeterminism
    u = uuid.uuid4()  # expect: nondeterminism
    d = datetime.datetime.now()  # expect: nondeterminism
    e = os.environ.get("HOME")  # expect: nondeterminism
    return a + t + r + i + len(str(u)) + d.second + len(e or "")


@cell
async def io(ctx) -> None:
    await kv_get(ctx, "a")
    print("hello")  # expect: io
    open("/dev/null").close()  # expect: io
    subprocess.run(["true"])  # expect: io
    time.sleep(0)  # expect: io


@cell
async def global_state(ctx) -> int:
    global LIMIT  # expect: global-write
    a = await kv_get(ctx, "a")
    CACHE[a] = a  # expect: mutable-global
    return a + LIMIT


@cell
async def tasks(ctx) -> None:
    t = asyncio.create_task(asyncio.sleep(0))  # expect: task
    await asyncio.wait([t])  # expect: race
    for f in asyncio.as_completed([t]):  # expect: race
        await f
    await kv_get(ctx, "a")


@cell
async def ctx_misuse(ctx) -> str:
    ctx.resource("world")  # expect: resource
    await kv_put(ctx, "k", 1)
    return ctx.request_id  # expect: ctx-escape


@cell
async def tracing(ctx, n: int) -> str:
    a = await kv_get(ctx, "a")
    s = f"a={a}"  # expect: format
    t = repr(a)  # expect: convert
    for _ in range(n):  # expect: range
        pass
    k = type(a)  # expect: invisible
    same = a is n  # expect: identity
    fine = a is None
    top = sorted([a, n])  # expect: ordering
    j = json.dumps(a)  # expect: opaque-call
    try:
        pass
    except BaseException:  # expect: catch-all
        pass
    return s + t + str(k) + str(same) + str(fine) + str(top) + j  # expect: convert


@cell
async def clean(ctx, n: int) -> tuple:
    """Nothing to report: the patterns the tracer handles."""
    a = await kv_get(ctx, "a")
    handles = [kv_get(ctx, k) for k in ("a", "b")]
    values = await asyncio.gather(*handles)
    total = sum(values) + len(values) + double(a) + LIMIT
    if a is None or n > a:
        await kv_put(ctx, "total", total)
    return (total, tuple(values), a)
