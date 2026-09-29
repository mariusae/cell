"""Property tests over randomly generated cell programs (DESIGN §9).

Each program is generated from a small grammar: awaited and unawaited
calls to pure and effectful leaves, handles passed as arguments, ops,
arithmetic, branches on call results, caught errors, fixed and
data-dependent loops, and an effect domain. Each is run on a few random
inputs. For every input the program is traced, and each resulting graph
is run compiled on every input. A deopt is also forced at each node of one
graph. Every run must match eager execution: the same outcome, journal and
effects (DESIGN §2's central invariant).

Leaves are deterministic functions of their arguments that fail on some
of them, so calls fail and deopts happen too. Effectful leaves don't read
state, because the order of effects in a domain that nothing waited for is
not defined (DESIGN §4.3): only their issue order is.
"""

import asyncio
import linecache
import random
from collections import Counter

import pytest

from cell import Runtime, cell, effects, op, outcome_digest, pure
from cell.trace import trace
from examples.harness import World

SEED = 20260928
PROGRAMS = 60
INPUTS = 4


class Boom(Exception):
    pass


@cell(pure)
async def get(ctx, k: int) -> int:
    await ctx.resource(World).enter(ctx, "get")
    if k % 11 == 10:
        raise Boom(k)
    return (k * 7 + 3) % 23


@cell(effects("kv"))
async def put(ctx, k: int, v: int) -> None:
    w = await ctx.resource(World).enter(ctx, "put")
    w.effect(ctx, "put", k=k, v=v)


@cell(effects("kv"))
async def bump(ctx, k: int) -> int:
    w = await ctx.resource(World).enter(ctx, "bump")
    if k % 13 == 12:
        raise Boom(k)
    w.effect(ctx, "bump", k=k)
    return (k + 5) % 19


@cell(effects("audit"), domain="audit")
async def log(ctx, v: int) -> None:
    w = await ctx.resource(World).enter(ctx, "log")
    w.effect(ctx, "log", v=v)


@op
def mix(a: int, b: int) -> int:
    return (a * 31 + b) % 97


NAMESPACE = {"cell": cell, "get": get, "put": put, "bump": bump, "log": log, "mix": mix, "Boom": Boom}


class Generator:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.lines: list[str] = []
        self.vars = ["a", "b"]
        self.handles: list[str] = []
        self.n = 0
        self.callee: str | None = None  # a composite cell the program may call

    def var(self) -> str:
        return self.rng.choice(self.vars)

    def expr(self) -> str:
        return self.rng.choice([
            lambda: self.var(),
            lambda: str(self.rng.randrange(20)),
            lambda: f"({self.var()} + {self.rng.randrange(1, 9)}) % 20",
            lambda: f"{self.var()} * {self.var()} % 20",
        ])()

    def fresh(self) -> str:
        self.n += 1
        return f"v{self.n}"

    def emit(self, indent: int, line: str) -> None:
        self.lines.append("    " * indent + line)

    def target(self, top: bool) -> str:
        # New variables only at the top level, so every variable is always defined.
        if top:
            name = self.fresh()
            self.vars.append(name)
            return name
        return self.rng.choice([v for v in self.vars if v not in ("a", "b")] or ["a"])

    def block(self, indent: int, depth: int, count: int, top: bool) -> None:
        for _ in range(count):
            self.stmt(indent, depth, top)

    def stmt(self, indent: int, depth: int, top: bool) -> None:
        kinds = ["get", "arith", "op", "put", "put_bg", "log", "bump", "try"]
        if top:
            kinds += ["handle", "handle", "fixed_loop"]
        if self.handles:
            kinds += ["await_handle", "put_handle"]
        if depth < 2:
            kinds += ["if", "if"]
        if self.rng.random() < 0.05:
            kinds.append("data_loop")
        if self.callee is not None:
            kinds += ["callee", "callee"] + (["callee_bg"] if top else [])
        kind = self.rng.choice(kinds)
        if kind == "get":
            e = self.expr()
            self.emit(indent, f"{self.target(top)} = await get(ctx, {e})")
        elif kind == "arith":
            e = self.expr()
            self.emit(indent, f"{self.target(top)} = {e}")
        elif kind == "op":
            x, y = self.var(), self.var()
            self.emit(indent, f"{self.target(top)} = mix({x}, {y})")
        elif kind == "put":
            self.emit(indent, f"await put(ctx, {self.expr()}, {self.expr()})")
        elif kind == "put_bg":
            self.emit(indent, f"put(ctx, {self.expr()}, {self.expr()})")
        elif kind == "log":
            self.emit(indent, f"await log(ctx, {self.expr()})")
        elif kind == "bump":
            e = self.expr()
            self.emit(indent, f"{self.target(top)} = await bump(ctx, {e})")
        elif kind == "try":
            e = self.expr()
            t = self.target(top)
            self.emit(indent, "try:")
            self.emit(indent + 1, f"{t} = await get(ctx, {e})")
            self.emit(indent, "except Boom:")
            self.emit(indent + 1, f"{t} = 99")
        elif kind == "handle":
            h = f"h{self.n}"
            self.n += 1
            self.emit(indent, f"{h} = get(ctx, {self.expr()})")
            self.handles.append(h)
        elif kind == "await_handle":
            h = self.rng.choice(self.handles)
            self.emit(indent, f"{self.target(top)} = await {h}")
        elif kind == "put_handle":
            h = self.rng.choice(self.handles)
            self.emit(indent, f"await put(ctx, {self.expr()}, {h})")
        elif kind == "fixed_loop":
            t = self.target(True)
            self.emit(indent, f"{t} = {self.var()}")
            self.emit(indent, "for _ in range(2):")
            self.emit(indent + 1, f"{t} = await get(ctx, ({t} + 1) % 20)")
        elif kind == "data_loop":
            t = self.target(top)
            self.emit(indent, f"for _ in range({self.var()} % 3):")
            self.emit(indent + 1, f"{t} = await get(ctx, ({t} + 2) % 20)")
        elif kind == "callee":
            e1, e2 = self.expr(), self.expr()
            self.emit(indent, f"{self.target(top)} = (await {self.callee}(ctx, {e1}, {e2}))[-1]")
        elif kind == "callee_bg":
            self.emit(indent, f"{self.callee}(ctx, {self.expr()}, {self.expr()})")
        elif kind == "if":
            self.emit(indent, f"if {self.expr()} > {self.expr()}:")
            self.block(indent + 1, depth + 1, self.rng.randrange(1, 3), False)
            self.emit(indent, "else:")
            self.block(indent + 1, depth + 1, self.rng.randrange(1, 3), False)

    def program(self) -> str:
        self.emit(0, "@cell")
        self.emit(0, "async def prog(ctx, a: int, b: int) -> tuple:")
        self.block(1, 0, self.rng.randrange(3, 9), True)
        self.emit(1, "return (" + ", ".join(self.vars) + ",)")
        return "\n".join(self.lines) + "\n"


def assert_effects_match(r, eager, world, eager_world, context=""):
    """Every effect eager execution performs happens exactly once. Any other
    effect must come from a call replay never reached, which domains permit
    only when the cell failed (DESIGN §4.4)."""
    ours = Counter(e.key() for e in world.effects)
    theirs = Counter(e.key() for e in eager_world.effects)
    assert not theirs - ours, context
    extra = ours - theirs
    ran_ahead = {j.path + (seq,) for j in r.journal.walk() for seq in j.unconsumed}
    assert all(any(path[: len(p)] == p for p in ran_ahead) for path, _, _ in extra), context
    assert not ran_ahead or r.error is not None, context


def make(i: int):
    rng = random.Random(SEED + i)
    source = Generator(rng).program()
    namespace = dict(NAMESPACE, __name__=f"generated_{i}")
    filename = f"<generated {i}>"
    # Register the source, so inspect (and static extraction) can find it.
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, "exec"), namespace)
    inputs = [(rng.randrange(20), rng.randrange(20)) for _ in range(INPUTS)]
    return source, namespace["prog"], inputs


def run(coro):
    return asyncio.run(coro)


def eager(prog, args):
    w = World()
    return run(Runtime(resources={World: w}).execute(prog, args, request_id="r")), w


def check(prog, graph, args, source, fail_at=None):
    expected, ew = eager(prog, args)
    w = World()
    rt = Runtime(resources={World: w})
    rt.install(graph).fail_at = fail_at
    r = run(rt.execute(prog, args, request_id="r"))
    context = f"\n{source}\nargs={args} fail_at={fail_at}\n{graph.format()}\n{r.journal.format()}"
    assert outcome_digest(r.outcome) == outcome_digest(expected.outcome), context
    assert r.journal.summary() == expected.journal.summary(), context
    assert_effects_match(r, expected, w, ew, context)
    assert not Counter(ew.calls) - Counter(w.calls), context
    return r


@pytest.mark.parametrize("i", range(PROGRAMS))
def test_random_program(i: int):
    source, prog, inputs = make(i)
    graphs = []
    for args in inputs:
        expected, ew = eager(prog, args)
        w = World()
        traced = run(trace(Runtime(resources={World: w}), prog, args, request_id="r"))
        # Tracing doesn't change the request.
        assert outcome_digest(traced.run.outcome) == outcome_digest(expected.outcome), source
        assert Counter(e.key() for e in w.effects) == Counter(e.key() for e in ew.effects), source
        graphs.append(traced.graph)
    for graph in graphs:
        for args in inputs:
            check(prog, graph, args, source)
    for k in range(len(graphs[0].nodes)):
        check(prog, graphs[0], inputs[0], source, fail_at=k)


def test_programs_exercise_compiled_runs_and_deopts():
    """The generator is useful: some runs complete compiled, others deopt."""
    modes = Counter()
    for i in range(PROGRAMS):
        _, prog, inputs = make(i)
        graph = run(trace(Runtime(resources={World: World()}), prog, inputs[0], request_id="r")).graph
        for args in inputs:
            rt = Runtime(resources={World: World()})
            rt.install(graph)
            modes[run(rt.execute(prog, args, request_id="r")).journal.mode] += 1
    assert modes["compiled"] > PROGRAMS // 2 and modes["deopt"] > PROGRAMS // 2, modes
