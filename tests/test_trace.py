"""The tracer, checked on every example scenario and on targeted programs.

For each scenario:

- tracing doesn't change the request: the outcome and journal match eager
  execution, and every leaf call and effect happens exactly once, whether
  or not the trace broke;
- the graph matches its golden file in tests/golden (run with
  CELL_UPDATE_GOLDEN=1 to rewrite them after a deliberate change);
- the graph satisfies the IR's invariants and survives JSON round-trips.
"""

import asyncio
import json
import os
import pathlib
from collections import Counter

import pytest

from cell import UNIQUE, Runtime, cell, effects, op, outcome_digest, pure
from cell.graph import Const, Graph
from cell.trace import TraceError, trace
from examples import SCENARIOS
from examples.features import kv_get, kv_put
from examples.harness import Scenario, World

ROOT = pathlib.Path(__file__).parent.parent
GOLDEN = ROOT / "tests" / "golden"
UPDATE = os.environ.get("CELL_UPDATE_GOLDEN") == "1"

ids = [s.name for s in SCENARIOS]


def run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize("s", SCENARIOS, ids=ids)
def test_tracing_does_not_change_the_request(s: Scenario):
    eager, eager_world = run(s.run())
    traced, world = run(s.trace())
    assert outcome_digest(traced.run.outcome) == outcome_digest(eager.outcome)
    assert traced.run.journal.summary() == eager.journal.summary()
    assert Counter(world.calls) == Counter(eager_world.calls)
    assert Counter(e.key() for e in world.effects) == Counter(e.key() for e in eager_world.effects)
    assert traced.replayed == (not traced.graph.complete)


@pytest.mark.parametrize("s", SCENARIOS, ids=ids)
def test_golden_graph(s: Scenario):
    traced, _ = run(s.trace())
    path = GOLDEN / (s.name.replace("/", ".") + ".graph")
    text = traced.graph.format(root=str(ROOT)) + "\n"  # sites relative to the repo
    if UPDATE or not path.exists():
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)
    assert text == path.read_text()


@pytest.mark.parametrize("s", SCENARIOS, ids=ids)
def test_graph_invariants(s: Scenario):
    traced, _ = run(s.trace())
    check_invariants(traced.graph, traced.run.journal)


@pytest.mark.parametrize("s", SCENARIOS, ids=ids)
def test_json_roundtrip(s: Scenario):
    g = run(s.trace())[0].graph
    j = json.loads(json.dumps(g.to_json()))
    g2 = Graph.from_json(j)
    assert g2.to_json() == g.to_json()
    assert g2.hash == g.hash
    assert g2.format() == g.format()


def check_invariants(g: Graph, journal) -> None:
    assert [n.id for n in g.nodes] == list(range(len(g.nodes)))
    params = [n for n in g.nodes if n.kind == "param"]
    assert [n.attrs["name"] for n in params] == g.params
    assert g.nodes[: len(params)] == params
    for n in g.nodes:
        assert all(r < n.id for r in n.refs), n
        assert n.after is None or n.after < n.id
        if not n.effectful:
            assert not n.effect and n.after is None, n
            assert not n.path or n.kind == "deopt", n
    terminals = [n for n in g.nodes if n.kind in ("return", "deopt")]
    assert terminals == [g.nodes[-1]]
    # Calls and sources line up with the journal, in order.
    issued = [n for n in g.nodes if n.kind in ("call", "source")]
    seqs = [n.attrs["seq"] for n in issued]
    assert seqs == sorted(set(seqs))
    for n in issued:
        entry = journal.entry(n.attrs["seq"])
        assert entry.target == (n.attrs["cell"] if n.kind == "call" else n.attrs["name"])


# Targeted programs


def world():
    return World({"kv": {"a": 1, "b": 2, "s": "Hello"}})


def traced(cell_, *args, **kwargs):
    w = world()
    rt = Runtime(resources={World: w}, clock=lambda: 7.0)
    t = run(trace(rt, cell_, args, kwargs, request_id="t"))
    eager = run(Runtime(resources={World: World(world().tables)}, clock=lambda: 7.0).execute(
        cell_, args, kwargs, request_id="t"))
    assert outcome_digest(t.run.outcome) == outcome_digest(eager.outcome)
    check_invariants(t.graph, t.run.journal)
    return t


def lines(t) -> list[str]:
    """The graph's nodes, without sites, for compact assertions."""
    out = []
    for line in t.graph.format().splitlines()[1:]:
        parts = [p for p in line.strip().split("  ") if p]
        if parts and ".py:" in parts[-1]:
            parts = parts[:-1]
        out.append("  ".join(parts))
    return out


def deopt_reason(t) -> str:
    last = t.graph.nodes[-1]
    assert last.kind == "deopt", t.graph.format()
    return last.attrs["reason"]


def test_arithmetic_and_reflection():
    @cell
    async def f(ctx) -> int:
        a = await kv_get(ctx, "a")
        return (10 - a) * 2 + -a

    t = traced(f)
    assert lines(t) == [
        "%0 = call kv_get('a')  pure  seq=0",
        "%1 = op 10 - %0",
        "%2 = op %1 * 2",
        "%3 = op neg(%0)",
        "%4 = op %2 + %3",
        "return %4",
    ]
    assert t.run.value == 17


def test_comparisons_become_guards():
    @cell
    async def f(ctx) -> str:
        a = await kv_get(ctx, "a")
        return "big" if a > 0 else "small"

    t = traced(f)
    assert lines(t)[1:3] == ["%1 = op %0 > 0", "%2 = guard %1 == True"]


def test_fields_and_methods():
    @cell
    async def f(ctx) -> str:
        s = await kv_get(ctx, "s")
        return s.lower().replace("l", "L")

    t = traced(f)
    assert lines(t)[1:3] == ["%1 = op %0.lower()", "%2 = op %1.replace('l', 'L')"]
    assert t.run.value == "heLLo"


def test_disallowed_method_breaks():
    @cell
    async def f(ctx) -> str:
        s = await kv_get(ctx, "s")
        return s.format()

    assert "attribute 'format'" in deopt_reason(traced(f))


@pytest.mark.parametrize(
    "body, reason",
    [
        (lambda v: str(v), "str()"),
        (lambda v: f"{v}", "formatting"),
        (lambda v: repr(v), "repr()"),
        (lambda v: {v: 1}, "hash()"),
        (lambda v: isinstance(v, int), "type of a traced value"),
        (lambda v: int(v), "int()"),
        (lambda v: list(range(v)), "index"),
        (lambda v: v(), "calling a traced value"),
    ],
    ids=["str", "format", "repr", "hash", "isinstance", "int", "range", "call"],
)
def test_concretization_breaks(body, reason):
    @cell
    async def f(ctx) -> object:
        return body(await kv_get(ctx, "a"))

    t = traced(f)
    assert reason in deopt_reason(t)
    assert t.replayed


def test_len_and_iteration_guard_lengths():
    @cell
    async def f(ctx, xs: tuple[int, ...]) -> int:
        total = 0
        for x in xs:
            total = total + x
        return total + len(xs)

    t = traced(f, (1, 2))
    assert lines(t) == [
        "%0 = param xs",
        "%1 = op len(%0)",
        "%2 = guard %1 == 2",
        "%3 = op %0[0]",
        "%4 = op 0 + %3",
        "%5 = op %0[1]",
        "%6 = op %4 + %5",
        "%7 = op len(%0)",
        "%8 = guard %7 == 2",
        "%9 = op %6 + 2",
        "return %9",
    ]


def test_iterating_a_dict_breaks():
    @cell
    async def f(ctx, d: dict[str, int]) -> list[str]:
        return [k for k in d]

    assert "iterating over a traced dict" in deopt_reason(traced(f, {"a": 1}))


def test_none_bool_and_enum_are_concrete_with_guards():
    import enum

    class Tier(enum.Enum):
        GOLD = 1

    @cell(pure)
    async def lookup(ctx, key: str) -> object:
        return {"none": None, "bool": True, "enum": Tier.GOLD}[key]

    @cell
    async def f(ctx) -> str:
        n = await lookup(ctx, "none")
        b = await lookup(ctx, "bool")
        e = await lookup(ctx, "enum")
        return "ok" if n is None and b is True and e is Tier.GOLD else "wrong"

    t = traced(f)
    assert t.run.value == "ok"
    guards = [n for n in t.graph.nodes if n.kind == "guard"]
    assert [g.attrs["expected"] for g in guards] == [None, True, Tier.GOLD]


def test_cells_returning_none_need_no_guard():
    @cell
    async def f(ctx) -> None:
        await kv_put(ctx, "k", 1)

    assert lines(traced(f)) == ["%0 = call kv_put('k', 1)  effectful[main]  seq=0", "return None"]


def test_mutation_breaks():
    @cell
    async def f(ctx) -> list[int]:
        xs = await sorted_keys(ctx)
        xs[0] = 5
        return xs

    @cell(pure)
    async def sorted_keys(ctx) -> list[int]:
        return [1, 2]

    assert "immutable" in deopt_reason(traced(f))


def test_guard_budget():
    @cell
    async def f(ctx, xs: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(sorted(xs))

    t = traced(f, tuple(range(100, 0, -1)))
    assert "guards; use an @op" in deopt_reason(t)
    assert t.run.value == tuple(range(1, 101))


def test_ops_are_single_nodes():
    @op
    def total(xs: tuple[int, ...], *, scale: int) -> int:
        return sum(sorted(xs)) * scale

    @cell
    async def f(ctx, xs: tuple[int, ...]) -> int:
        return total(xs, scale=await kv_get(ctx, "b"))

    t = traced(f, (3, 1, 2))
    assert lines(t) == [
        "%0 = param xs",
        "%1 = call kv_get('b')  pure  seq=0",
        "%2 = op total(%0, %1)",
        "return %2",
    ]
    assert t.graph.nodes[2].attrs["kwargs"] == ["scale"]
    assert t.run.value == 12


def test_ops_without_traced_arguments_just_run():
    @op
    def double(x: int) -> int:
        return 2 * x

    @cell
    async def f(ctx) -> int:
        return double(21)

    assert lines(traced(f)) == ["return 42"]


def test_failing_op_breaks():
    @op
    def div(a: int, b: int) -> float:
        return a / b

    @cell
    async def f(ctx) -> float:
        return div(await kv_get(ctx, "a"), 0)

    t = traced(f)
    assert "op div raised ZeroDivisionError" in deopt_reason(t)
    assert isinstance(t.run.error, ZeroDivisionError)


def test_handle_to_op_breaks():
    @op
    def ident(x: int) -> int:
        return x

    @cell
    async def f(ctx) -> int:
        return ident(kv_get(ctx, "a"))

    assert "await it first" in deopt_reason(traced(f))


def test_ctx_escapes_break():
    @cell
    async def f(ctx) -> str:
        return ctx.request_id

    @cell
    async def g(ctx) -> object:
        return ctx.resource(World)

    assert "request_id" in deopt_reason(traced(f))
    assert "resource" in deopt_reason(traced(g))


def test_a_break_is_not_caught_by_except_exception():
    @cell
    async def f(ctx) -> str:
        a = await kv_get(ctx, "a")
        try:
            return "value " + str(a)
        except Exception:
            return "fallback"

    t = traced(f)
    assert t.run.value == "value 1"
    assert t.replayed


def test_a_swallowed_break_still_replays_and_issues_nothing_more():
    @cell
    async def f(ctx) -> int:
        a = await kv_get(ctx, "a")
        try:
            str(a)
        except BaseException:
            pass
        await kv_put(ctx, "k", 1)  # refused: the invocation is stopped
        return 0

    w = world()
    t = run(trace(Runtime(resources={World: w}), f))
    assert t.replayed and t.run.value == 0
    assert [e.name for e in w.effects] == ["kv_put"]


def test_body_errors_break_and_replay():
    @cell
    async def f(ctx) -> int:
        a = await kv_get(ctx, "a")
        if a > 0:
            raise ValueError("positive")
        return a

    t = traced(f)
    assert "the body raised ValueError" in deopt_reason(t)
    assert isinstance(t.run.error, ValueError)


def test_tracers_cannot_be_used_after_the_trace():
    leaked = []

    @cell
    async def f(ctx) -> int:
        a = await kv_get(ctx, "a")
        leaked.append(a)
        return a

    traced(f)
    with pytest.raises(TraceError):
        leaked[0] + 1
    assert repr(leaked[0]) == "<tracer %0>"


# Control edges (DESIGN §4.3, §4.4)


@cell(effects("audit"), domain="audit")
async def log(ctx, event: str) -> None:
    ctx.resource(World).effect(ctx, "log", event=event)


@cell(effects("mail"), domain=UNIQUE)
async def mail(ctx, to: str) -> None:
    ctx.resource(World).effect(ctx, "mail", to=to)


def edges(t, seq):
    n = next(n for n in t.graph.nodes if n.kind == "call" and n.attrs["seq"] == seq)
    return (n.effect, n.path, n.after)


def test_effects_in_a_domain_are_ordered():
    @cell
    async def f(ctx) -> None:
        await kv_put(ctx, "a", 1)  # %0
        kv_put(ctx, "b", 2)  # %1: waits for %0 to complete
        kv_put(ctx, "c", 3)  # %2: issued after %1, which was never awaited

    t = traced(f)
    assert edges(t, 1) == ([0], [], None)
    assert edges(t, 2) == ([0], [], 1)


def test_effects_in_other_domains_are_not_ordered():
    @cell
    async def f(ctx) -> None:
        await log(ctx, "start")  # %0, audit
        await kv_put(ctx, "a", 1)  # %1, main: no edge to %0
        await mail(ctx, "x")  # %2, UNIQUE: no edges at all
        await log(ctx, "end")  # %3, audit: after %0

    t = traced(f)
    assert edges(t, 1) == ([], [], None)
    assert edges(t, 2) == ([], [], None)
    assert edges(t, 3) == ([0], [], None)


def test_effects_follow_the_guards_that_lead_to_them():
    @cell
    async def f(ctx) -> None:
        a = await kv_get(ctx, "a")  # %0
        if a > 0:  # %1, %2
            await kv_put(ctx, "k", 1)  # path edge to the guard, which implies %0 and %1

    t = traced(f)
    assert edges(t, 1) == ([], [2], None)


def test_pure_calls_have_no_control_edges():
    @cell
    async def f(ctx) -> int:
        await kv_put(ctx, "a", 1)
        return await kv_get(ctx, "a")

    assert edges(traced(f), 1) == ([], [], None)


def test_passing_a_handle_is_not_observing_it():
    @cell
    async def f(ctx) -> None:
        a = kv_get(ctx, "a")  # %0, never awaited
        await kv_put(ctx, "k", a)  # data edge to %0, and no path edge

    t = traced(f)
    call = t.graph.nodes[1]
    assert call.inputs == [Const("k"), 0]
    assert (call.effect, call.path) == ([], [])
