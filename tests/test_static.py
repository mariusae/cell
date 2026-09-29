"""Static extraction (DESIGN §9, milestone M3).

- Lint finds exactly the violations seeded in tests/lint_cases.py, each
  marked with `# expect: <rule>` on its line.
- The examples lint clean, apart from the one real trace-breaking pattern.
- The static call graph includes every call any trace records: over the
  example scenarios, and over the randomly generated programs.
"""

import asyncio
import pathlib
import re

import pytest

import lint_cases
from cell import Runtime, cell
from cell.static import call_graph, cells_of, extract, lint
from cell.trace import trace
from examples import SCENARIOS, checkout, features, feed, home
from examples.harness import World
from test_random_programs import PROGRAMS, make

ERRORS = {"nondeterminism", "io", "global-write", "mutable-global", "task", "race", "resource"}


def run(coro):
    return asyncio.run(coro)


def expected(module) -> set[tuple[int, str]]:
    lines = pathlib.Path(module.__file__).read_text().splitlines()
    return {(i + 1, m.group(1)) for i, line in enumerate(lines) if (m := re.search(r"# expect: ([\w-]+)", line))}


def found(module) -> set[tuple[int, str]]:
    return {(int(f.site.rsplit(":", 1)[1]), f.rule) for c in cells_of(module) for f in lint(c)}


def test_lint_finds_the_seeded_violations():
    assert found(lint_cases) == expected(lint_cases)


def test_severities():
    for c in cells_of(lint_cases):
        for f in lint(c):
            if f.rule == "io" and "prints" in f.message:
                assert f.severity == "warning"
            else:
                assert f.severity == ("error" if f.rule in ERRORS else "warning"), f


def test_clean_code_has_no_findings():
    assert lint(lint_cases.clean) == []


def test_examples_lint_clean_except_the_data_dependent_loop():
    findings = [f for m in (home, feed, checkout, features) for c in cells_of(m) for f in lint(c)]
    assert [(f.cell, f.rule) for f in findings] == [("examples.features.loop", "range")]


def test_leaf_cells_are_not_linted():
    ex = extract(checkout.charge)
    assert ex.leaf and ex.findings == [] and ex.uses_resources


def test_call_graph_of_an_example():
    ex = extract(home.home)
    assert {c.id.rsplit(".", 1)[1] for c in ex.calls} == {"get_user", "get_prefs", "get_items", "rank"}
    assert {o.id.rsplit(".", 1)[1] for o in ex.ops} == {"greeting"}


def test_helpers_are_analyzed_as_part_of_the_body():
    ex = extract(features.helpers)
    assert [h.__name__ for h in ex.helpers] == ["get_both"]
    assert {c.id for c in ex.calls} == {features.kv_get.id}


def test_transitive_call_graph():
    graph = call_graph([features.nested])
    assert graph[features.nested] == {features.sum_keys}
    assert graph[features.sum_keys] == {features.kv_get}
    assert graph[features.kv_get] == set()


def test_references_count_even_if_not_called():
    @cell
    async def choose(ctx, fast: bool) -> int:
        getter = features.kv_get if fast else features.boom
        return await getter(ctx, "a")

    assert extract(choose).calls == {features.kv_get, features.boom}


def test_closures_over_mutable_state_are_flagged():
    seen = []

    @cell
    async def remembers(ctx) -> int:
        a = await features.kv_get(ctx, "a")
        seen.append(a)
        return len(seen)

    assert [f.rule for f in lint(remembers)] == ["mutable-global"]


def called(graph) -> set[str]:
    return {n.attrs["cell"] for n in graph.nodes if n.kind == "call"}


@pytest.mark.parametrize("s", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_static_calls_include_traced_calls(s):
    graph = run(s.trace())[0].graph
    assert called(graph) <= {c.id for c in extract(s.cell).calls}


@pytest.mark.parametrize("i", range(PROGRAMS))
def test_static_calls_include_traced_calls_of_random_programs(i):
    source, prog, inputs = make(i)
    static = {c.id for c in extract(prog).calls}
    assert static, source  # the source was found
    for args in inputs:
        graph = run(trace(Runtime(resources={World: World()}), prog, args)).graph
        assert called(graph) <= static, source
