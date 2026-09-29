"""The demo server: walks an example scenario through the system.

For a scenario, /api/run returns each step:

1. the cell's source, and lint findings for it;
2. an eager run: its journal, outcome and effects;
3. a trace: the graph it recorded, as IR and as a Mermaid flowchart;
4. the graph after the selected rewrites (inline, fold, dedup);
5. a compiled run of that graph: whether it ran compiled or deopted, and
   whether its outcome and effects match the eager run.

/api/simulate runs many requests against simulated services (sim.py) for
eager execution, the traced graph, and the rewritten graph (optionally
with caching), and returns latency distributions and throughput.
"""

from __future__ import annotations

import asyncio
import inspect
import os
from collections import Counter
from typing import Any

from flask import Flask, jsonify, render_template, request

from cell import outcome_digest
from cell.graph import Graph
from cell.passes import optimize
from cell.static import call_graph, lint
from examples import SCENARIOS, scenario
from examples.bench import PROFILES, callee_graphs
from examples.harness import Scenario, World

from . import sim
from .mermaid import to_mermaid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True  # a demo: pick up edits to the page without a restart


def _flag(name: str, default: bool = True) -> bool:
    value = request.args.get(name) if request.method == "GET" else (request.get_json() or {}).get(name)
    if value is None:
        return default
    return value in (True, "1", "true", "on")


def _graph_view(graph: Graph) -> dict[str, Any]:
    kinds: dict[str, int] = {}
    for n in graph.nodes:
        kinds[n.kind] = kinds.get(n.kind, 0) + 1
    return {
        "ir": graph.format(root=ROOT),
        "mermaid": to_mermaid(graph),
        "nodes": len(graph.nodes),
        "kinds": kinds,
        "complete": graph.complete,
        "hash": graph.hash,
    }


def _pure_cells(s: Scenario) -> list[Any]:
    return [c for c in call_graph([s.cell]) if not c.effectful]


def _graphs(s: Scenario, source: Scenario, inlining: bool, folding: bool, dedupe: bool) -> tuple[Any, Graph]:
    async def go() -> tuple[Any, Graph]:
        traced, _ = await source.trace()
        callees = await callee_graphs(source, traced.run.journal)
        rewritten = optimize(traced.graph, callees, inlining=inlining, folding=folding, dedupe=dedupe)
        return traced, rewritten

    return asyncio.run(go())


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.get("/api/scenarios")
def scenarios() -> Any:
    out = []
    for s in SCENARIOS:
        doc = inspect.getdoc(s.cell.fn) or inspect.getdoc(inspect.getmodule(s.cell.fn)) or ""
        out.append(
            {
                "name": s.name,
                "cell": s.cell.id.rsplit(".", 1)[-1],
                "module": s.cell.fn.__module__,
                "args": ", ".join([*(repr(a) for a in s.args), *(f"{k}={v!r}" for k, v in s.kwargs.items())]),
                "doc": doc.split("\n\n")[0],
                "siblings": [t.name for t in SCENARIOS if t.cell is s.cell],
            }
        )
    return jsonify(out)


@app.get("/api/run")
def run() -> Any:
    s = scenario(request.args["name"])
    source = scenario(request.args.get("trace_from") or s.name)
    if source.cell is not s.cell:
        return jsonify({"error": "trace_from must be a scenario of the same cell"}), 400
    inlining, folding, dedupe, caching = _flag("inline"), _flag("fold"), _flag("dedup"), _flag("cache", False)

    eager, eager_world = asyncio.run(s.run())
    traced, rewritten = _graphs(s, source, inlining, folding, dedupe)

    world = s.world()
    rt = s.runtime(world)
    for c in _pure_cells(s) if caching else []:
        rt.cache(c, 60.0)
    rt.install(rewritten)
    compiled = asyncio.run(rt.execute(s.cell, s.args, s.kwargs, request_id="req"))
    if caching:  # a second request shows the cache at work
        compiled = asyncio.run(rt.execute(s.cell, s.args, s.kwargs, request_id="req2"))
    batch = _batch(s, rewritten)

    return jsonify(
        {
            "name": s.name,
            "source": inspect.getsource(s.cell.fn),
            "lint": [f.format(ROOT) for f in lint(s.cell)],
            "cached_cells": [c.id.rsplit(".", 1)[-1] for c in _pure_cells(s)],
            "eager": {
                "journal": eager.journal.format(),
                "outcome": repr(eager.outcome),
                "effects": [f"{name} {args}" for name, args in eager_world.effects_in_path_order()],
            },
            "traced": {
                **_graph_view(traced.graph),
                "from": source.name,
                "replayed": traced.replayed,
                "outcome": repr(traced.run.outcome),
            },
            "rewritten": {**_graph_view(rewritten), "passes": {"inline": inlining, "fold": folding, "dedup": dedupe}},
            "compiled": {
                "journal": compiled.journal.format(),
                "outcome": repr(compiled.outcome),
                "mode": compiled.journal.mode,
                "deopt": compiled.journal.deopt,
                "cached": caching,
                "matches_eager": outcome_digest(compiled.outcome) == outcome_digest(eager.outcome)
                and compiled.journal.effects() == eager.journal.effects(),
                "effects": [f"{name} {args}" for name, args in world.effects_in_path_order()],
            },
            "batch": batch,
        }
    )


def _batch(s: Scenario, graph: Graph) -> dict[str, Any]:
    """Every scenario of the cell, twice, as one batch through the graph (DESIGN §10).

    Each request is a lane: it deopts on its own if a guard fails, and the
    lanes share vector calls. Compared with running the same requests one
    at a time through the same graph.
    """
    siblings = [t for t in SCENARIOS if t.cell is s.cell] * 2

    async def go() -> tuple[list[Any], Any, Any, Any]:
        world = World(s.tables())
        rt = s.runtime(world)
        rt.install(graph)
        runs = await rt.execute_batch(s.cell, [(t.args, t.kwargs, f"lane{i}") for i, t in enumerate(siblings)])
        alone = World(s.tables())
        rt_alone = s.runtime(alone)
        rt_alone.install(graph)
        for i, t in enumerate(siblings):
            await rt_alone.execute(t.cell, t.args, t.kwargs, request_id=f"alone{i}")
        return runs, world, rt, alone

    runs, world, rt, alone = asyncio.run(go())
    lanes = []
    for t, r in zip(siblings, runs):
        eager, _ = asyncio.run(t.run())
        lanes.append(
            {
                "scenario": t.name,
                "mode": r.journal.mode,
                "deopt": r.journal.deopt,
                "matches_eager": outcome_digest(r.outcome) == outcome_digest(eager.outcome)
                and r.journal.effects() == eager.journal.effects(),
                "batched_calls": sum(e.batched for j in r.journal.walk() for e in j.entries),
            }
        )
    count = lambda w: dict(sorted(Counter(c.name for c in w.calls).items()))  # noqa: E731
    return {
        "lanes": lanes,
        "journal": runs[siblings.index(s)].journal.format(),
        "vector_calls": rt.vector_stats["calls"],
        "vector_elements": rt.vector_stats["elements"],
        "calls": count(world),
        "calls_one_by_one": count(alone),
    }


def _settings(body: dict[str, Any], requests: int = 400) -> sim.Settings:
    return sim.Settings(
        requests=max(1, min(int(body.get("requests", requests)), 5000)),
        clients=max(1, min(int(body.get("clients", 32)), 256)),
        capacity=max(1, min(int(body.get("capacity", 8)), 256)),
        median_ms=max(0.1, float(body.get("median_ms", 10.0))),
        sigma=max(0.0, min(float(body.get("sigma", 0.5)), 3.0)),
        seed=int(body.get("seed", 1)),
    )


def _variants(s: Scenario, body: dict[str, Any]) -> list[sim.Variant]:
    """eager; the traced graph; the rewritten graph (and cache); and that, batched."""
    source = scenario(body.get("trace_from") or s.name)
    inlining, folding, dedupe, caching = _flag("inline"), _flag("fold"), _flag("dedup"), _flag("cache", False)
    traced, rewritten = _graphs(s, source, inlining, folding, dedupe)
    passes = [p for p, on in (("inline", inlining), ("fold", folding), ("dedup", dedupe), ("cache", caching)) if on]
    cache = tuple(_pure_cells(s)) if caching else ()
    window = max(0.0, float(body.get("batch_window_ms", 2.0))) / 1000
    size = max(1, min(int(body.get("batch_max", 16)), 1024))
    label = "compiled + " + (", ".join(passes) if passes else "no rewrites")
    return [
        sim.Variant("eager"),
        sim.Variant("compiled", traced.graph),
        sim.Variant(label, rewritten, cache),
        sim.Variant(f"{label} + batching", rewritten, cache, (size, window)),
    ]


@app.post("/api/simulate")
def simulate() -> Any:
    body = request.get_json() or {}
    s = scenario(body["name"])
    settings = _settings(body)
    medians = PROFILES.get(s.cell.fn.__module__, {})
    raw = [sim.run(s, v, settings, medians) for v in _variants(s, body)]
    results = [{**r.summarize(), **r.stats} for r in raw]
    return jsonify({"results": results, "histogram": sim.histogram(results, raw), "settings": settings.__dict__})


@app.post("/api/curve")
def load_curve() -> Any:
    """Latency and throughput as offered load grows (open-loop, Poisson arrivals)."""
    body = request.get_json() or {}
    s = scenario(body["name"])
    settings = _settings(body, requests=200)
    medians = PROFILES.get(s.cell.fn.__module__, {})
    top = max(1.0, float(body.get("rate_max", 1000.0)))
    points = max(2, min(int(body.get("points", 8)), 16))
    rates = [round(top / 32 * 32 ** (i / (points - 1)), 1) for i in range(points)]
    variants = _variants(s, body)
    chosen = [variants[0], variants[2], variants[3]]  # eager, rewritten, batched
    return jsonify({"rates": rates, "curves": sim.curve(s, chosen, settings, medians, rates), "settings": settings.__dict__})


def main() -> None:
    port = int(os.environ.get("PORT", "5050"))
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=port, debug=bool(os.environ.get("DEBUG")))
