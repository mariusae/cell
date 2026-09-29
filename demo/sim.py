"""Load simulation in simulated time.

Every leaf call is a request to a simulated service:

- its latency is drawn from a log-normal distribution with a given median
  and spread (sigma), so there is a tail;
- each service handles a limited number of calls at once (its capacity),
  and the rest queue. Load therefore costs latency, and throughput depends
  on how many calls each request makes, not only on how they overlap.

Clients run closed-loop: each issues its next request when the previous
one completes. Time is simulated (examples/simtime.py), so a run of
thousands of requests takes a fraction of a second, and results are
repeatable for a given seed.
"""

from __future__ import annotations

import asyncio
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from cell import Runtime
from cell.graph import Graph
from examples import simtime
from examples.harness import BATCH_COST, NOW, Scenario, World


@dataclass
class Settings:
    requests: int = 400
    clients: int = 32
    capacity: int = 8  # concurrent calls per service
    median_ms: float = 10.0
    sigma: float = 0.5  # spread of the log-normal: 0 is constant latency
    seed: int = 1


class Services:
    """Simulated services: sampled latency, and a concurrency limit each."""

    def __init__(self, settings: Settings, medians: dict[str, float]):
        self.settings = settings
        self.medians = medians
        self.rng = random.Random(settings.seed)
        self.calls: Counter[str] = Counter()
        self.elements: Counter[str] = Counter()
        self._slots: dict[str, asyncio.Semaphore] = {}

    async def serve(self, name: str, n: int = 1, key: str = "") -> None:
        """One call, serving n elements if it's a vector call. It takes one slot,
        and costs a little more per element (examples.harness.BATCH_COST).

        The latency is drawn from a generator seeded by `key`, which names the
        call (its request, path and service). The same call then takes the same
        time in every variant, whatever order calls happen in: comparisons
        between variants aren't noise from drawing different samples.
        """
        self.calls[name] += 1
        self.elements[name] += n
        median = self.medians.get(name, self.settings.median_ms / 1000)
        rng = random.Random(f"{self.settings.seed}|{key or name}|{self.calls[name] if not key else ''}")
        latency = median * math.exp(rng.gauss(0.0, self.settings.sigma)) * (1 + BATCH_COST * (n - 1))
        slots = self._slots.setdefault(name, asyncio.Semaphore(self.settings.capacity))
        async with slots:
            await asyncio.sleep(latency)


class SimWorld:
    """Stands in for the World resource: each request gets its own copy of
    the scenario's tables, and every call goes through the services."""

    def __init__(self, scenario: Scenario, services: Services):
        self.scenario = scenario
        self.services = services
        self.views: dict[str, World] = {}

    async def enter(self, ctx: Any, name: str, n: int = 1) -> World:
        await self.services.serve(name, n, key=f"{ctx.request_id}|{ctx.path}|{name}")
        # Vector calls serve many requests: they read a shared copy.
        key = "vector" if ctx.request_id.startswith("vector-") else ctx.request_id
        view = self.views.get(key)
        if view is None:
            view = self.views[key] = World(self.scenario.tables())
        return view


@dataclass
class Result:
    label: str
    latencies_ms: list[float]
    elapsed_s: float
    modes: Counter[str]
    errors: int
    calls: Counter[str]
    stats: dict[str, float] = field(default_factory=dict)

    def summarize(self) -> dict[str, Any]:
        xs = sorted(self.latencies_ms)
        n = len(xs)
        pct = lambda p: xs[min(n - 1, max(0, math.ceil(p / 100 * n) - 1))]  # noqa: E731
        return {
            "label": self.label,
            "requests": n,
            "mean": sum(xs) / n,
            "p50": pct(50),
            "p90": pct(90),
            "p99": pct(99),
            "max": xs[-1],
            "throughput": n / self.elapsed_s if self.elapsed_s else 0.0,
            "calls_per_request": sum(self.calls.values()) / n,
            "calls": dict(self.calls),
            "modes": dict(self.modes),
            "errors": self.errors,
            "cdf": [pct(p) for p in range(1, 101)],
        }


@dataclass(frozen=True)
class Variant:
    """How requests run: eagerly, or from a graph, with optional caching and batching."""

    label: str
    graph: Graph | None = None
    cache: tuple[Any, ...] = ()
    batch: tuple[int, float] | None = None  # max batch size, window in seconds


def run(
    s: Scenario,
    variant: Variant,
    settings: Settings,
    medians: dict[str, float],
    rate: float | None = None,
) -> Result:
    """Run settings.requests requests: closed-loop from settings.clients
    clients, or, given a rate, open-loop with Poisson arrivals at that rate
    (requests per second). Latency is measured from arrival to completion."""

    async def go() -> Result:
        loop = asyncio.get_running_loop()
        services = Services(settings, medians)
        world = SimWorld(s, services)
        rt = Runtime(
            config=s.config,
            resources={World: world},
            clock=lambda: NOW + loop.time(),
            cache={c: 60.0 for c in variant.cache},
        )
        if variant.graph is not None:
            rt.install(variant.graph)
            if variant.batch is not None:
                rt.batch(s.cell, max_size=variant.batch[0], window=variant.batch[1])
        latencies: list[float] = []
        modes: Counter[str] = Counter()
        errors = 0

        async def request(i: int) -> None:
            nonlocal errors
            start = loop.time()
            r = await rt.execute(s.cell, s.args, s.kwargs, request_id=f"r{i}")
            latencies.append((loop.time() - start) * 1000)
            modes[r.journal.mode] += 1
            errors += r.error is not None
            world.views.pop(f"r{i}", None)

        start = loop.time()
        if rate is None:
            ids = iter(range(settings.requests))

            async def client() -> None:
                for i in ids:
                    await request(i)

            await asyncio.gather(*(client() for _ in range(settings.clients)))
        else:
            arrivals = random.Random(settings.seed + 1)
            tasks = []
            for i in range(settings.requests):
                tasks.append(asyncio.create_task(request(i)))
                await asyncio.sleep(arrivals.expovariate(rate))
            await asyncio.gather(*tasks)
        result = Result(variant.label, latencies, loop.time() - start, modes, errors, services.calls)
        result.stats = {"vector_calls": rt.vector_stats["calls"], "batches": rt.batch_stats["batches"]}
        return result

    result, _ = simtime.run(go())
    return result


def curve(
    s: Scenario, variants: list[Variant], settings: Settings, medians: dict[str, float], rates: list[float]
) -> list[dict[str, Any]]:
    """Latency and achieved throughput as offered load grows, per variant.

    Once a variant saturates (its p50 passes 20 times its unloaded p50),
    higher rates only make its queues longer, so they are skipped.
    """
    out = []
    for v in variants:
        points: list[dict[str, Any]] = []
        base = None
        for rate in rates:
            summary = run(s, v, settings, medians, rate=rate).summarize()
            base = base or summary["p50"]
            points.append({"rate": rate, **{k: summary[k] for k in ("p50", "p99", "max", "throughput")}})
            if summary["p50"] > 20 * base:
                break
        out.append({"label": v.label, "points": points})
    return out


def histogram(results: list[dict[str, Any]], raw: list[Result], bins: int = 30) -> dict[str, Any]:
    """Shared bins over all results, so the distributions can be overlaid."""
    top = max(max(r.latencies_ms) for r in raw) or 1.0
    width = top / bins
    edges = [round(i * width, 2) for i in range(bins + 1)]
    counts = []
    for r in raw:
        c = [0] * bins
        for x in r.latencies_ms:
            c[min(bins - 1, int(x / width))] += 1
        counts.append(c)
    return {"edges": edges, "counts": counts}
