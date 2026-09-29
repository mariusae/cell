"""Differential validation of graphs against recorded requests (DESIGN §9).

A recorded request is an eager run's journal: its inputs, every call and
source it issued, and their outcomes. Validation re-runs each request in
compiled mode against its journal, so calls are answered from the record
rather than executed, and checks that the result and journal match.

A graph whose guards fail on a request still passes, as long as the deopt
reproduces the request exactly. What validation catches is a graph that
succeeds with a different result or different calls: a tracer hole that
froze a value into the graph (DESIGN §5.4, defense 5).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from .graph import Graph
from .journal import Journal, outcome_digest
from .runtime import Run, Runtime


@dataclass(frozen=True)
class Mismatch:
    request_id: str
    reason: str
    expected: Journal
    actual: Journal


@dataclass
class Validation:
    compiled: int = 0  # requests that ran compiled to completion
    deopted: int = 0  # requests that deopted (and still matched, unless listed below)
    mismatches: list[Mismatch] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.mismatches


async def validate(graph: Graph, recorded: Iterable[Run | Journal], *, runtime: Runtime | None = None) -> Validation:
    """Re-run recorded requests from the graph and compare with the recordings.

    `runtime` supplies configuration; by default a bare one. It gets no
    resources: calls must be answered from the journals.
    """
    rt = Runtime(config=runtime.config, seed=runtime.seed) if runtime is not None else Runtime()
    plan = rt.install(graph)
    result = Validation()
    for record in recorded:
        journal = record.journal if isinstance(record, Run) else record
        if journal.cell != graph.cell:
            raise ValueError(f"a journal of {journal.cell} can't validate a graph of {graph.cell}")
        run = await rt.replay(plan.cell, journal)
        if run.journal.mode == "compiled":
            result.compiled += 1
        else:
            result.deopted += 1
        if outcome_digest(run.outcome) != outcome_digest(journal.outcome):
            result.mismatches.append(Mismatch(journal.request_id, "outcome", journal, run.journal))
        elif run.journal.summary() != journal.summary():
            result.mismatches.append(Mismatch(journal.request_id, "calls", journal, run.journal))
    return result


async def promote(runtime: Runtime, graph: Graph, recorded: Iterable[Run | Journal]) -> Validation:
    """Install the graph in the runtime if it validates against the recordings."""
    result = await validate(graph, recorded, runtime=runtime)
    if result.passed:
        runtime.install(graph)
    return result
