"""The example scenarios, checked against eager execution and replay.

For each scenario:

- eager: the result or error, and the effects, match the expectation;
- full replay: replaying the journal on a fresh world reproduces the
  outcome and journal without executing any leaf call;
- interrupt and replay: stopping the body before each seq and then
  replaying from the partial journal, on the same world, gives the same
  outcome with every leaf call and effect happening exactly once. This is
  the deopt path of M2 (DESIGN §6.3), exercised at every position.
"""

import asyncio
from collections import Counter

import pytest

from cell import Err, Ok, outcome_digest
from examples import SCENARIOS
from examples.harness import Scenario

ids = [s.name for s in SCENARIOS]


def run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize("s", SCENARIOS, ids=ids)
def test_eager(s: Scenario):
    r, world = run(s.run())
    if s.raises is not None:
        assert isinstance(r.outcome, Err), r.outcome
        assert isinstance(r.error, s.raises), r.error
    else:
        assert isinstance(r.outcome, Ok), r.outcome
        assert r.value == s.expect
    assert world.effects_in_path_order() == [(name, dict(args)) for name, args in s.effects]
    # Each leaf call and effect happens once, at a distinct path.
    assert len({(c.path, c.name) for c in world.calls}) == len(world.calls)


@pytest.mark.parametrize("s", SCENARIOS, ids=ids)
def test_full_replay(s: Scenario):
    first, _ = run(s.run())
    world = s.world()
    replayed = run(s.runtime(world).replay(s.cell, first.journal))
    assert outcome_digest(replayed.outcome) == outcome_digest(first.outcome)
    assert replayed.journal.summary() == first.journal.summary()
    assert world.calls == []
    assert world.effects == []
    assert replayed.unconsumed == ()
    assert all(e.replayed for e in replayed.journal.entries if e.started)


@pytest.mark.parametrize("s", SCENARIOS, ids=ids)
def test_interrupt_then_replay(s: Scenario):
    reference, ref_world = run(s.run())
    n = len(reference.journal.entries)
    for k in range(n + 1):
        world = s.world()
        partial, _ = run(s.run(world, interrupt_at=k))
        assert partial.interrupted == (k < n)
        assert [e.seq for e in partial.journal.entries] == list(range(min(k, n)))
        resumed = run(s.runtime(world).replay(s.cell, partial.journal))
        assert outcome_digest(resumed.outcome) == outcome_digest(reference.outcome), k
        assert resumed.journal.summary() == reference.journal.summary(), k
        assert Counter(world.calls) == Counter(ref_world.calls), k
        assert Counter(e.key() for e in world.effects) == Counter(e.key() for e in ref_world.effects), k
        assert resumed.unconsumed == ()


def test_scenario_names_are_unique():
    assert len(set(ids)) == len(ids)
