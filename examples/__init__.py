"""Example programs, and the scenarios every milestone is checked against."""

from . import checkout, features, feed, home, profile
from .harness import Scenario

SCENARIOS: list[Scenario] = [
    *home.SCENARIOS,
    *feed.SCENARIOS,
    *checkout.SCENARIOS,
    *profile.SCENARIOS,
    *features.SCENARIOS,
]


def scenario(name: str) -> Scenario:
    for s in SCENARIOS:
        if s.name == name:
            return s
    raise KeyError(name)
