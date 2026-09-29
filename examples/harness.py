"""Fake services and scenarios for the example programs.

A `World` stands in for the services leaf cells talk to: named tables,
simulated latency, and a log of every leaf call and every effect, each
tagged with its call path. Because paths are deterministic, the log says
exactly which call did what, so tests can check that effects happened
exactly once across runs, replays and interruptions.

A `Scenario` is one program run: a cell, its arguments, an initial world,
and the expected result, error and effects. The same scenarios are meant
to be checked by every milestone: eager now, compiled later.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from cell import Cell, Ctx, Run, Runtime

NOW = 1_000_000.0
"""The fixed clock scenarios run with."""


class NotFound(Exception):
    pass


@dataclass(frozen=True)
class Call:
    path: tuple[int, ...]
    name: str


@dataclass(frozen=True)
class Effect:
    path: tuple[int, ...]
    name: str
    args: Mapping[str, Any]

    def key(self) -> tuple[tuple[int, ...], str, str]:
        return (self.path, self.name, repr(sorted(self.args.items())))


class World:
    """Fake services: tables, latency, and a log of calls and effects."""

    def __init__(self, tables: Mapping[str, Mapping[Any, Any]] | None = None, latency: Mapping[str, float] | None = None):
        self.tables: dict[str, dict[Any, Any]] = {name: dict(rows) for name, rows in (tables or {}).items()}
        self.latency = dict(latency or {})
        self.calls: list[Call] = []
        self.effects: list[Effect] = []

    def table(self, name: str) -> dict[Any, Any]:
        return self.tables.setdefault(name, {})

    async def enter(self, ctx: Ctx, name: str) -> World:
        """Record a leaf call and simulate its latency."""
        self.calls.append(Call(ctx.path, name))
        delay = self.latency.get(name, 0.0)
        if delay:
            await asyncio.sleep(delay)
        return self

    def effect(self, ctx: Ctx, name: str, **args: Any) -> None:
        self.effects.append(Effect(ctx.path, name, args))

    def effects_in_path_order(self) -> list[tuple[str, dict[str, Any]]]:
        return [(e.name, dict(e.args)) for e in sorted(self.effects, key=lambda e: e.path)]


async def service(ctx: Ctx, name: str) -> World:
    """Called by leaf cells: the world, after recording the call."""
    world: World = ctx.resource(World)
    return await world.enter(ctx, name)


def path_id(ctx: Ctx) -> str:
    """A deterministic id for this invocation, usable as an idempotency key."""
    return ".".join(str(p) for p in ctx.path) or "root"


@dataclass(frozen=True)
class Scenario:
    name: str
    cell: Cell
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    tables: Callable[[], Mapping[str, Mapping[Any, Any]]] = dict
    config: Mapping[str, Any] = field(default_factory=dict)
    expect: Any = None
    raises: type[BaseException] | None = None
    effects: tuple[tuple[str, Mapping[str, Any]], ...] = ()

    def world(self) -> World:
        return World(self.tables())

    def runtime(self, world: World) -> Runtime:
        return Runtime(config=self.config, resources={World: world}, clock=lambda: NOW, seed=0)

    async def run(self, world: World | None = None, **options: Any) -> tuple[Run, World]:
        world = world or self.world()
        run = await self.runtime(world).execute(self.cell, self.args, self.kwargs, request_id="req", **options)
        return run, world

    def __repr__(self) -> str:
        return f"<scenario {self.name}>"
