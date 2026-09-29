"""ctx.map: a fan-out over a list, as one call (DESIGN §10, milestone M5).

`await ctx.map(features, posts, uid=uid)` calls `features` once per post,
binding each item to the first parameter not given as a keyword, and
returns the results in order.

It is issued as a single call to one of the cells below, so it takes one
`seq`, however many items there are: the calls after it keep their places,
and the tracer records one node instead of unrolling a loop with a guard on
its length. The per-item calls are journaled under it.

If the cell has a vector form, the items go out as one vector call. If
that fails, they go out one by one, so errors are attributed to items.
"""

from __future__ import annotations

from typing import Any

from .core import Cell, cell, find_cell
from .errors import CellError
from .semantics import pure


async def _map(ctx: Any, target_id: str, code: str, param: str, items: list[Any], fixed: dict[str, Any]) -> list[Any]:
    target = find_cell(target_id, code)
    if target is None:
        raise CellError(f"map: no loaded cell {target_id} with code {code}")
    if not items:
        return []
    if target.vector is not None:
        n = len(items)
        columns = {p: list(items) if p == param else [fixed[p]] * n for p in target.params if p == param or p in fixed}
        try:
            results = await target.vector(ctx, **columns)
            if isinstance(results, (list, tuple)) and len(results) == n:
                return list(results)
        except Exception:
            pass  # attribute errors to items: call them one by one
    handles = [target(ctx, **{param: item}, **fixed) for item in items]
    return [await h for h in handles]


@cell(pure)
async def map_pure(ctx, cell: str, code: str, param: str, items: list, fixed: dict) -> list:  # type: ignore[type-arg]
    return await _map(ctx, cell, code, param, items, fixed)


@cell
async def map_effectful(ctx, cell: str, code: str, param: str, items: list, fixed: dict) -> list:  # type: ignore[type-arg]
    return await _map(ctx, cell, code, param, items, fixed)


MAP_CELLS = {map_pure.id, map_effectful.id}


def map_call(target: Cell, items: Any, fixed: dict[str, Any]) -> tuple[Cell, dict[str, Any]]:
    """The map cell and its arguments for ctx.map(target, items, **fixed)."""
    free = [p for p in target.params if p not in fixed]
    if not free:
        raise TypeError(f"map over {target.id}: every parameter is given; nothing to map over")
    arguments = {"cell": target.id, "code": target.code_hash, "param": free[0], "items": items, "fixed": dict(fixed)}
    return (map_pure if not target.effectful else map_effectful), arguments


def map_target(arguments: dict[str, Any]) -> Cell | None:
    return find_cell(arguments["cell"], arguments["code"])
