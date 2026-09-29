"""A home page: fan-out, a branch, and parallelism hidden by program order.

This is the example of DESIGN §4.6. `get_items` doesn't depend on
`get_user`, but the code issues it after awaiting `get_user`; a compiled
graph can run them in parallel.
"""

from __future__ import annotations

from dataclasses import dataclass

from cell import cell, effects, op, pure

from .harness import NotFound, Scenario, service


@dataclass(frozen=True)
class User:
    id: int
    name: str
    tier: str


@dataclass(frozen=True)
class Prefs:
    boost: str  # a category to rank higher


@dataclass(frozen=True)
class Item:
    id: int
    title: str
    category: str
    score: float


@dataclass(frozen=True)
class Page:
    title: str
    items: tuple[Item, ...]


@cell(pure)
async def get_user(ctx, uid: int) -> User:
    w = await service(ctx, "get_user")
    try:
        return w.table("users")[uid]
    except KeyError:
        raise NotFound(f"user {uid}") from None


@cell(effects("prefs"))
async def get_prefs(ctx, uid: int) -> Prefs:
    w = await service(ctx, "get_prefs")
    return w.table("prefs").get(uid, Prefs(boost=""))


@cell(pure)
async def get_items(ctx, uid: int) -> tuple[Item, ...]:
    w = await service(ctx, "get_items")
    return w.table("items").get(uid, ())


@cell(pure)
async def rank(ctx, items: tuple[Item, ...], prefs: Prefs) -> tuple[Item, ...]:
    await service(ctx, "rank")
    boost = lambda i: 1.0 if i.category == prefs.boost else 0.0
    return tuple(sorted(items, key=lambda i: (-(i.score + boost(i)), i.id)))


@op
def greeting(user: User) -> str:
    return f"Hello, {user.name}"


@cell
async def home(ctx, uid: int) -> Page:
    user = await get_user(ctx, uid)
    prefs = get_prefs(ctx, user.id)  # issued, not yet awaited
    items = get_items(ctx, uid)  # issued, not yet awaited
    if user.tier == "premium":
        ranked = await rank(ctx, await items, await prefs)
    else:
        ranked = await items
    return Page(title=greeting(user), items=ranked)


@cell
async def home_pushdown(ctx, uid: int) -> Page:
    """`home`, passing the handles into `rank` instead of awaiting them (DESIGN §1.5)."""
    user = await get_user(ctx, uid)
    prefs = get_prefs(ctx, user.id)
    items = get_items(ctx, uid)
    if user.tier == "premium":
        ranked = await rank(ctx, items, prefs)
    else:
        ranked = await items
    return Page(title=greeting(user), items=ranked)


DUNE = Item(10, "Dune", "books", 0.5)
TEA = Item(11, "Tea", "food", 0.9)
SICP = Item(12, "SICP", "books", 0.3)
LAMP = Item(20, "Lamp", "home", 0.4)
PEN = Item(21, "Pen", "office", 0.2)


def tables():
    return {
        "users": {1: User(1, "Ada", "premium"), 2: User(2, "Bob", "basic")},
        "prefs": {1: Prefs(boost="books")},
        "items": {1: (DUNE, TEA, SICP), 2: (LAMP, PEN)},
    }


SCENARIOS = [
    Scenario(
        "home/premium",
        home,
        (1,),
        tables=tables,
        expect=Page("Hello, Ada", (DUNE, SICP, TEA)),
    ),
    Scenario("home/basic", home, (2,), tables=tables, expect=Page("Hello, Bob", (LAMP, PEN))),
    Scenario("home/unknown-user", home, (3,), tables=tables, raises=NotFound),
    Scenario(
        "home_pushdown/premium",
        home_pushdown,
        (1,),
        tables=tables,
        expect=Page("Hello, Ada", (DUNE, SICP, TEA)),
    ),
    Scenario("home_pushdown/basic", home_pushdown, (2,), tables=tables, expect=Page("Hello, Bob", (LAMP, PEN))),
]
