"""A profile page built from composite helpers: what inlining is for.

`profile` calls two composite cells, `header` and `card`, which fetch some
of the same things the page does. Called as cells, each runs on its own:
`header` fetches the user, then the settings, then a greeting, one after
another in program order, and the page and its helpers fetch the user and
the settings more than once.

Inlining the helpers into the page's graph (DESIGN §5.7) lets the other
rewrites see across the boundary:

- dataflow: header's settings and user fetches run in parallel;
- dedup: the user and the settings are fetched once for the whole page;
- fold: `card` branches on its `style` argument, a constant at each call
  site. Where the page asks for the "full" card, the branch folds away.
  The "compact" card doesn't match the traced specialization at all, so
  that call site isn't inlined; it stays a call.
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
    best_friend: int


@dataclass(frozen=True)
class Settings:
    locale: str
    theme: str


@dataclass(frozen=True)
class Header:
    title: str
    theme: str


@dataclass(frozen=True)
class Card:
    name: str
    badge: str


@dataclass(frozen=True)
class Profile:
    header: Header
    me: Card
    friend: Card
    locale: str


@cell(pure)
async def get_user(ctx, uid: int) -> User:
    w = await service(ctx, "get_user")
    try:
        return w.table("users")[uid]
    except KeyError:
        raise NotFound(f"user {uid}") from None


@cell(pure)
async def get_settings(ctx, uid: int) -> Settings:
    w = await service(ctx, "get_settings")
    return w.table("settings").get(uid, Settings("en", "light"))


@cell(pure)
async def translate(ctx, key: str, locale: str) -> str:
    w = await service(ctx, "translate")
    return w.table("strings").get((key, locale), key)


@cell(effects("views"))
async def record_view(ctx, uid: int, locale: str) -> None:
    w = await service(ctx, "record_view")
    w.effect(ctx, "record_view", uid=uid, locale=locale)


@op
def greeting(word: str, user: User) -> str:
    return f"{word}, {user.name}"


@cell(pure)
async def header(ctx, uid: int) -> Header:
    user = await get_user(ctx, uid)
    settings = await get_settings(ctx, uid)  # doesn't need the user, but waits for it
    word = await translate(ctx, "welcome", settings.locale)
    return Header(greeting(word, user), settings.theme)


@cell(pure)
async def card(ctx, uid: int, style: str) -> Card:
    user = await get_user(ctx, uid)
    if style == "compact":
        return Card(user.name, "")
    return Card(user.name, user.tier)


@cell
async def profile(ctx, uid: int) -> Profile:
    head = header(ctx, uid)
    me = await card(ctx, uid, "full")
    user = await get_user(ctx, uid)
    friend = card(ctx, user.best_friend, "compact")
    settings = await get_settings(ctx, uid)
    await record_view(ctx, uid, settings.locale)
    return Profile(await head, me, await friend, settings.locale)


def tables():
    return {
        "users": {1: User(1, "Ada", "premium", 2), 2: User(2, "Bob", "basic", 1), 3: User(3, "Cy", "basic", 9)},
        "settings": {1: Settings("en", "dark"), 2: Settings("fr", "light")},
        "strings": {("welcome", "en"): "Welcome", ("welcome", "fr"): "Bienvenue"},
    }


SCENARIOS = [
    Scenario(
        "profile/ada",
        profile,
        (1,),
        tables=tables,
        expect=Profile(Header("Welcome, Ada", "dark"), Card("Ada", "premium"), Card("Bob", ""), "en"),
        effects=(("record_view", {"uid": 1, "locale": "en"}),),
    ),
    Scenario(
        "profile/bob",
        profile,
        (2,),
        tables=tables,
        expect=Profile(Header("Bienvenue, Bob", "light"), Card("Bob", "basic"), Card("Ada", ""), "fr"),
        effects=(("record_view", {"uid": 2, "locale": "fr"}),),
    ),
    Scenario(
        "profile/missing-friend",
        profile,
        (3,),
        tables=tables,
        raises=NotFound,
        effects=(("record_view", {"uid": 3, "locale": "en"}),),
    ),
]
