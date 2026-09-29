"""A feed: fan-out over authors and posts, per-post features, a write.

Two versions of the same logic:

- `feed` loops over authors and posts in Python. The tracer unrolls the
  loops, with a guard on each length, so a graph fits only users who follow
  the same number of authors, with the same numbers of posts.
- `feed_mapped` fans out with ctx.map. Each fan-out is one node, whatever
  its size, so one graph fits every user; and since the leaves have vector
  forms, each fan-out is one call, and a batch of requests shares them.
"""

from __future__ import annotations

from dataclasses import dataclass

from cell import cell, effects, op, pure

from .harness import NOW, Scenario, service


@dataclass(frozen=True)
class Post:
    id: int
    author: int
    text: str
    ts: float
    likes: int


@dataclass(frozen=True)
class Features:
    post_id: int
    affinity: float


@dataclass(frozen=True)
class Feed:
    user: int
    posts: tuple[Post, ...]


@cell(pure)
async def following(ctx, uid: int) -> tuple[int, ...]:
    w = await service(ctx, "following")
    return w.table("follows").get(uid, ())


@cell(pure)
async def recent_posts(ctx, author: int) -> tuple[Post, ...]:
    w = await service(ctx, "recent_posts")
    return w.table("posts").get(author, ())


@cell(pure)
async def features(ctx, uid: int, post: Post) -> Features:
    w = await service(ctx, "features")
    return Features(post.id, w.table("affinity").get((uid, post.author), 0.0))


@following.vectorized
async def following_many(ctx, uid: list[int]) -> list[tuple[int, ...]]:
    w = await service(ctx, "following_many", len(uid))
    return [w.table("follows").get(u, ()) for u in uid]


@recent_posts.vectorized
async def recent_posts_many(ctx, author: list[int]) -> list[tuple[Post, ...]]:
    w = await service(ctx, "recent_posts_many", len(author))
    return [w.table("posts").get(a, ()) for a in author]


@features.vectorized
async def features_many(ctx, uid: list[int], post: list[Post]) -> list[Features]:
    w = await service(ctx, "features_many", len(post))
    return [Features(p.id, w.table("affinity").get((u, p.author), 0.0)) for u, p in zip(uid, post)]


@cell(effects("seen"))
async def mark_seen(ctx, uid: int, post_ids: tuple[int, ...]) -> None:
    w = await service(ctx, "mark_seen")
    w.table("seen").setdefault(uid, set()).update(post_ids)
    w.effect(ctx, "mark_seen", uid=uid, post_ids=post_ids)


@op
def score(post: Post, f: Features, now: float) -> float:
    return f.affinity + post.likes / 100 - (now - post.ts) / 3600


@op
def top_k(scored: tuple[tuple[Post, float], ...], k: int) -> tuple[Post, ...]:
    ranked = sorted(scored, key=lambda ps: (-ps[1], ps[0].id))
    return tuple(p for p, _ in ranked[:k])


@cell
async def feed(ctx, uid: int, k: int = 3) -> Feed:
    now = ctx.now()
    authors = await following(ctx, uid)
    post_lists = [recent_posts(ctx, a) for a in authors]  # fan-out: all issued at once
    posts = [p for h in post_lists for p in await h]
    feats = [features(ctx, uid, p) for p in posts]
    scored = [(p, score(p, await f, now)) for p, f in zip(posts, feats)]
    top = top_k(tuple(scored), k)
    if top:
        await mark_seen(ctx, uid, tuple(p.id for p in top))
    return Feed(uid, top)


@op
def concat(lists: list[tuple[Post, ...]]) -> tuple[Post, ...]:
    return tuple(p for ps in lists for p in ps)


@op
def score_all(posts: tuple[Post, ...], feats: list[Features], now: float) -> tuple[tuple[Post, float], ...]:
    return tuple((p, score(p, f, now)) for p, f in zip(posts, feats))


@op
def post_ids(posts: tuple[Post, ...]) -> tuple[int, ...]:
    return tuple(p.id for p in posts)


@cell
async def feed_mapped(ctx, uid: int, k: int = 3) -> Feed:
    """The feed, with ctx.map instead of loops: one graph for every user."""
    now = ctx.now()
    authors = await following(ctx, uid)
    post_lists = await ctx.map(recent_posts, authors)
    posts = concat(post_lists)
    feats = await ctx.map(features, posts, uid=uid)
    top = top_k(score_all(posts, feats, now), k)
    if top:
        await mark_seen(ctx, uid, post_ids(top))
    return Feed(uid, top)


P201 = Post(201, 2, "old but liked", NOW - 3600, 10)  # 0.5 + 0.10 - 1.0   = -0.40
P202 = Post(202, 2, "recent", NOW - 600, 0)  # 0.5 + 0.00 - 0.167 =  0.33
P301 = Post(301, 3, "fresh and liked", NOW - 60, 50)  # 0.1 + 0.50 - 0.017 =  0.58


# A busy user follows six authors with three posts each.
BUSY = 5
BUSY_AUTHORS = (2, 3, 6, 7, 8, 9)
EXTRA_POSTS = {
    a: tuple(Post(a * 100 + i, a, f"post {i} by {a}", NOW - 600 * (i + 1) * (a - 4), (a * 7 + i * 13) % 40) for i in range(3))
    for a in (6, 7, 8, 9)
}


def tables():
    return {
        "follows": {1: (2, 3), 4: (), BUSY: BUSY_AUTHORS},
        "posts": {2: (P201, P202), 3: (P301,), **EXTRA_POSTS},
        "affinity": {(1, 2): 0.5, (1, 3): 0.1, **{(BUSY, a): 0.1 * (a % 4) for a in BUSY_AUTHORS}},
    }


def expected(uid: int, k: int = 3) -> Feed:
    """The feed a user should see, computed directly from the tables."""
    t = tables()
    posts = tuple(p for a in t["follows"][uid] for p in t["posts"].get(a, ()))
    feats = [Features(p.id, t["affinity"].get((uid, p.author), 0.0)) for p in posts]
    return Feed(uid, top_k(score_all(posts, feats, NOW), k))


SCENARIOS = [
    Scenario(
        "feed/top2",
        feed,
        (1,),
        {"k": 2},
        tables=tables,
        expect=Feed(1, (P301, P202)),
        effects=(("mark_seen", {"uid": 1, "post_ids": (301, 202)}),),
    ),
    Scenario(
        "feed/all",
        feed,
        (1,),
        tables=tables,
        expect=Feed(1, (P301, P202, P201)),
        effects=(("mark_seen", {"uid": 1, "post_ids": (301, 202, 201)}),),
    ),
    Scenario("feed/follows-nobody", feed, (4,), tables=tables, expect=Feed(4, ())),
    Scenario(
        "feed/busy",
        feed,
        (BUSY,),
        tables=tables,
        expect=expected(BUSY),
        effects=(("mark_seen", {"uid": BUSY, "post_ids": post_ids(expected(BUSY).posts)}),),
    ),
    Scenario(
        "feed_mapped/top2",
        feed_mapped,
        (1,),
        {"k": 2},
        tables=tables,
        expect=Feed(1, (P301, P202)),
        effects=(("mark_seen", {"uid": 1, "post_ids": (301, 202)}),),
    ),
    Scenario(
        "feed_mapped/busy",
        feed_mapped,
        (BUSY,),
        tables=tables,
        expect=expected(BUSY),
        effects=(("mark_seen", {"uid": BUSY, "post_ids": post_ids(expected(BUSY).posts)}),),
    ),
    Scenario("feed_mapped/follows-nobody", feed_mapped, (4,), tables=tables, expect=Feed(4, ())),
]
