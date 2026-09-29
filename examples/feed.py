"""A feed: fan-out over authors and posts, per-post features, a write.

This is the shape later milestones vectorize: `features` is called once
per post, and many requests call it at once.
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


P201 = Post(201, 2, "old but liked", NOW - 3600, 10)  # 0.5 + 0.10 - 1.0   = -0.40
P202 = Post(202, 2, "recent", NOW - 600, 0)  # 0.5 + 0.00 - 0.167 =  0.33
P301 = Post(301, 3, "fresh and liked", NOW - 60, 50)  # 0.1 + 0.50 - 0.017 =  0.58


def tables():
    return {
        "follows": {1: (2, 3), 4: ()},
        "posts": {2: (P201, P202), 3: (P301,)},
        "affinity": {(1, 2): 0.5, (1, 3): 0.1},
    }


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
]
