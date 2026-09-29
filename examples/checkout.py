"""A checkout: effects, their order, and failure handling.

`audit` is in its own effect domain and `notify` in a singleton domain
(DESIGN §4.4). The eager runtime runs everything in program order, which
is one allowed order; the domains matter once calls can be reordered.
Compensation is written by hand for now (NOTES §3 would derive it).
"""

from __future__ import annotations

from dataclasses import dataclass

from cell import UNIQUE, cell, effects

from .harness import Scenario, path_id, service


class OutOfStock(Exception):
    pass


class Declined(Exception):
    pass


@dataclass(frozen=True)
class Order:
    id: str
    sku: str
    qty: int
    card: str
    amount: int


@dataclass(frozen=True)
class Receipt:
    order_id: str
    charge_id: str
    amount: int


@cell(effects("audit"), domain="audit")
async def audit(ctx, event: str, ref: str) -> None:
    w = await service(ctx, "audit")
    w.effect(ctx, "audit", event=event, ref=ref)


@cell(effects("inventory"))
async def reserve(ctx, sku: str, qty: int) -> str:
    w = await service(ctx, "reserve")
    stock = w.table("inventory")
    if stock.get(sku, 0) < qty:
        raise OutOfStock(sku)
    stock[sku] -= qty
    reservation = f"rsv-{path_id(ctx)}"
    w.table("reservations")[reservation] = (sku, qty)
    w.effect(ctx, "reserve", sku=sku, qty=qty)
    return reservation


@cell(effects("inventory"))
async def release(ctx, reservation: str) -> None:
    w = await service(ctx, "release")
    sku, qty = w.table("reservations").pop(reservation)
    w.table("inventory")[sku] += qty
    w.effect(ctx, "release", reservation=reservation)


@cell(effects("payments"))
async def charge(ctx, card: str, amount: int) -> str:
    w = await service(ctx, "charge")
    balances = w.table("cards")
    if balances.get(card, 0) < amount:
        raise Declined(card)
    balances[card] -= amount
    w.effect(ctx, "charge", card=card, amount=amount)
    return f"ch-{path_id(ctx)}"


@cell(effects("email"), domain=UNIQUE)
async def notify(ctx, receipt: Receipt) -> None:
    w = await service(ctx, "notify")
    w.effect(ctx, "notify", order_id=receipt.order_id)


@cell
async def checkout(ctx, order: Order) -> Receipt:
    await audit(ctx, "attempt", order.id)
    reservation = await reserve(ctx, order.sku, order.qty)
    try:
        charge_id = await charge(ctx, order.card, order.amount)
    except Declined:
        await release(ctx, reservation)
        await audit(ctx, "declined", order.id)
        raise
    receipt = Receipt(order.id, charge_id, order.amount)
    await notify(ctx, receipt)
    await audit(ctx, "charged", charge_id)  # after the charge, by data
    return receipt


def tables():
    return {"inventory": {"book": 5, "lamp": 0}, "cards": {"good": 10_000, "poor": 5}}


SCENARIOS = [
    Scenario(
        "checkout/ok",
        checkout,
        (Order("o1", "book", 2, "good", 300),),
        tables=tables,
        expect=Receipt("o1", "ch-2", 300),
        effects=(
            ("audit", {"event": "attempt", "ref": "o1"}),
            ("reserve", {"sku": "book", "qty": 2}),
            ("charge", {"card": "good", "amount": 300}),
            ("notify", {"order_id": "o1"}),
            ("audit", {"event": "charged", "ref": "ch-2"}),
        ),
    ),
    Scenario(
        "checkout/out-of-stock",
        checkout,
        (Order("o2", "lamp", 1, "good", 100),),
        tables=tables,
        raises=OutOfStock,
        effects=(("audit", {"event": "attempt", "ref": "o2"}),),
    ),
    Scenario(
        "checkout/declined",
        checkout,
        (Order("o3", "book", 1, "poor", 300),),
        tables=tables,
        raises=Declined,
        effects=(
            ("audit", {"event": "attempt", "ref": "o3"}),
            ("reserve", {"sku": "book", "qty": 1}),
            ("release", {"reservation": "rsv-1"}),
            ("audit", {"event": "declined", "ref": "o3"}),
        ),
    ),
]
