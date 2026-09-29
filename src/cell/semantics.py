"""Declared semantics of cells (DESIGN §1.1, §4.4).

v0 distinguishes only pure from effectful. The resource names given to
`effects(...)` are recorded now so the richer vocabulary (reads/writes,
idempotence, compensation) can attach to the same declaration later.
"""

from __future__ import annotations

from dataclasses import dataclass

MAIN = "main"
"""The default effect domain."""


class _Unique:
    """The domain that makes every call its own singleton domain."""

    _instance: _Unique | None = None

    def __new__(cls) -> _Unique:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNIQUE"


UNIQUE = _Unique()

type Domain = str | _Unique


@dataclass(frozen=True)
class Semantics:
    pure: bool
    resources: frozenset[str] | None  # None: unknown effects (external)

    def __repr__(self) -> str:
        if self.pure:
            return "pure"
        if self.resources is None:
            return "external"
        return "effects(" + ", ".join(repr(r) for r in sorted(self.resources)) + ")"


pure = Semantics(pure=True, resources=frozenset())
"""No effects. Calls may be reordered, repeated, speculated or cached."""

external = Semantics(pure=False, resources=None)
"""Unknown effects: the conservative default for undeclared cells."""


def effects(*resources: str) -> Semantics:
    """Effects on the named resources."""
    if not resources:
        raise ValueError("effects() needs at least one resource name; use `external` for unknown effects")
    return Semantics(pure=False, resources=frozenset(resources))


def check_domain(domain: object) -> Domain:
    if domain is UNIQUE:
        return UNIQUE
    if isinstance(domain, str) and domain:
        return domain
    raise ValueError(f"an effect domain is a non-empty str or UNIQUE, not {domain!r}")
