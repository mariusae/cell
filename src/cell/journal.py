"""Journals: what a cell invocation did, keyed by call path (DESIGN §3).

Every call and every ctx source a cell body issues gets an entry, keyed by
its `seq` (issue order within the invocation), its target and a digest of
its arguments. A call's entry holds the callee's own journal, so a run's
journal is a tree. Replay uses the entries to return recorded outcomes
instead of executing again; later, deopt and durability use the same
mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NoReturn

from . import data
from .semantics import Domain

if TYPE_CHECKING:
    from .context import Handle


@dataclass(frozen=True)
class Ok:
    value: Any

    def unwrap(self) -> Any:
        return self.value

    def __repr__(self) -> str:
        return f"Ok({_short(self.value)})"


@dataclass(frozen=True)
class Err:
    error: BaseException

    def unwrap(self) -> NoReturn:
        raise self.error

    def __repr__(self) -> str:
        return f"Err({type(self.error).__name__}: {self.error})"


type Outcome = Ok | Err


def outcome_digest(outcome: Outcome | None) -> str | None:
    """Compare outcomes by value: data digests for results, type and args for errors."""
    if outcome is None:
        return None
    if isinstance(outcome, Ok):
        return "ok:" + data.digest(outcome.value)
    e = outcome.error
    return f"err:{type(e).__module__}.{type(e).__qualname__}:{e.args!r}"


@dataclass(eq=False)
class Entry:
    """One call or ctx source issued by a cell body."""

    seq: int
    kind: str  # "call" or "source"
    target: str  # cell id, or source name ("now", "random", "config")
    effectful: bool = False
    domain: Domain | None = None  # effect domain of an effectful call
    args: dict[str, Any] | None = None  # None until handle arguments resolve
    args_digest: str | None = None
    started: bool = False  # False if a handle argument failed (the callee never ran)
    outcome: Outcome | None = None  # None while in flight
    awaited: bool = False
    replayed: bool = False  # the outcome came from a journal, not execution
    cached: bool = False  # the outcome came from the runtime's cache (no child journal)
    child: Journal | None = None  # the callee's journal
    handle: Handle[Any] | None = field(default=None, repr=False)

    @property
    def key(self) -> tuple[int, str, str, str | None]:
        return (self.seq, self.kind, self.target, self.args_digest)

    def summary(self) -> tuple[Any, ...]:
        return (
            self.seq,
            self.kind,
            self.target,
            self.args_digest,
            self.started,
            outcome_digest(self.outcome),
            None if self.child is None else self.child.summary(),
        )


@dataclass(eq=False)
class Journal:
    """What one cell invocation did."""

    request_id: str
    path: tuple[int, ...]
    cell: str
    args: dict[str, Any]
    entries: list[Entry] = field(default_factory=list)
    outcome: Outcome | None = None  # None if the invocation was interrupted
    mode: str = "eager"  # "eager", "compiled", or "deopt" (compiled, then replayed eagerly)
    deopt: str | None = None  # why a compiled run deopted
    # Seqs of effectful calls in the journal this invocation replayed from
    # that it never reached: effects in other domains that ran ahead of a
    # failure (DESIGN §4.4).
    unconsumed: list[int] = field(default_factory=list)

    def merged_over(self, base: Journal | None) -> Journal:
        """A journal with this one's entries, and base's for seqs this one lacks.

        After a deopt, replay needs both what the compiled attempt issued and,
        when that attempt was itself replaying, the journal it was replaying.
        """
        entries = {e.seq: e for e in base.entries} if base is not None else {}
        entries.update((e.seq, e) for e in self.entries)
        return Journal(self.request_id, self.path, self.cell, self.args, [entries[s] for s in sorted(entries)])

    def entry(self, seq: int) -> Entry | None:
        for e in self.entries:
            if e.seq == seq:
                return e
        return None

    def walk(self) -> list[Journal]:
        """This journal and all nested ones, in path order."""
        out = [self]
        for e in self.entries:
            if e.child is not None:
                out.extend(e.child.walk())
        return out

    def effects(self) -> tuple[Any, ...]:
        """A comparable form of what matters to the outside: the outcome, and
        every effectful call (recursively) with its arguments and outcome.

        Optimized runs may issue fewer pure calls (deduplicated, cached) or
        more (speculated) than eager execution, but must match it here.
        """
        return (
            self.cell,
            self.path,
            outcome_digest(self.outcome),
            tuple(
                (
                    e.seq,
                    e.target,
                    e.args_digest,
                    e.started,
                    outcome_digest(e.outcome),
                    None if e.child is None else e.child.effects(),
                )
                for e in self.entries
                if e.kind == "call" and e.effectful
            ),
        )

    def summary(self) -> tuple[Any, ...]:
        """A comparable form: equal for runs that did the same things with the same results."""
        return (
            self.cell,
            self.path,
            data.digest(self.args),
            outcome_digest(self.outcome),
            tuple(e.summary() for e in self.entries),
        )

    def format(self, indent: str = "") -> str:
        """A readable tree of the invocation, for debugging (and later EXPLAIN ANALYZE)."""
        args = ", ".join(f"{k}={_short(v, 30)}" for k, v in self.args.items())
        outcome = "interrupted" if self.outcome is None else repr(self.outcome)
        mode = {"eager": "", "compiled": "  [compiled]", "deopt": f"  [deopt: {self.deopt}]"}[self.mode]
        lines = [f"{indent}{_name(self.cell)}({args}) -> {outcome}{mode}"]
        for e in self.entries:
            lines.append(_format_entry(e, indent + "  "))
            if e.child is not None and not e.replayed:
                lines.extend(e.child.format(indent + "    ").splitlines()[1:])
        return "\n".join(lines)


def _format_entry(e: Entry, indent: str) -> str:
    args = "…" if e.args is None else ", ".join(f"{k}={_short(v, 30)}" for k, v in e.args.items())
    tags = []
    if e.effectful:
        tags.append(f"effectful[{e.domain}]")
    if e.replayed:
        tags.append("replayed")
    if e.cached:
        tags.append("cached")
    if e.kind == "call" and not e.awaited:
        tags.append("unawaited")
    if e.kind == "call" and not e.started and e.outcome is not None:
        tags.append("not started")
    tag = f"  ({', '.join(tags)})" if tags else ""
    outcome = "in flight" if e.outcome is None else repr(e.outcome)
    name = _name(e.target) if e.kind == "call" else f"ctx.{e.target}"
    return f"{indent}#{e.seq} {name}({args}) -> {outcome}{tag}"


def _name(cell_id: str) -> str:
    return cell_id.rsplit(".", 1)[-1]


def _short(value: Any, limit: int = 70) -> str:
    r = repr(value)
    return r if len(r) <= limit else r[: limit - 1] + "…"
