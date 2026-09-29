"""Cells and ops: the units users write (DESIGN §1.1, §1.3)."""

from __future__ import annotations

import functools
import hashlib
import inspect
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, overload

from .errors import ContextError
from .semantics import MAIN, Domain, Semantics, check_domain, external

if TYPE_CHECKING:
    from .context import Handle

_ALLOWED_KINDS = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)

# Every cell and op, by id, so a graph's references can be resolved. More
# than one can share an id (a function redefined in a test, say); the code
# hash tells them apart.
_CELLS: dict[str, weakref.WeakSet[Cell]] = {}
_OPS: dict[str, weakref.WeakSet[Op]] = {}


def find_cell(id: str, code: str) -> Cell | None:
    """The cell with this id and code hash, if one exists in this process."""
    return next((c for c in _CELLS.get(id, ()) if c.code_hash == code), None)


def find_op(id: str, code: str) -> Op | None:
    """The op with this id and code hash, if one exists in this process."""
    return next((o for o in _OPS.get(id, ()) if o.code_hash == code), None)


class Cell:
    """An async function whose calls go through a ctx.

    Calling a cell issues the call immediately and returns a Handle
    (DESIGN §1.4, rule 2). The cell object is its own typed reference.
    """

    def __init__(self, fn: Callable[..., Any], semantics: Semantics, domain: Domain | None):
        if not inspect.iscoroutinefunction(fn):
            raise TypeError(f"cell {fn.__qualname__} must be an async function")
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        if not params or params[0].kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise TypeError(f"cell {fn.__qualname__} must take ctx as its first parameter")
        for p in params[1:]:
            if p.kind not in _ALLOWED_KINDS:
                raise TypeError(
                    f"cell {fn.__qualname__}: parameter {p.name!r} must be a regular or "
                    "keyword-only parameter (no *args, **kwargs or positional-only)"
                )
        self.fn = fn
        self.semantics = semantics
        self.domain = domain
        self.vector: Cell | None = None  # a batched implementation (DESIGN §10)
        self.scalar: Cell | None = None  # for a vector form: the cell it batches
        self.id = f"{fn.__module__}.{fn.__qualname__}"
        self._sig = sig
        self._ctx_param = params[0].name
        functools.update_wrapper(self, fn)
        _CELLS.setdefault(self.id, weakref.WeakSet()).add(self)

    @property
    def effectful(self) -> bool:
        return not self.semantics.pure

    @property
    def params(self) -> list[str]:
        """Parameter names, excluding ctx."""
        return [p for p in self._sig.parameters if p != self._ctx_param]

    def vectorized(self, fn: Callable[..., Any]) -> Cell:
        """Declare a vector form: the same parameters, each a list, returning a list.

            @features.vectorized
            async def features_batch(ctx, uid: list[int], post: list[Post]) -> list[Features]: ...

        It must compute, element by element, what the cell computes:
        `vector(xs)[i] == cell(xs[i])`. Only pure cells may have one, since
        a vector call replaces many calls with one.
        """
        if self.effectful:
            raise ValueError(f"{self.id} is {self.semantics!r}; only pure cells can have vector forms")
        vector = Cell(fn, self.semantics, None)
        if vector.params != self.params:
            raise TypeError(f"vector form {vector.id} must take the parameters of {self.id}: {self.params}")
        self.vector = vector
        vector.scalar = self
        return vector

    @functools.cached_property
    def returns_none(self) -> bool:
        """Annotated `-> None`. The runtime enforces it, and the tracer relies on it."""
        return self._sig.return_annotation in (None, "None", type(None))

    @functools.cached_property
    def code_hash(self) -> str:
        return _code_hash(self.fn)

    def bind(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        """Bind call arguments to parameter names, with defaults applied."""
        try:
            bound = self._sig.bind(None, *args, **kwargs)
        except TypeError as e:
            raise TypeError(f"{self.id}: {e}") from None
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        del arguments[self._ctx_param]
        return arguments

    def __call__(self, ctx: Any, *args: Any, **kwargs: Any) -> Handle[Any]:
        from .context import Ctx

        if not isinstance(ctx, Ctx):
            raise ContextError(f"{self.id} must be called with a ctx as its first argument")
        return ctx._call(self, self.bind(args, kwargs))

    def __repr__(self) -> str:
        return f"<cell {self.id} {self.semantics!r}>"


@overload
def cell(fn: Callable[..., Any], /) -> Cell: ...
@overload
def cell(semantics: Semantics | None = None, /, *, domain: Domain | None = None) -> Callable[[Callable[..., Any]], Cell]: ...


def cell(arg: Any = None, /, *, domain: Domain | None = None) -> Any:
    """Declare a cell.

        @cell                                  # external: unknown effects
        @cell(pure)
        @cell(effects("prefs"))
        @cell(effects("audit"), domain="audit")
    """

    def make(fn: Callable[..., Any], semantics: Semantics) -> Cell:
        if semantics.pure:
            if domain is not None:
                raise ValueError(f"pure cell {fn.__qualname__} cannot have an effect domain")
            return Cell(fn, semantics, None)
        return Cell(fn, semantics, MAIN if domain is None else check_domain(domain))

    if callable(arg):
        return make(arg, external)
    if arg is not None and not isinstance(arg, Semantics):
        raise TypeError(f"@cell takes semantics such as pure or effects(...), not {arg!r}")
    semantics = external if arg is None else arg
    return lambda fn: make(fn, semantics)


class Op:
    """A pure, synchronous, deterministic local function (DESIGN §1.3).

    In eager mode an op is just a function call. The tracer records it as
    one node and runs it on concrete values.
    """

    def __init__(self, fn: Callable[..., Any]):
        if inspect.iscoroutinefunction(fn):
            raise TypeError(f"op {fn.__qualname__} must be synchronous")
        self.fn = fn
        self.id = f"{fn.__module__}.{fn.__qualname__}"
        functools.update_wrapper(self, fn)
        _OPS.setdefault(self.id, weakref.WeakSet()).add(self)

    @functools.cached_property
    def code_hash(self) -> str:
        return _code_hash(self.fn)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        from .trace import recorder_of

        rec = recorder_of((args, kwargs))
        if rec is None:
            return self.fn(*args, **kwargs)
        return rec.op(self, args, kwargs)

    def __repr__(self) -> str:
        return f"<op {self.id}>"


def op(fn: Callable[..., Any]) -> Op:
    """Declare an op."""
    return Op(fn)


def _code_hash(fn: Callable[..., Any]) -> str:
    try:
        source = inspect.getsource(fn).encode()
    except (OSError, TypeError):
        source = fn.__code__.co_code
    return hashlib.sha256(source).hexdigest()[:16]
