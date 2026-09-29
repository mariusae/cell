"""Tracer strictness with sys.monitoring (DESIGN §5.4, defense 3; milestone M3).

The tracer intercepts what Python does with a traced value through its
dunder methods. Code implemented in C can bypass them: `type(t)` and
`id(t)` never ask the value, and a C function that checks types directly
raises TypeError, which user code may catch and silently take another
path. While a trace is active, two sys.monitoring events close these:

- CALL: a C-level callable that isn't known to be safe, called with a
  traced value as its first argument, is a graph break. CALL only exposes
  the first argument (for a method call, `self`), so this is partial.
- EXCEPTION_HANDLED: a TypeError caught anywhere in the traced body's task
  is a graph break, since the tracer can't tell whether a traced value
  caused it. This covers what CALL can't see, such as `", ".join(names)`
  with traced names inside `try/except TypeError`.

Monitoring is process-wide while any trace is active, but callbacks act
only in the task running a traced body, and only on code outside this
package and asyncio. Calls from those are disabled after their first
event, so they cost nothing afterwards.
"""

from __future__ import annotations

import asyncio
import builtins
import operator
import os
import sys
import sysconfig
import types
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .trace import Recorder

_M = sys.monitoring
_TOOL_IDS = (4, 3)  # free by default: 0-2 and 5 are reserved for debuggers, coverage, profilers, optimizers
_HERE = os.path.dirname(os.path.abspath(__file__))
_ASYNCIO = os.path.dirname(os.path.abspath(asyncio.__file__))
_STDLIB = sysconfig.get_paths()["stdlib"]


def _safe_callables() -> set[int]:
    """C callables that reach traced values only through their dunders, so the tracer sees them."""
    names = (
        "abs all any bool bytes dict divmod enumerate filter float format frozenset getattr hasattr "
        "hash int isinstance iter len list map max min next pow print range repr reversed round set "
        "sorted str sum tuple zip"
    ).split()
    safe: list[Any] = [getattr(builtins, n) for n in names]
    safe += [getattr(operator, n) for n in dir(operator) if not n.startswith("_")]
    # Storing a reference: how frozen dataclasses' generated __init__ sets fields.
    safe += [object.__setattr__, object.__new__]
    # Methods of local containers: building a list of traced values is fine.
    for t, methods in {
        list: "append extend insert pop copy index count",
        dict: "get items keys values update setdefault pop copy",
        set: "add update discard",
        tuple: "index count",
    }.items():
        safe += [getattr(t, m) for m in methods.split()]
    return {id(c) for c in safe}


_SAFE = _safe_callables()


def _ours(code: types.CodeType) -> bool:
    filename = code.co_filename
    return filename.startswith(_HERE) or filename.startswith(_ASYNCIO)


def _library(filename: str) -> bool:
    return filename.startswith(_STDLIB) or filename.startswith("<")


def _from_tracer(frame: types.FrameType | None) -> bool:
    """True if the nearest caller outside the standard library is this package.

    The tracer itself calls library code with traced values (dataclasses,
    for one); those calls are not the body's.
    """
    while frame is not None and _library(frame.f_code.co_filename):
        frame = frame.f_back
    return frame is not None and frame.f_code.co_filename.startswith(_HERE)


def _python_level(fn: Any) -> bool:
    """True for callables written in Python, which the tracer traces through."""
    if isinstance(fn, types.FunctionType):
        return True
    if isinstance(fn, types.MethodType):
        return isinstance(fn.__func__, types.FunctionType)
    if isinstance(fn, type):
        for klass in fn.__mro__:
            if klass.__module__ == "builtins":
                continue
            for name in ("__new__", "__init__"):
                if isinstance(klass.__dict__.get(name), (types.FunctionType, staticmethod)):
                    return True
        return False
    return isinstance(getattr(type(fn), "__call__", None), types.FunctionType)


def _name(fn: Any) -> str:
    return getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or type(fn).__name__


class _Monitor:
    def __init__(self) -> None:
        self.active: dict[asyncio.Task[Any], Recorder] = {}
        self.tool: int | None = None
        self._busy = False

    def register(self, rec: Recorder) -> asyncio.Task[Any] | None:
        """Watch the current task, which runs rec's body. Returns it, to unregister."""
        task = asyncio.current_task()
        if task is None:
            return None
        if not self.active:
            self._start()
        self.active[task] = rec
        return task

    def unregister(self, task: asyncio.Task[Any] | None) -> None:
        if task is None or self.active.pop(task, None) is None:
            return
        if not self.active:
            self._stop()

    @property
    def available(self) -> bool:
        return self.tool is not None

    def _start(self) -> None:
        for tool in _TOOL_IDS:
            try:
                _M.use_tool_id(tool, "cell tracer")
            except ValueError:
                continue
            self.tool = tool
            _M.register_callback(tool, _M.events.CALL, self._on_call)
            _M.register_callback(tool, _M.events.EXCEPTION_HANDLED, self._on_handled)
            _M.set_events(tool, _M.events.CALL | _M.events.EXCEPTION_HANDLED)
            return

    def _stop(self) -> None:
        if self.tool is not None:
            _M.set_events(self.tool, 0)
            _M.register_callback(self.tool, _M.events.CALL, None)
            _M.register_callback(self.tool, _M.events.EXCEPTION_HANDLED, None)
            _M.free_tool_id(self.tool)
            self.tool = None

    def _recorder(self) -> Recorder | None:
        if self._busy:
            return None
        try:
            task = asyncio.current_task()
        except RuntimeError:
            return None
        rec = self.active.get(task) if task is not None else None
        if rec is None or not rec.active or rec.broken:
            return None
        return rec

    def _on_call(self, code: types.CodeType, offset: int, fn: Any, arg0: Any) -> Any:
        if _ours(code):
            return _M.DISABLE
        rec = self._recorder()
        if rec is None or arg0 is _M.MISSING:
            return None
        from .trace import Tracer, recorder_of

        if type(fn) is Tracer or id(fn) in _SAFE or _python_level(fn):
            return None
        self._busy = True
        try:
            traced = recorder_of(arg0) is rec
        finally:
            self._busy = False
        if traced and not _from_tracer(sys._getframe(1)):
            rec.break_(f"a traced value passed to {_name(fn)}, which the tracer can't see into; use an @op")
        return None

    def _on_handled(self, code: types.CodeType, offset: int, exc: BaseException) -> Any:
        if _ours(code) or not isinstance(exc, TypeError):
            return None
        rec = self._recorder()
        if rec is not None and not _from_tracer(sys._getframe(1)):
            rec.break_("a TypeError was caught while tracing; a traced value may have caused it")
        return None


MONITOR = _Monitor()
