"""Static extraction: call graphs and lint from cell source (DESIGN §9, milestone M3).

The extractor reads a cell's source, and the source of plain helper
functions it calls, and resolves names through the function's globals and
closure. It produces:

- the **call graph**: every cell the body could call, found through typed
  references (any reference, called or not, so it over-approximates every
  graph the tracer can record);
- **lint** for composite cells: violations of the rules cell bodies must
  follow (DESIGN §1.4), which are errors, and code the tracer will break
  on or specialize heavily, which are warnings.

Leaf cells (those that call no cells and use resources) implement I/O,
so the determinism rules don't apply to them, and they aren't traced;
they get no findings.

This is for safety checks, never for optimization (DESIGN §9).
"""

from __future__ import annotations

import ast
import builtins
import dataclasses
import enum
import inspect
import os
import sysconfig
import textwrap
import types
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from .core import Cell, Op

_STDLIB = sysconfig.get_paths()["stdlib"]
_HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True, order=True)
class Finding:
    site: str  # path:line
    rule: str
    severity: str  # "error": breaks a rule of cell bodies; "warning": the tracer will break or specialize
    message: str
    cell: str = ""

    def format(self, root: str | None = None) -> str:
        root = os.path.join(os.path.abspath(root or os.getcwd()), "")
        site = self.site[len(root):] if self.site.startswith(root) else self.site
        return f"{site}: {self.severity}: {self.message} [{self.rule}]"


@dataclass
class Extraction:
    cell: Cell
    calls: set[Cell] = field(default_factory=set)
    ops: set[Op] = field(default_factory=set)
    helpers: list[Callable[..., Any]] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    uses_resources: bool = False

    @property
    def leaf(self) -> bool:
        """Calls no cells and uses resources: it implements I/O, and isn't linted."""
        return not self.calls and self.uses_resources


def extract(cell: Cell) -> Extraction:
    """The cells and ops a cell's body can reach, and lint findings for it."""
    ex = Extraction(cell)
    pending: list[Finding] = []
    seen: set[types.CodeType] = set()
    _Function(cell.fn, cell.fn.__code__.co_varnames[0], ex, pending, seen, helper=False).analyze()
    if not ex.leaf:
        ex.findings = sorted(dataclasses.replace(f, cell=cell.id) for f in pending)
    return ex


def lint(cell: Cell) -> list[Finding]:
    return extract(cell).findings


def call_graph(roots: Iterable[Cell]) -> dict[Cell, set[Cell]]:
    """Each cell reachable from the roots, and the cells it may call."""
    graph: dict[Cell, set[Cell]] = {}
    todo = list(roots)
    while todo:
        c = todo.pop()
        if c in graph:
            continue
        graph[c] = extract(c).calls
        todo.extend(graph[c] - graph.keys())
    return graph


def cells_of(module: types.ModuleType) -> list[Cell]:
    return [v for v in vars(module).values() if isinstance(v, Cell) and v.fn.__module__ == module.__name__]


# Classification of what names resolve to


_NONDETERMINISTIC = {
    "time.time", "time.time_ns", "time.monotonic", "time.monotonic_ns", "time.perf_counter",
    "time.perf_counter_ns", "time.process_time", "time.process_time_ns", "datetime.datetime.now",
    "datetime.datetime.utcnow", "datetime.datetime.today", "datetime.date.today", "os.urandom",
    "os.getpid", "os.getenv", "uuid.uuid1", "uuid.uuid4",
}
_NONDETERMINISTIC_MODULES = ("random.", "secrets.")
_IO = {"io.open", "builtins.open", "builtins.input", "time.sleep", "os.system", "os.popen"}
_IO_MODULES = ("socket.", "subprocess.", "urllib.", "http.", "requests.", "httpx.", "sqlite3.", "shutil.")
_TASKS = {"asyncio.create_task", "asyncio.ensure_future", "asyncio.TaskGroup"}
_RACES = {"asyncio.wait", "asyncio.as_completed", "asyncio.wait_for", "asyncio.timeout"}
_SAFE = {
    "builtins.len", "builtins.tuple", "builtins.list", "builtins.dict", "builtins.sum", "builtins.zip",
    "builtins.enumerate", "builtins.any", "builtins.all", "builtins.abs", "builtins.iter", "builtins.next",
    "builtins.reversed", "builtins.map", "builtins.filter", "builtins.bool", "asyncio.gather",
}
_CONVERSIONS = {
    "builtins.str", "builtins.int", "builtins.float", "builtins.bytes", "builtins.repr", "builtins.hash",
    "builtins.format", "builtins.isinstance",
}
_INVISIBLE = {"builtins.type", "builtins.id", "builtins.callable"}
_ORDERING = {"builtins.sorted", "builtins.min", "builtins.max"}


def _qualname(obj: Any) -> str | None:
    if isinstance(obj, types.ModuleType):
        return obj.__name__
    module = getattr(obj, "__module__", None)
    name = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None)
    if not isinstance(module, str) or not isinstance(name, str):
        return None
    return f"{_module(module)}.{name}"


def _module(name: str) -> str:
    return {"posix": "os", "nt": "os"}.get(name, name.lstrip("_"))


def _names(e: ast.AST, obj: Any, resolve: Callable[[ast.AST], tuple[bool, Any]]) -> list[str]:
    """What a call target is called: its dotted path from a module, if it has
    one, then its qualified name."""
    names = []
    if (q := _qualname(obj)) is None and hasattr(obj, "__self__"):
        owner = obj.__self__ if isinstance(obj.__self__, type) else type(obj.__self__)
        if (o := _qualname(owner)) is not None:
            q = f"{o}.{getattr(obj, '__name__', '')}"
    if q is not None:
        names.append(q)
    # The dotted path from a module: datetime.datetime.now, random.random.
    parts = []
    while isinstance(e, ast.Attribute):
        parts.append(e.attr)
        e = e.value
    ok, root = resolve(e)
    if parts and ok and isinstance(root, types.ModuleType):
        names.insert(0, ".".join([_module(root.__name__), *reversed(parts)]))
    return names


def _mutable(obj: Any) -> bool:
    immutable = (
        type(None), bool, int, float, complex, str, bytes, range, frozenset, enum.Enum, type,
        types.ModuleType, types.FunctionType, types.BuiltinFunctionType, types.MethodType, Cell, Op,
    )
    if isinstance(obj, immutable):
        return False
    if isinstance(obj, tuple):
        return any(_mutable(x) for x in obj)
    if dataclasses.is_dataclass(obj):
        return not type(obj).__dataclass_params__.frozen  # type: ignore[attr-defined]
    return not callable(obj)


def _is_data_class(obj: Any) -> bool:
    if not isinstance(obj, type):
        return False
    if dataclasses.is_dataclass(obj):
        return obj.__dataclass_params__.frozen  # type: ignore[attr-defined]
    return issubclass(obj, tuple) and hasattr(obj, "_fields")


def _user_code(fn: Any) -> bool:
    if not isinstance(fn, types.FunctionType):
        return False
    filename = fn.__code__.co_filename
    return not (filename.startswith(_STDLIB) or filename.startswith(_HERE) or "site-packages" in filename)


# Analysis of one function


class _Function:
    def __init__(
        self,
        fn: Callable[..., Any],
        ctx: str | None,
        ex: Extraction,
        findings: list[Finding],
        seen: set[types.CodeType],
        helper: bool,
    ):
        self.fn = fn
        self.ctx = ctx
        self.ex = ex
        self.findings = findings
        self.seen = seen
        self.helper = helper
        self.closure: dict[str, Any] = {}
        for name, c in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
            try:
                self.closure[name] = c.cell_contents
            except ValueError:
                pass
        self.reported: set[tuple[str, int, str]] = set()
        self.mutable: set[str] = set()

    def analyze(self) -> None:
        code = self.fn.__code__
        if code in self.seen:
            return
        self.seen.add(code)
        try:
            lines, start = inspect.getsourcelines(self.fn)
            self.filename = os.path.abspath(inspect.getsourcefile(self.fn) or code.co_filename)
        except (OSError, TypeError):
            return  # no source: nothing to extract
        self.offset = start - 1
        tree = ast.parse(textwrap.dedent("".join(lines)))
        node = tree.body[0]
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        self.node = node
        self.locals = self._locals(node)
        self.tainted = self._taint(node)
        if self.helper:
            self.ex.helpers.append(self.fn)
        for n in ast.walk(node):
            self._visit(n)

    # Names

    def _locals(self, fn: ast.AST) -> set[str]:
        names: set[str] = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.arg):
                names.add(n.arg)
            elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                names.add(n.id)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n is not fn:
                names.add(n.name)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                names.update((a.asname or a.name).split(".")[0] for a in n.names)
        for n in ast.walk(fn):
            if isinstance(n, (ast.Global, ast.Nonlocal)):
                names.difference_update(n.names)
        return names

    def resolve(self, e: ast.AST) -> tuple[bool, Any]:
        if isinstance(e, ast.Name):
            if e.id in self.locals:
                return False, None
            if e.id in self.closure:
                return True, self.closure[e.id]
            g = self.fn.__globals__
            if e.id in g:
                return True, g[e.id]
            if hasattr(builtins, e.id):
                return True, getattr(builtins, e.id)
            return False, None
        if isinstance(e, ast.Attribute):
            ok, base = self.resolve(e.value)
            if ok and isinstance(base, (types.ModuleType, type)):
                try:
                    return True, getattr(base, e.attr)
                except AttributeError:
                    return False, None
        return False, None

    def _global_read(self, e: ast.Name) -> bool:
        return e.id not in self.locals and (e.id in self.closure or e.id in self.fn.__globals__)

    # Taint: values that depend on inputs, calls or sources, which are
    # traced values in trace mode.

    def _taint(self, fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
        args = fn.args
        params = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
        tainted = {p for p in params if p != self.ctx}
        flows: list[tuple[ast.AST, ast.AST]] = []
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign):
                flows += [(t, n.value) for t in n.targets]
            elif isinstance(n, (ast.AnnAssign, ast.AugAssign)) and n.value is not None:
                flows.append((n.target, n.value))
            elif isinstance(n, (ast.For, ast.AsyncFor, ast.comprehension)):
                flows.append((n.target, n.iter))
            elif isinstance(n, ast.NamedExpr):
                flows.append((n.target, n.value))
        changed = True
        while changed:
            changed = False
            for target, value in flows:
                if self._depends(value, tainted):
                    for t in ast.walk(target):
                        if isinstance(t, ast.Name) and t.id not in tainted:
                            tainted.add(t.id)
                            changed = True
        return tainted

    def _depends(self, e: ast.AST, tainted: set[str]) -> bool:
        for n in ast.walk(e):
            if isinstance(n, ast.Await):
                return True
            if isinstance(n, ast.Name) and n.id in tainted:
                return True
            if isinstance(n, ast.Call):
                ok, fn = self.resolve(n.func)
                if ok and isinstance(fn, Cell):
                    return True  # a handle
                if self._ctx_attr(n.func) in ("now", "random", "config"):
                    return True
        return False

    def _traced(self, e: ast.AST) -> bool:
        return self._depends(e, self.tainted)

    def _ctx_attr(self, e: ast.AST) -> str | None:
        if isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name) and e.value.id == self.ctx:
            return e.attr
        return None

    # Findings

    def report(self, node: ast.AST, rule: str, severity: str, message: str) -> None:
        line = getattr(node, "lineno", self.node.lineno) + self.offset
        key = (rule, line, message)
        if key not in self.reported:
            self.reported.add(key)
            self.findings.append(Finding(f"{self.filename}:{line}", rule, severity, message))

    def _visit(self, n: ast.AST) -> None:
        if isinstance(n, (ast.Name, ast.Attribute)) and isinstance(n.ctx, ast.Load):
            self._reference(n)
        if isinstance(n, ast.Call):
            self._call(n)
        elif isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load):
            attr = self._ctx_attr(n)
            if attr == "resource":
                self.ex.uses_resources = True
                self.report(n, "resource", "error", "ctx.resource in a composite cell; only leaf cells may use resources")
            elif attr in ("request_id", "path"):
                self.report(n, "ctx-escape", "warning", f"reading ctx.{attr} breaks the trace")
        elif isinstance(n, ast.Global):
            self.report(n, "global-write", "error", f"assigns global {', '.join(n.names)}")
        elif isinstance(n, ast.JoinedStr):
            if any(isinstance(v, ast.FormattedValue) and self._traced(v.value) for v in n.values):
                self.report(n, "format", "warning", "formatting a traced value breaks the trace; use an @op")
        elif isinstance(n, ast.Compare):
            self._compare(n)
        elif isinstance(n, ast.ExceptHandler):
            ok, t = self.resolve(n.type) if n.type is not None else (True, BaseException)
            if ok and t is BaseException:
                self.report(n, "catch-all", "warning", "catching BaseException also catches graph breaks, which forces a replay")

    def _reference(self, n: ast.Name | ast.Attribute) -> None:
        ok, obj = self.resolve(n)
        if not ok:
            return
        if isinstance(obj, Cell):
            self.ex.calls.add(obj)
            return
        if isinstance(obj, Op):
            self.ex.ops.add(obj)
            return
        if obj is os.environ:
            self.report(n, "nondeterminism", "error", "reads the environment; use ctx.config")
        elif isinstance(n, ast.Name) and self._global_read(n) and _mutable(obj) and n.id not in self.mutable:
            self.mutable.add(n.id)  # reported once per name
            self.report(n, "mutable-global", "error", f"reads mutable global {n.id!r}; its value isn't journaled")

    def _call(self, n: ast.Call) -> None:
        if isinstance(n.func, ast.Attribute) and n.func.attr == "create_task":
            self.report(n, "task", "error", "spawns a task; issue calls from the body and await their handles")
            return
        ok, fn = self.resolve(n.func)
        if not ok or isinstance(fn, (Cell, Op)):
            return
        found = _names(n.func, fn, self.resolve)
        name = found[0] if found else "?"
        names = set(found)
        short = name.split(".")[-1]
        args = [*n.args, *(k.value for k in n.keywords)]
        traced = any(self._traced(a) for a in args)
        if names & _NONDETERMINISTIC or any(x.startswith(_NONDETERMINISTIC_MODULES) for x in names):
            self.report(n, "nondeterminism", "error", f"calls {name}; use ctx.now, ctx.random or ctx.config")
        elif "builtins.print" in names:
            self.report(n, "io", "warning", "prints; output is lost in compiled mode and repeated on replay")
        elif names & _IO or any(x.startswith(_IO_MODULES) for x in names):
            self.report(n, "io", "error", f"calls {name}; I/O belongs in a leaf cell")
        elif names & _TASKS:
            self.report(n, "task", "error", "spawns a task; issue calls from the body and await their handles")
        elif names & _RACES:
            self.report(n, "race", "error", f"calls {name}, whose outcome depends on timing")
        elif not traced:
            pass
        elif "builtins.range" in names:
            self.report(n, "range", "warning", "a loop over a data-dependent range breaks the trace")
        elif names & _INVISIBLE:
            self.report(n, "invisible", "warning", f"{short}() of a traced value breaks the trace (and is invisible without strict tracing)")
        elif names & _CONVERSIONS:
            self.report(n, "convert", "warning", f"{short}() of a traced value breaks the trace; use an @op")
        elif names & _ORDERING:
            self.report(n, "ordering", "warning", f"{short}() compares traced values: one guard per comparison; use an @op")
        elif not _user_code(fn) and not names & _SAFE and not _is_data_class(fn):
            self.report(n, "opaque-call", "warning", f"passes a traced value to {name}, which the tracer can't see into; use an @op")
        if _user_code(fn):
            self._helper(n, fn)  # traced through, so analyzed as part of the body

    def _helper(self, n: ast.Call, fn: types.FunctionType) -> None:
        """Analyze a plain function called from the body, as part of the body."""
        ctx = None
        params = list(inspect.signature(fn).parameters)
        for i, a in enumerate(n.args):
            if isinstance(a, ast.Name) and a.id == self.ctx and i < len(params):
                ctx = params[i]
        for k in n.keywords:
            if isinstance(k.value, ast.Name) and k.value.id == self.ctx:
                ctx = k.arg
        _Function(fn, ctx, self.ex, self.findings, self.seen, helper=True).analyze()

    def _compare(self, n: ast.Compare) -> None:
        operands = [n.left, *n.comparators]
        for op, left, right in zip(n.ops, operands, operands[1:]):
            if not isinstance(op, (ast.Is, ast.IsNot)):
                continue
            for a, b in ((left, right), (right, left)):
                # None, bools and Enum members are concrete in trace mode (DESIGN §5.2).
                constant = isinstance(b, ast.Constant) and b.value in (None, True, False)
                ok, member = self.resolve(b)
                if self._traced(a) and not constant and not (ok and isinstance(member, enum.Enum)):
                    self.report(n, "identity", "warning", "`is` on a traced value isn't traced; use ==")
                    return


def main(argv: list[str]) -> int:
    """Lint every cell defined in the given modules: python -m cell.static MODULE..."""
    import importlib

    findings = [f for name in argv for c in cells_of(importlib.import_module(name)) for f in lint(c)]
    for f in findings:
        print(f.format())
    return 1 if any(f.severity == "error" for f in findings) else 0


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv[1:]))
