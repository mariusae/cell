# Cell

Cell is an experimental online serving system.

Status: draft proposal. Background and motivation are in
[NOTES.md](NOTES.md).

This document covers the core the rest of the system is built on:

1. the **programming model** for cells,
2. a **graph IR** that represents a cell's dataflow,
3. a **tracer** that recovers that graph from ordinary Python code, and
4. **guards, graph breaks and deopt**, which keep compiled execution
   exactly equivalent to eager execution.

Everything here runs in one local process. Distribution, vmap, planning
and budgets are out of scope, but the IR is designed so they can be added
as passes later (§10).

---

## 0. Goals and non-goals

**Goals**

- **Correctness first.** Running a compiled graph must be observably
  equivalent to running the cell eagerly: the same result, and the same
  effectful calls with the same arguments. When the system is unsure, it
  falls back to eager.
- **A small, explicit IR.** A few node kinds, SSA form, serializable, and
  easy to print, diff and hash.
- **Tracing ordinary Python.** Engineers write normal `async` code. Plain
  helper functions are traced through; they don't need to be cells.
- **Graph breaks are explicit and diagnosable,** not silent.
- **One journal mechanism** for deopt, replay and (later) durability.

**Non-goals for now**

- Distributed execution, placement, vmap, planner passes, budgets.
- Merging traces into graphs with real control flow (trace trees). We
  start with one specialization per cell and add this later (§8).
- Tracing arbitrary Python perfectly. We aim for a *strict* tracer that
  either records correctly or refuses (§5.4).

---

## 1. Programming model

### 1.1 Cells

```python
from cell import cell, op, pure, effects

@cell(pure)
async def get_user(ctx, uid: int) -> User: ...

@cell(effects("prefs"))           # reads/writes declared later; v0: "not pure"
async def get_prefs(ctx, uid: int) -> Prefs: ...

@cell                              # undeclared = external (conservative)
async def home(ctx, uid: int) -> Page:
    ...
```

- A cell is an `async` function whose first parameter is `ctx`.
- Calling a cell requires a ctx: `get_user(ctx, uid)`. The cell object is
  a typed reference; there are no string names.
- Semantics are declared in the decorator. The v0 tracer and runtime need
  only one distinction: **pure** vs. **effectful**. Undeclared cells are
  effectful (`external`). The richer vocabulary (reads/writes, idempotent,
  compensable, vector forms) attaches to the same declaration later.
- A cell annotated `-> None` must return None; the runtime enforces it.
  The tracer relies on it: the result of such a call needs no guard.

### 1.2 Data

Cell arguments and results are **data**:

- `None`, `bool`, `int`, `float`, `str`, `bytes`
- `tuple`, `list`, `dict` with `str` keys
- frozen dataclasses and `NamedTuple`s
- `Enum`s

Data is immutable by convention (lists and dicts are treated as values),
serializable, and has a known structure. That makes it a *pytree*: the
tracer can flatten and rebuild it (§5.3), and operations on it (field
access, indexing) are known to be pure.

Pytrees are also the basis for vectorization later (§10). As in JAX's
`vmap`, a vector of values is represented leaf by leaf: a vector of
`User`s becomes one `User`-shaped tree whose leaves are vectors. So
getting data types right now is a prerequisite for vmap, not only for
tracing.

### 1.3 Ops

```python
@op
def greeting(user: User) -> str:
    return f"Hello, {user.name}"
```

An `@op` is a pure, synchronous, deterministic local function. The
tracer records a call to an op as **one node** and does not trace inside
it. The op always runs on concrete values. Ops are how arbitrary Python
(formatting, sorting, library code) enters a graph without tracer
limitations.

### 1.4 Rules for cell bodies

These rules already follow from NOTES §4. The tracer depends on them.

1. **Nondeterminism goes through ctx.** Use `ctx.now()`, `ctx.random()`,
   `ctx.config(key)`. No direct I/O, clocks, randomness, or mutable
   globals.
2. **Calls are issued when made, not when awaited.** `h = f(ctx, x)`
   starts `f` immediately and returns a handle; `await h` waits for it.
   This matches dataflow semantics (like JS promises, unlike Python's lazy
   coroutines) and makes program order well defined.
3. **Structured concurrency.** A cell completes only after every call it
   issued has completed. Errors from awaited calls raise at the `await`.
   An effectful call that is never awaited fails its parent if it fails.
   A pure call that is never awaited has its result and error discarded.
4. **No racing in user code.** `asyncio.wait(FIRST_COMPLETED)` and
   `as_completed` are nondeterministic. If a race is needed later, it will
   be a ctx primitive (`ctx.race`) whose outcome is journaled.
5. **Calls and sources come from the body's own task.** If a body spawned
   tasks that each issued calls, the order of issues would depend on
   scheduling, and `seq` numbers would not be deterministic. The runtime
   rejects calls made from any other task. Concurrency comes from issuing
   calls and awaiting their handles, which never needs another task.
   `asyncio.gather` over handles is fine.

**Leaf cells and I/O.** Leaf cells implement the I/O boundary: clients,
databases, fakes. They get live objects from `ctx.resource(key)`, which is
provided by the runtime and not journaled. A composite cell must not use
resources, or it is no longer deterministic. Replay never runs a leaf
whose call is journaled, so leaves don't need to be deterministic
themselves.

Plain helpers, sync or async, that take ctx and call cells are fine. The
tracer sees through them. **Cell boundaries mark where policy applies,
not how code is organized.**

### 1.5 Passing handles (pushdown)

A handle can be passed to a call without awaiting it first:

```python
if user.tier == "premium":
    ranked = await rank(ctx, items, prefs)     # handles, not values
```

This pushes the reference down into the call, and the value is resolved
where it is needed. The caller never waits for `items` or `prefs`, and
never has to hold their values. This is *promise pipelining*, as in the E
language and Cap'n Proto.

Rules:

- **Where a handle may go.** A `Handle[T]` may be passed anywhere a `T`
  is expected by a cell: as an argument, or as a leaf inside data
  (`Req(items=items)`).
- **Issue vs. start.** The call is *issued* at the call site, in program
  order. Its `seq`, effect edges and effect domain are the same as for
  any call. It *starts* only when its handle arguments have resolved.
- **Projection doesn't wait.** `prefs.weights` and `items[0]` on a
  handle give a handle to the field or element. Anything that needs the
  actual value (ops, branching, arithmetic) requires an `await`.
- **Errors.** If a handle argument fails, the call fails with that error
  before it starts. The callee never runs, so an effectful callee has no
  effect. The error surfaces at the `await` of the call rather than at an
  `await` of the handle. The cell's outcome is the same error.
- **Passing a handle doesn't observe it.** So it creates no path edge
  (§4.3) for later effects. This is the real semantic difference from
  `rank(ctx, await items, await prefs)`: the caller no longer depends on
  `items` and `prefs` having succeeded. Only `rank` does.

**Where the value is resolved.** Locally, the resolver just waits for the
arguments before starting the callee. When execution is distributed
(§10), the value travels along the data edge directly from producer to
consumer, and never passes through the caller. This is the same
behavior the dataflow model gives compiled graphs (NOTES §11), made
available to eager code.

**In the graph** nothing new is needed. A data edge already *is* a
reference:

```
%8 = call rank(%4, %3)       # same node whether or not the caller awaited
```

The only difference is that `%4` and `%3` are not observed, so later
effects get no path edges to them. Compiled mode therefore already
behaves this way whenever nothing observes the values. The explicit form
matters for eager mode, for distribution, and for making the intent
clear.

**Later: lazy parameters.** By default a callee is strict: it starts
when all its arguments have resolved. That keeps cell bodies plain and
vectorization simple. A callee could instead declare a parameter as
`Ref[T]` and receive the handle itself. It could then start work before
the argument is ready, await it only on the paths that need it
(call-by-need), or pass it further down without materializing it.

---

## 2. Execution modes

| Mode | What happens | Used for |
|---|---|---|
| **Eager** | Run the Python body. Each call is resolved as it is issued. | Default; always correct |
| **Trace** | Eager, plus recording: values are wrapped in tracers and every call, op and branch is recorded. | Sampled requests; dev tooling |
| **Compiled** | Interpret the cell's graph. Calls are issued as soon as their inputs are ready. | Hot paths with a validated graph |

All three share one runtime object, the **resolver**. It maps a call
(cell, args, ctx) to an execution. The resolver decides, per call, whether
the callee runs eagerly or from a compiled graph. Callers can't tell the
difference.

**Central invariant.** For any cell C, input x, and call results R:

> Compiled(C, x, R) and Eager(C, x, R) produce the same result or error,
> and the same *effectful* calls (cell and args), in orders consistent
> with the ordering C's effect domains require (§4.4). Compiled may
> additionally issue *pure* calls speculatively.

With the default single domain, this is exactly eager's order. Declaring
other domains relaxes the program's meaning, so eager's sequential order
becomes one of several allowed orders (§4.4).

Here "call results R" means that each call receives the same result for
the same (cell, args) in both runs. §7 gives the argument for why the
design preserves this invariant.

---

## 3. Journals and call paths

The journal is the one mechanism behind deopt (§6), and later behind
replay and durability.

**Call path.** Each call made by a cell invocation gets a key:

```
key = (parent_path, seq, cell_id, args_digest)
```

- `seq` is the issue index within the parent invocation (0, 1, 2, …) in
  program order.
- `cell_id` and `args_digest` let replay detect a key that is reused for
  a different call.

A child's path is `parent_path + (seq,)`. Given the determinism rules,
eager execution of a cell always produces the same sequence of keys for
the same call results.

**Journal.** A per-invocation map from key to entry. An entry is either
completed (a result or an error) or in flight (a handle).

**Replay.** Run the cell eagerly, with its calls resolved as follows. For
each issued call, look up its key:

- **Hit** (seq, cell and args all match): return the journaled result, or
  await the in-flight handle. The call is not re-executed.
- **Miss:** execute the call normally.
- **Mismatch on an effectful call** (same seq, different cell or args):
  a determinism violation. Fail loudly. §7 explains why this can't happen
  if the invariants hold.

**Unconsumed entries.** When replay ends, the journal may still hold
effectful calls that replay never reached. This happens only when effect
domains are relaxed (§4.4): an effect in another domain ran ahead of a
call that then failed. These effects did happen. Replay waits for them to
complete (structured concurrency) and reports them with the cell's
outcome, so that EXPLAIN ANALYZE, and later durability, account for them.

---

## 4. Graph IR

### 4.1 Shape

One graph per specialization of a cell. SSA form: every node defines at
most one value, and nodes are stored in a topological order that follows
program order.

```
Graph
  cell:      str                 # cell id
  code:      str                 # the cell's code hash
  params:    [str]               # parameter names (excluding ctx)
  nodes:     [Node]              # topological, program order; ends in return or deopt

Node
  id:        int                 # also its value's id, if it produces one
  kind:      param | call | op | pack | source | guard | deopt | return
  inputs:    [int | Const]       # data dependencies, or constants
  attrs:     kind-specific
  path:      [int]               # path edges (§4.3)
  effect:    [int]               # effect edges: completion, same domain (§4.3, §4.4)
  after:     int | None          # issue-order edge: previous effect, same domain
  site:      file:line           # for EXPLAIN and diagnostics
```

Constants are operands, not nodes: they print inline (`%5 == 'premium'`)
and don't need ids. A node's value is its result; `refs` (data inputs,
`path` and `effect`) are the nodes that must complete successfully first.

### 4.2 Node kinds

| Kind | Inputs | Attrs | Meaning |
|---|---|---|---|
| `param` | — | name, index | A cell input |
| `call` | args (one per parameter) | cell ref, semantics (pure/effectful), effect domain, `seq` | Issue a call to a cell |
| `op` | args | op ref (id + code hash), or a builtin name (§5.2) | Pure local computation |
| `pack` | leaves | treedef (structure) | Build a data value from its leaves |
| `source` | — | `now` \| `random` \| `config(key)` | Nondeterminism through ctx; journaled |
| `guard` | value | expected value | Assert value == expected, otherwise deopt |
| `deopt` | — | reason | Unconditional deopt: a graph break (§6.2) |
| `return` | value | — | Result of the cell |

That's the whole v0 IR. Later additions (§10): `map` (structured fan-out
with a subgraph), `switch`/`merge` (trace trees), `send`/`recv`
(partitioning), and a vector annotation on every value.

### 4.3 Dependencies and ordering

Data edges alone would allow any reordering. That is fine for pure nodes
and wrong for effectful ones. The rules:

- **Pure nodes** (`param`, `op`, `pack`, pure `call`) are ordered
  only by data edges. They may run earlier than in eager, including
  speculatively before a guard that precedes them in program order. The
  cost of a failed speculation is only wasted work.
- **Effectful calls** get two kinds of control edges. Both refer to nodes
  *observed* before the call was issued, in program order. A node is
  observed when user code consumes its value:
  - a call is observed when awaited,
  - an op when it evaluates,
  - a guard when it is checked.

  The two kinds:

  - **Path edges** go to every observed guard, op and *pure* call. They
    ensure the effect happens only on the path eager would take, and only
    if the values that led to it were computed without error. **Path
    edges are never relaxed.**
  - **Effect edges** go to other effectful calls *in the same effect
    domain* (§4.4). There are two kinds:
    - `effect`: the *completion* of every observed effect in the domain,
      so an effect proceeds only if the earlier ones it waited for in
      eager succeeded;
    - `after`: the *issue* of the previous effect in the domain, so
      effects in a domain are issued in program order even when nothing
      waited for them.

  With the default single domain (`main`), this is exactly eager's
  behavior: an effect runs only if everything before it succeeded and
  passed.
- **`source` nodes** are journaled, so they are ordered only by data.

Calls issued but not yet awaited before an effectful call are *not* its
control dependencies, because eager doesn't wait for them either.

Data edges always apply, whatever the domains. If an effect uses another
effect's result, or code branches on it, the data edge or the guard's
path edge orders them.

**Implied edges are left out.** A node's successful completion implies
that of everything it depends on through data, `path` and `effect`
edges. So an edge to X is omitted when X is already implied by the node's
other edges, or by another edge in the same list; `after` is omitted when
the previous effect's completion is already implied. For example,
`get_prefs(user.id)` needs no path edge to `get_user`: it already depends
on it through `user.id`.

### 4.4 Effect domains

By default, every effect in a cell body is in the domain `main` and is
ordered with every other effect, as above. This is safe but
over-constrained. An audit-log write awaited at the top of `checkout`
would hold up the reservation until the log write completes, and a failed
log write would cancel the order.

An **effect domain** relaxes this. Effects are ordered, by effect edges,
only with other effects in the *same* domain. Effects in different
domains are ordered only by data and path edges.

```python
@cell(effects("audit"), domain="audit")      # the cell's default domain
async def audit(ctx, event: Event) -> None: ...

@cell(effects("email"), domain=UNIQUE)       # every call is its own domain
async def notify(ctx, receipt: Receipt) -> None: ...

@cell
async def checkout(ctx, order: Order) -> Receipt:
    await audit(ctx, Attempt(order.id))      # "audit"
    await reserve(ctx, order)                # main
    receipt = await charge(ctx, order)       # main: after reserve
    await notify(ctx, receipt)               # own domain: after charge, via data
    await audit(ctx, Charged(receipt.id))    # "audit": after Attempt (domain) and charge (data)
    return receipt
```

In this example:

- `reserve` no longer waits for the first audit write.
- `notify` still runs after `charge`, because it uses `receipt`.
- The second audit write is ordered after the first by its domain, and
  after `charge` by data.

**Where the domain comes from.** Ordering is a relation between calls in
the *caller's* body, so a domain is fundamentally a property of the call
site. A cell declares a default domain for calls to it (as above). A call
site can override it:

```python
with ctx.domain("audit"):
    await write_metrics(ctx, m)
```

Domain names are scoped to the calling cell's body. The special domain
`UNIQUE` makes every call a singleton domain: it has no effect edges at
all, not even with other calls to the same cell.

**Meaning.** A domain declaration is an assertion by the author: *effects
in this domain don't depend on effects in other domains having happened,
or having succeeded, unless there is a data or path dependency.* This
changes the program's meaning, and the reference semantics change with
it:

- Effects form a **partial order**: program order within each domain,
  plus data and path edges.
- A failed effect gates later effects **in its own domain only**. The
  failure still surfaces at its `await` and fails the cell as usual.
- As a result, when a cell fails, effects in *other* domains that come
  later in program order may already have happened. Replay reports them
  as unconsumed journal entries (§3).

Eager's sequential run is one linearization of this partial order, so
eager remains a correct implementation. With only `main`, the partial
order is total and nothing changes. The analogy is a memory model:
`main` is sequential consistency, and domains are declared relaxations.

**Domains never relax path edges.** Letting an effect run before the
guards that lead to it would mean running effects on paths eager doesn't
take. That is *speculating effects*. It needs a different semantic
(compensable or reservation-style effects) and is future work.

**Relation to reads/writes.** Domains express the author's intent about
ordering between call sites. Declared `reads`/`writes` sets (NOTES §3)
are about which effects commute. A later pass can use commutativity to
drop effect edges *within* a domain. The two are complementary. A planner
could also propose default domains from write sets, but only the author
can assert the independence a domain promises.

### 4.5 Identity and serialization

- **Graph identity** is a content hash of the normalized node list: kinds,
  attrs, edges, and the hashes of the cells and ops it references. The
  same code and the same trace shape give the same graph ID.
- **Graph key** for lookup is `(cell_id, cell_code_hash, graph_hash)`.
  When code changes, its graphs become invalid and are re-traced.
- **Serialization** in v0 is JSON, chosen for readability. A compact form
  can come later.
- **Logical vs. physical.** The graph is the logical plan: it holds no
  placement, mechanism or vector width. Physical plans are later
  *annotations* keyed by node ID, so the logical graph stays stable and
  can be diffed.

### 4.6 Example

```python
@op
def greeting(user: User) -> str:
    return f"Hello, {user.name}"

@cell
async def home(ctx, uid: int) -> Page:
    user = await get_user(ctx, uid)
    prefs = get_prefs(ctx, user.id)          # issued, not yet awaited
    items = get_items(ctx, uid)              # issued, not yet awaited
    if user.tier == "premium":
        ranked = await rank(ctx, await items, await prefs)
    else:
        ranked = await items
    return Page(title=greeting(user), items=ranked)
```

Traced on a premium user (`get_user`, `get_items` and `rank` are pure;
`get_prefs` is effectful):

```
graph home #605d3784 (uid)
  %0 = param uid
  %1 = call get_user(%0)               pure  seq=0             home.py:78
  %2 = op %1.id                                                home.py:79
  %3 = call get_prefs(%2)              effectful[main]  seq=1  home.py:79
  %4 = call get_items(%0)              pure  seq=2             home.py:80
  %5 = op %1.tier                                              home.py:81
  %6 = op %5 == 'premium'                                      home.py:81
  %7 = guard %6 == True                                        home.py:81
  %8 = call rank(%4, %3)               pure  seq=3             home.py:82
  %9 = op greeting(%1)                                         home.py:85
  %10 = pack Page(title=%9, items=%8)
  return %10
```

This is the actual output of the tracer, from `examples/home.py`. The
graph of every example scenario is in `tests/golden/`.

Things to notice:

- **`get_items` doesn't depend on `get_user`.** Program order serialized
  them; the graph doesn't, so compiled mode runs them in parallel. This is
  the first optimization tracing gives us for free.
- **`get_prefs` needs no path edge.** Eager only reaches it after
  `await get_user` has succeeded, but it depends on `get_user` through
  `%2` already.
- **`rank` can start speculatively before `%7` is checked,** because it
  is pure.
- **The trace is specialized to the premium branch.** A non-premium user
  fails `%7` and deopts (§6).

---

## 5. Tracer

### 5.1 Mechanism: concrete tracing

The tracer runs a **real request eagerly**, with every value that comes
from a parameter or a call wrapped in a `Tracer`. A tracer holds:

- the concrete value, and
- the value ID of the node that produced it.

Calls really execute, through the same resolver as eager mode, so user
code always has a concrete value available. The tracer therefore never
has to guess a value; it only has to decide what to *record*.

This is concolic tracing, as in Dynamo: correct behavior for the current
request, plus a graph valid under the recorded guards.

In trace mode `ctx` is a `TracingCtx`. It:

- records a `call` node for each call, with `seq` and control edges
  computed as in §4.3,
- delegates to the real resolver, and
- wraps the result in a tracer when it is awaited, which also marks the
  call as observed.

### 5.2 What can be done with a tracer

| Operation | Recorded as | Notes |
|---|---|---|
| Pass to a cell call | `call` input | Nested tracers are flattened and packed (§5.3) |
| Pass an *unawaited handle* to a cell call | `call` input | Data edge, not observed (§1.5) |
| `.field`, `[i]` on an unawaited handle | `op getattr` / `op getitem` | Projection; not observed (§1.5) |
| Pass to an `@op` | `op` node | The op runs on unwrapped concrete values; its result is wrapped |
| `.field` on data | `op getattr` | Allowed only on data types (§1.2), so it is known to be pure |
| `[i]`, `[k]` | `op getitem` | |
| `+ - * / // % == != < <= > >= & \| ^ ~ -x` | `op <operator>` | Comparison results are tracers too |
| Pure methods on immutable builtins (`str.lower`, `tuple.index`, …) | `op method` | From a fixed allowlist |
| `bool(t)`, `if t:`, `and`/`or`/`not` | `op truth` + `guard` on the result | Concretizes the value (§5.4) |
| `len(t)` | `op len` + `guard` on the result | Python requires `len` to return an int |
| `iter(t)`, `for x in t` | `guard` on length, then one `getitem` per element | Tuples and lists only. Unrolled. Use `ctx.map` for data-dependent fan-out (§10) |
| A value that is None, a bool or an Enum member | Returned concretely, with a `guard` on its value | Code compares these with `is`, which a tracer can't intercept |
| `int(t)`, `float(t)`, `hash(t)`, `str(t)`, `repr(t)`, `format(t)`, … | **graph break** | A guard on the exact value would almost never hold, so break instead |
| `isinstance(t, C)`, `t.__class__` | **graph break** | The type of a value is data-dependent too (§5.4) |
| Mutation (`setattr`, `setitem`, `append`, …) | **graph break** | Data is immutable |
| An op or builtin that raises while tracing | **graph break** | Replay raises it in eager mode, where user code may handle it |
| Passing to any other callable | **graph break** (§5.4) | Not yet enforced; see §5.4, defense 3 |

### 5.3 Pytrees

When a structure containing tracers is passed to a call or an op, or
returned from the cell, the tracer:

1. flattens it into leaves and a treedef (the structure),
2. makes concrete leaves constant operands, and
3. emits a `pack` node. Identical packs (the same structure and operands)
   share a node, so a value used twice is packed once.

User code builds values the ordinary way (`Page(title=t, items=u)`), and
the graph gets a `pack`. Construction runs the dataclass's
`__init__`/`__post_init__` on tracers. If that code inspects a value, the
inspection is caught by the rules above, typically as a guard.

The reverse direction needs no special node: `getattr` and `getitem` take
data apart.

### 5.4 Strictness: what makes the tracer sound

The tracer is sound only if **every way a traced value can influence
Python's behavior goes through a tracer hook.** Any hook we miss is a
silent constant: a value that depends on the input gets frozen into the
graph. Defenses, in order:

1. **Every concretization becomes a guard or a break.** Concretizing
   dunders (`__bool__`, `__len__`, `__iter__`, `__index__`, `__hash__`,
   `__str__`, …) either record a guard or trigger a graph break. None
   returns a concrete value without recording. Guards are limited to
   booleans, lengths, and values that are None, bools or Enum members;
   anything else breaks.
2. **Tracers are opaque, and type tests break.** A tracer is not a
   subclass of the wrapped type. `isinstance` consults `__class__` when
   the type doesn't match, so a tracer's `__class__` is a property that
   breaks: a type test never silently answers False. `type(t)` can't be
   intercepted, and neither can `is` on values other than None, bools and
   Enum members (which are never tracers, see §5.2). Those are left to
   defenses 3, 5 and 6.
3. **Calls from cell frames are checked.** During tracing, `sys.monitoring`
   (PEP 669) watches calls made from traced frames: the cell body and any
   plain helpers it calls. If a tracer is passed to anything that is not a
   cell, an `@op`, an allowlisted builtin, or a traced helper, it's a
   graph break. (To verify: exactly which call events and arguments are
   visible. The fallback is a conservative break whenever an unknown
   callable is invoked while any tracer is reachable from its arguments.)
4. **Guard budget.** If one trace records more than *N* guards (e.g.
   `sorted()` on a list of tracers, which guards on every comparison),
   break and report it. The fix is an `@op`.
5. **Differential validation before promotion.** A graph is used for
   compiled execution only after it has been replayed against recorded
   requests and matched eager mode exactly: same result, same effectful
   calls (§9). This catches whatever 1–4 miss.
6. **Static lint.** Flag reads of mutable globals, direct I/O, and calls
   to non-op functions with values derived from calls.

Defenses 1–4 make the tracer strict; 5 makes the system safe even when
strictness has a hole.

M1 implements defenses 1, 2 and 4. Until defense 3 exists, one hole is
known: a tracer passed to library code that raises (for example
`", ".join(names)` with traced names raises `TypeError`), where user code
catches the error with `except Exception`, takes the except branch
silently. `GraphBreak` itself is a `BaseException`, so `except Exception`
never swallows a break.

### 5.5 Breaks during tracing

When the tracer hits a break at some point P in the trace:

1. **Stop recording.** Everything recorded before P stays in the graph.
   Append a `deopt` node with the reason and source location. The
   invocation is stopped too: it refuses any further calls, even from
   `finally` blocks or code that catches the break.
2. **Finish the request correctly.** The traced request is a real
   request and must still complete. Raise an internal `GraphBreak`, abort
   the body, and re-run the cell eagerly **with replay from the trace's
   journal** (§3). The calls already made are replayed, not re-executed,
   so effectful calls happen exactly once. This is the same path as
   deopt (§6), so there is only one mechanism to get right.

   The alternative is *pass-through*: keep running and unwrap tracers
   into their concrete values from then on. It avoids re-running the
   body. But tracers the user code already holds (in locals, or inside
   containers) would keep recording, or would need unwrapping in place,
   and that is hard to get right. Start with replay.

### 5.6 Exceptions

- **Call errors.** A trace records the calls that succeeded. If a call
  fails in compiled mode, that's a deopt: eager replay re-raises the
  journaled error at the `await`, and the user's `try/except` handles it.
  The graph doesn't model exception control flow.
- **A call that fails while tracing** is a graph break at its `await`,
  whether or not user code would catch the error. Replay re-raises it
  there, and the user's `try/except` handles it eagerly.
- **An exception that escapes the body while tracing** is a break too.
  The tracer can't tell a genuine error from one caused by a tracer
  reaching code that can't handle it, so it lets replay produce the true
  outcome.
- **Errors raised by ops** also deopt in compiled mode. Replay raises the
  same error from the op, because ops are deterministic. While tracing,
  an op that raises is a break.

### 5.7 Nested cells and inlining

Each cell is traced separately and gets its own graphs. Tracing a request
traces only its root cell; the cells it calls run eagerly. A `call` to a
composite cell is a black box in its parent's graph. At run time, the
resolver decides independently whether the callee runs eagerly or
compiled.

**Inlining** (a later pass) replaces a `call` node with the callee's
graph. Each inlined guard keeps a **deopt scope**: the call site it came
from. If it fails, only that callee deopts. The callee replays eagerly
from its own journal, and its result feeds back into the parent graph at
the call site. So the unit of deopt is always the cell whose assumption
failed.

---

## 6. Guards, graph breaks and deopt

### 6.1 Guards

A `guard` checks that a value equals the one observed during tracing.
In v0 the value is a bool (from `op truth` or a comparison), a length
(from `op len`), or a None, bool or Enum value user code saw concretely.
It runs as soon as its input is ready, which is often before the Python
program would have reached it.

### 6.2 Graph breaks

A graph break is a `deopt` node: a guard that always fails. The graph is
a **compiled prefix**. Compiled execution runs the prefix, then
hands off to eager at the break.

Guard failure and graph break are the same event with the same handling.
There's one code path.

Even a prefix is useful. The calls in it still get dataflow parallelism,
and later vectorization. The eager part is correct, just slower.

**Future work:** resume tracing after a break, producing multi-segment
graphs, with a new trace keyed on the break's position.

### 6.3 Deopt

When a guard fails or a `deopt` node is reached during compiled execution
of cell C:

1. **Stop issuing new nodes.** Cancel in-flight *pure* calls that nothing
   will need, or let them finish in the background and journal them.
   In-flight effectful calls continue; they cannot be cancelled safely.
2. **Build the journal** from every call node that was issued: a key
   (from `seq`, cell, args) and its entry, completed or in flight.
3. **Run C eagerly with replay from that journal** (§3). Journaled calls
   return their recorded results, and effectful calls are never
   re-executed. Eager mode continues past the point of failure normally.
4. **Record the failure** (which guard, how often). The planner uses it to
   decide whether to retrace or add a specialization.

Speculated pure calls on the wrong branch may sit in the journal under a
`seq` that eager assigns to a different call. The key check detects this
as a miss, and a miss on a pure call just executes.

**What deopts.** In M2, any failure in a compiled run deopts: a guard that
doesn't hold, a `deopt` node, and any op or call that raises. Errors
therefore always come from eager execution, which defines them. A failed
call deopts even if nothing used its result, because the graph can't tell
a call that was awaited and ignored (eager raises) from one that was
never awaited (eager discards the error). Recording which calls were
observed would let the executor skip that deopt.

**Replay after a deopt** uses the compiled attempt's journal, merged over
the journal the invocation was itself replaying from, if any (as in
validation, §9). Calls the attempt left in flight are waited for, by
replay where it reaches them and by the invocation afterwards, so every
call issued still completes before the invocation does.

**Executing a graph** (`compiled.py`). A node starts once every node it
refers to (data inputs, `path`, `effect`) has completed successfully and
its `after` node has been issued. Calls and sources are issued at the
`seq` the graph recorded, so the journal is keyed as eager would key it.
A call to a *pure* cell may also start when an input call has only been
issued, taking its handle (§1.5). An effectful call never does: edge
reduction (§4.3) leaves out path edges that a data input implies, which
holds only if data inputs complete before the node starts.

---

## 7. Correctness argument

Two assumptions:

- **(A1) Determinism.** Given call results and `source` values, the cell
  body is deterministic (the rules of §1.4).
- **(A2) Tracer completeness.** Every influence of a traced value on the
  body's behavior was recorded as an op, a guard or a break (§5.4).

**Claim (no deopt).** Suppose compiled execution of graph G reaches
`return`. Then eager execution, with the same call results, takes the same
path. The reason: every branch the eager body takes depends only on
traced values (A2), and every such dependency is a guard that passed. By
A1, eager computes the same arguments for the same calls and the same
result. G's effectful calls are exactly eager's, because an effectful call
was recorded only on the traced path, and the guards pin that path. Their
order is consistent with the partial order the domains require (§4.4),
because of the effect edges. With only `main`, that is eager's order.

**Claim (with deopt).** Take any effectful call E that was issued before
the failure. Its path edges passed: every guard, op and pure call observed
before it in program order had succeeded. So E is on eager's path, and
it has the same arguments (data edges) and the same `seq`. There are two
cases:

- **Nothing before E in program order fails in eager.** Then eager
  reaches E, replay hits it in the journal, and E is not re-executed.
- **An effect X before E in program order fails.** Because E didn't wait
  for X, X must be in a different domain. Eager raises at X and never
  reaches E, so E stays unconsumed. The domain semantics (§4.4) allow
  exactly this: a failure gates only its own domain.

From the failure onward, execution is plain eager. So the combined
execution is an allowed behavior of C. With only `main`, the second case
can't occur, and the result is exactly eager's, apart from the pure calls
that were speculated.

**Where it can break:** only through a violation of A1 or A2. A1
violations are caught by the mismatch check in §3 and by lint. A2
violations are caught by the strict tracer (§5.4, defenses 1–4), backed
by differential validation (defense 5).

**Tests should target this invariant directly** (§9).

---

## 8. Specializations

v0 policy:

- **Each cell has at most one active graph:** the most frequent trace
  shape among sampled traces that passed validation.
- **Guard failure rate is tracked per guard.** When a guard fails often,
  retrace from failing requests. Keep the new graph as a second
  specialization.
- **Dispatch** in v0: try the active graph, and on failure deopt. There is
  no cross-specialization dispatch yet.

**Later (trace trees):** merge specializations that share a prefix into
one graph, with `switch`/`merge` nodes at the points where they diverge.
A guard failure then jumps to the sibling branch instead of deopting. This
turns guard-specialized graphs into real dataflow control flow, which the
vmap pass needs anyway to split vectors (§10).

---

## 9. Static extraction and validation

**Static extractor** (AST over the cell body and the helpers it reaches):

- produces the over-approximated **call graph**: every cell the body could
  call, found through typed references,
- produces **lint**: mutable global reads, direct I/O, clocks, randomness,
  racing primitives, and values from calls passed to non-op functions,
- is used for the safety checks in NOTES (capabilities, legality of
  policies). It is never used for optimization.

**Differential validator** (`validate.py`):

- Take recorded requests (inputs plus journaled call results) and run each
  one in compiled mode against the recorded results. The record is the
  eager run, so only the compiled run is needed, and no services: calls
  are answered from the journal.
- Compare the result, the error, the multiset of effectful calls (cell and
  args), and the order constraints between them.
- A graph is **promoted** to active only after passing on a sample set.
- Also run continuously in the background on a sample of production
  traffic.

**Property tests for the runtime itself:** generate random cell programs
from a small grammar (calls, ops, branches, loops, effectful and pure
leaves) with random inputs and random call results. Assert the central
invariant of §2 for both no-deopt and forced-deopt runs, including a
guard forced to fail at each position.

---

## 10. Designed for later passes

The v0 IR avoids decisions that would block these:

- **vmap.** Every node kind needs a *batching rule*:
  - `param`: becomes a vector.
  - constant operands: broadcast.
  - `op`: maps over the vector, or uses a declared vector form.
  - `pack`: element-wise.
  - `call`: uses the declared vector form, or falls back to batching.
  - `guard`: *partitions* the vector. Failing elements deopt individually.
  - `source`: per element.

  ctx becomes a vector of contexts.
- **Structured fan-out.** `ctx.map(cell, xs)` becomes a `map` node with a
  subgraph. This is preferred over unrolled loops, because it
  vectorizes and doesn't need a length guard.
- **Partitioning.** Placement is an annotation on nodes. A pass inserts
  `send`/`recv` edges between partitions, and each partition becomes its
  own graph (endpoint projection). A data edge between partitions sends
  the value directly from producer to consumer, which is how handles
  passed through a caller (§1.5) avoid a round trip through it.
- **Planner rewrites** as graph-to-graph passes: dedup of identical pure
  calls, caching (wrap a pure call), dropping ordering edges between
  commuting effectful calls, inlining.
- **EXPLAIN** is the printer in §4.6, plus annotations. **EXPLAIN
  ANALYZE** is the same view, overlaid with a trace's timings and
  outcomes.

---

## 11. Components and milestones

All local, one process, `asyncio`, Python 3.12+ (for `sys.monitoring`),
few dependencies.

| # | Milestone | Contents | Done when |
|---|---|---|---|
| M0 | Eager runtime | `@cell`, `@op`, data types, ctx (`now`, `random`, `config`), resolver, calls issued when made, structured concurrency, call paths, journal, replay | Example apps run eagerly; replay reproduces a run exactly |
| M1 | Tracer + IR | `TracingCtx`, tracers, pytrees, guards, breaks via replay, printer, JSON serialization, content hashing | Golden-file tests of printed graphs for the example apps |
| M2 | Compiled execution | Dataflow graph interpreter, guard checks, deopt via journal, differential validator | Property tests of §9 pass, including forced deopts |
| M3 | Static extraction | AST call graph, lint, strictness via `sys.monitoring` | Lint catches seeded violations; call graph ⊇ traced graphs |
| M4 | First passes | Dataflow parallelism (free with M2), dedup, caching of pure calls, inlining | Measurable latency wins on the example apps under simulated latency |
| M5 | Local vmap | Collect similar requests at the root and vmap the graph; `map` nodes; guard partitioning | Throughput/latency curves vs. eager under a load generator |
| M6 | Multi-process | Partitioning, `send`/`recv`, local "cluster" of processes | Same results as M5 across processes |

**Test harness:** a **fake-service layer** for leaf cells with
configurable latency distributions, error rates and vector cost curves,
so the benefit of each idea can be measured locally.

**Example applications** (each exercises different parts):

1. **Home page:** the §4.6 example. Fan-out, a branch, parallelism
   recovered from program order.
2. **Feed with ranking:** a fan-out over items (`ctx.map`), a vectorized
   feature lookup and ranker. Cross-request vectorization.
3. **Checkout:** effectful calls (reserve, charge, confirm). Exercises
   ordering, control edges, and deopt correctness in the presence of
   effects. Later, sagas.

**Status.** M0 is implemented in `src/cell`. The example programs are in
`examples/`: the three applications above, plus `features.py`, a set of
small programs that each pin down one rule of §1. Each program comes
with scenarios: inputs, an initial fake world, and the expected result,
error and effects. `tests/test_examples.py` checks every scenario three
ways:

- **eager:** the expected result and effects;
- **full replay:** replaying the journal on a fresh world reproduces the
  run without executing any leaf call;
- **interrupt and replay:** interrupting the body before each `seq` and
  replaying from the partial journal gives the same outcome, with every
  leaf call and effect happening exactly once. This is the deopt path of
  §6.3, exercised at every position.

Later milestones add checks to the same scenarios, for example that
compiled execution matches eager.

M1 is implemented in `src/cell/graph.py` and `src/cell/trace.py`.
`tests/test_trace.py` checks every scenario:

- **tracing doesn't change the request:** the outcome and journal match
  eager execution, and every leaf call and effect happens exactly once,
  including when the trace breaks;
- **golden graphs:** each scenario's graph matches `tests/golden/`;
- **invariants:** topological order, edges only where §4.3 allows them,
  call and source `seq`s matching the journal;
- **JSON round-trips** preserve the graph and its hash.

Targeted tests cover each rule of §5.2 and the edges of §4.3–4.4.

M2 is implemented in `src/cell/compiled.py` (the executor), the runtime
(`Runtime.install`, and deopt into eager replay) and
`src/cell/validate.py`. `tests/test_compiled.py` checks:

- every graph traced from a scenario, run on every scenario of the same
  cell, matches eager execution, whether it runs compiled or deopts;
- a deopt forced at each node of each graph still matches;
- the validator accepts every graph against recorded runs, and rejects a
  graph that returns the wrong value;
- independent calls run in parallel, pure calls are speculated past a
  guard, effects are not, and nested cells run compiled.

`tests/test_random_programs.py` generates 60 programs from a small
grammar (§9), traces each on several inputs, and runs every graph on
every input, plus a deopt forced at each node, against eager execution.

---

## 12. Open questions

1. **Calls issued when made vs. Python coroutines.** Issuing on call is
   the right dataflow semantics, but it surprises Python users. Is a
   custom awaitable enough, or do we need linting against unawaited
   handles?
2. **Pass-through vs. replay for breaks during tracing.** §5.5 picks
   replay for simplicity. Is the cost of re-running the body acceptable
   for sampled traces? It should be.
3. **How much of `sys.monitoring` is enough** for strictness defense 3?
   Needs a spike.
4. **Guarding on values.** Should small enums be allowed as guard values
   (guard `tier == "premium"`), beyond booleans and lengths? The `eq`
   op plus a boolean guard in §4.6 already covers the common case.
5. **Errors of unawaited calls.** The rule in §1.4 is a first cut.
6. **Where do op code hashes come from:** the op's source only, or its
   transitive dependencies? Content-addressing (Unison-style) is the
   principled answer; source-plus-deploy-version is the practical one.
7. **Trace sampling policy and trace storage format,** including how much
   argument data to keep.
8. **Speculated calls share call paths.** A pure call speculated past a
   failing guard runs at its recorded `seq`, and eager replay may issue a
   different call at the same `seq`. Both run with the same call path.
   Effects are never speculated, so path-derived idempotency keys stay
   unique, but a leaf that uses its path for anything else would see it
   twice. Speculated calls could get a distinct path instead.

---

## 13. Design points to resolve

These came out of the effect-domain (§4.4) and pushdown (§1.5) designs.
Each one needs a decision before the feature it affects is built. The
v0 defaults are chosen so that none of them blocks M0–M2.

### 13.1 Domains whose failures don't fail the cell

**Question.** In §4.4, a failure in any domain still surfaces at its
`await` and fails the cell. Should a domain be able to declare that its
failures are handled by policy instead, and don't fail the cell? An
example is an audit or metrics write that is fire-and-forget, or retried
durably in the background.

**Considerations.**
- This would be a separate declaration from ordering. A domain says what
  is ordered; this says what happens when something fails.
- It interacts with `requires=completes` (NOTES §5). "Doesn't fail the
  cell" is only safe if something else guarantees the effect eventually
  happens, or if losing it is acceptable.
- Once the cell no longer waits for these effects, the caller can't
  observe their results. They could only be awaited for their value,
  which conflicts with fire-and-forget.

**v0 default.** Every failure fails the cell.

### 13.2 A fence across domains

**Question.** Is a `ctx.fence()` needed? It would wait for every effect
issued so far, in all domains, before any later effect is issued.

**Considerations.**
- A fence expresses "everything before this point has happened" without
  folding the effects back into `main`, for example before replying to a
  client or handing off to another system.
- Data dependencies and `main` may always be enough. A fence might only
  be a convenience for a pattern that could be written another way.
- In the graph, a fence would be a node with effect edges from all
  domains, and it would be a barrier for later effects.

**v0 default.** No fence. Revisit when the example apps need one.

If acquire/release orderings are adopted (§13.5), a fence is an
`acq_rel` effect that does nothing, and this point is resolved.

### 13.3 Checking that a domain declaration is correct

**Question.** A wrong domain declaration is a correctness bug, like a
wrong `pure`. It permits reorderings the application can't tolerate. How
can it be caught?

**Considerations.**
- Differential testing: run random linearizations of the partial order
  against fakes, and check invariants the application states. For
  example, "a receipt is never sent for a charge that failed".
- Fault injection: fail effects in one domain and check that the effects
  in other domains are still acceptable.
- Static lint: flag an effect in a non-`main` domain that follows an
  `await` of another domain's effect with no data dependency between
  them. That is where the author may have meant an ordering.
- The checks are only as good as the invariants the application states.
  This may argue for a way to declare invariants alongside cells.

**v0 default.** Domains are declared and recorded (the `checkout`
example uses them), but nothing reorders effects yet: the eager runtime
runs them in program order, which is always allowed. Reordering across
domains waits until there is a way to check the declarations.

### 13.4 How far pushdown goes

**Question.** §1.5 lets a caller pass unawaited handles, and makes
projection (`.field`, `[i]`) on a handle lazy. Should it go further?

- **Lazy `Ref[T]` parameters.** A callee declares a parameter as
  `Ref[T]` and receives the handle itself. It can start before the
  argument is ready, await it only on the paths that need it
  (call-by-need), or pass it on without materializing it.
- **Lazy ops.** Pure `@op`s applied to unawaited handles could produce
  handles too, instead of requiring an `await`.

**Considerations.**
- Lazy parameters change a cell's signature, so strictness becomes part
  of its interface. Vectorization has to handle arguments that are
  resolved at different times.
- A cell that awaits a `Ref` only on some paths gets data-dependent
  arguments. Is that a guard in its graph, or a new kind of edge?
- Lazy ops blur the line between eager and traced execution. Eager code
  would build small graphs of ops on handles. That may be a good
  thing (the eager and compiled models get closer), or a source of
  confusion.
- Errors in lazy values surface far from where they were created.
  EXPLAIN ANALYZE needs to show where a failed value came from.

**v0 default.** Handles may be passed as arguments and projected. Callees
are strict, and ops require values.

### 13.5 Acquire/release orderings

**Question.** Should effects carry a memory-model-style *ordering* in
addition to a domain? Domains are all-or-nothing: effects in the same
domain are fully ordered, and effects in different domains are not
ordered. Many real cases need ordering in one direction only.

**Proposal.** Each effect has an ordering: `relaxed` (the default),
`acquire`, `release` or `acq_rel`.

- **Within a domain:** program order, as in §4.4.
- **Across domains:** for effects E1 before E2 in program order, there is
  an effect edge E1 → E2 if E1 is `acquire` or E2 is `release`, and none
  otherwise.

So a **release** effect waits for every earlier effect, in every domain,
to complete successfully; later effects may still move ahead of it. An
**acquire** effect makes every later effect wait for it; earlier effects
may still move past it. `main` behaves as if its effects were `acq_rel`
with respect to each other.

```python
@cell(effects("locks"), order=ACQUIRE)
async def lock(ctx, key: str) -> Lease: ...

@cell(effects("email"), domain=UNIQUE, order=RELEASE)
async def notify(ctx, receipt: Receipt) -> None: ...
```

With `notify` as a release, a receipt is sent only after everything
before it succeeded, including the audit write in another domain. That
is the "reply to the client" or "commit" point. With `lock` as an
acquire, nothing after it runs before the lock is held, but an earlier
audit write may still move past it.

**Considerations.**
- Cross-domain edges wait for the earlier effect to *complete and
  succeed*. So a failed acquire blocks everything after it, and a release
  is blocked by any earlier failure. Path edges are unchanged.
- It makes domains easier to use safely (§13.3). An author can keep most
  effects relaxed and mark the few points that must see everything before
  them as releases, which is a smaller claim than "these domains are
  independent".
- It resolves the fence question (§13.2): a fence is an `acq_rel` no-op.
- Where it is declared: a default on the cell (`order=`), which a call
  site can override (`with ctx.order(RELEASE):`), like domains.
- **Across requests (later).** In memory models, acquire and release
  mainly synchronize *between* threads, by pairing on a location. The
  analogue is ordering across requests through storage cells: a release
  write returns a token, and an acquire read that presents the token sees
  everything before the release. That is causal consistency, and it
  connects to snapshot reads and isolation (NOTES §6–7). v0 would only
  order effects within one invocation.

**v0 default.** Not implemented. Nothing depends on it yet: the eager
runtime runs effects in program order, which satisfies every ordering.

