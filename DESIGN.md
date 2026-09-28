# Cells: graph and tracing design

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

Plain helpers, sync or async, that take ctx and call cells are fine. The
tracer sees through them. **Cell boundaries mark where policy applies,
not how code is organized.**

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
> with each other. Compiled may additionally issue *pure* calls
> speculatively.

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

---

## 4. Graph IR

### 4.1 Shape

One graph per specialization of a cell. SSA form: every node defines at
most one value, and nodes are stored in a topological order that follows
program order.

```
Graph
  cell:      CellRef             # cell id + code hash
  params:    [ValueId]           # one per cell parameter (excluding ctx)
  nodes:     [Node]              # topological, program order
  result:    ValueId
  guards:    [NodeId]            # index of guard/deopt nodes
  meta:      trace ids, counts, source file hashes

Node
  id:        NodeId              # also its ValueId, if it produces a value
  kind:      param | const | call | op | pack | source | guard | deopt | return
  inputs:    [ValueId]           # data dependencies
  ctrl:      [NodeId]            # control dependencies (§4.3)
  attrs:     kind-specific
  site:      file:line           # for EXPLAIN and diagnostics
```

### 4.2 Node kinds

| Kind | Inputs | Attrs | Meaning |
|---|---|---|---|
| `param` | — | index, type | A cell input |
| `const` | — | value (data) | A literal or a captured constant |
| `call` | args (one per parameter) | cell ref, semantics (pure/effectful), `seq` | Issue a call to a cell |
| `op` | args | op ref (name + code hash), or a builtin (§5.2) | Pure local computation |
| `pack` | leaves | treedef (structure) | Build a data value from its leaves |
| `source` | — | `now` \| `random` \| `config(key)` | Nondeterminism through ctx; journaled |
| `guard` | pred value | expected value | Assert pred == expected, otherwise deopt |
| `deopt` | — | reason | Unconditional deopt: a graph break (§6.2) |
| `return` | value | — | Result of the cell |

That's the whole v0 IR. Later additions (§10): `map` (structured fan-out
with a subgraph), `switch`/`merge` (trace trees), `send`/`recv`
(partitioning), and a vector annotation on every value.

### 4.3 Dependencies and ordering

Data edges alone would allow any reordering. That is fine for pure nodes
and wrong for effectful ones. The rules:

- **Pure nodes** (`const`, `op`, `pack`, pure `call`, `param`) are ordered
  only by data edges. They may run earlier than in eager, including
  speculatively before a guard that precedes them in program order. The
  cost of a failed speculation is only wasted work.
- **Effectful calls** get control edges to every node that was *observed*
  before the call was issued, in program order. A node is observed when
  user code consumes its value:
  - a call is observed when awaited,
  - an op when it evaluates,
  - a guard when it is checked.

  Eager execution only reaches an effectful call if all of those
  succeeded and passed. The control edges make compiled execution
  require the same.
- **Effectful calls are totally ordered with each other** in program
  order (a control edge to the previous effectful call's *issue*). v0 is
  conservative here. Declared `reads`/`writes` sets let a later pass drop
  edges between calls that commute.
- **`source` nodes** are journaled, so they are ordered only by data.

Calls issued but not yet awaited before an effectful call are *not* its
control dependencies, because eager doesn't wait for them either.

### 4.4 Identity and serialization

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

### 4.5 Example

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
graph home #3f2a9c  (uid: int) -> Page
  %0  = param 0 : int
  %1  = call get_user(%0)            pure        seq=0   home.py:14
  %2  = op getattr(%1, "id")                             home.py:15
  %3  = call get_prefs(%2)           effectful   seq=1   home.py:15  ctrl=[%1]
  %4  = call get_items(%0)           pure        seq=2   home.py:16
  %5  = op getattr(%1, "tier")                           home.py:17
  %6  = op eq(%5, "premium")                             home.py:17
  %7  = guard %6 == True                                 home.py:17
  %8  = call rank(%4, %3)            pure        seq=3   home.py:18
  %9  = op greeting(%1)                                  home.py:21
  %10 = pack Page(title=%9, items=%8)                    home.py:21
  return %10
```

Things to notice:

- **`get_items` doesn't depend on `get_user`.** Program order serialized
  them; the graph doesn't, so compiled mode runs them in parallel. This is
  the first optimization tracing gives us for free.
- **`get_prefs` has `ctrl=[%1]`.** Eager only reaches it after
  `await get_user` has succeeded.
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
| Pass to an `@op` | `op` node | The op runs on unwrapped concrete values; its result is wrapped |
| `.field` on data | `op getattr` | Allowed only on data types (§1.2), so it is known to be pure |
| `[i]`, `[k]` | `op getitem` | |
| `+ - * / // % == != < <= > >= & \| ^ ~ -x` | `op <operator>` | Comparison results are tracers too |
| Pure methods on immutable builtins (`str.lower`, `tuple.index`, …) | `op method` | From a fixed allowlist |
| `bool(t)`, `if t:`, `and`/`or`/`not` | `guard` on truthiness | Concretizes the value (§5.4) |
| `len(t)` | `op len` + `guard` on the result | Python requires `len` to return an int |
| `iter(t)`, `for x in t` | `guard` on length, then one `getitem` per element | Unrolled. Use `ctx.map` for data-dependent fan-out (§10) |
| `int(t)`, `float(t)`, `hash(t)`, `str(t)`, `format(t)`, … | **graph break** | A guard on the exact value would almost never hold, so break instead |
| Mutation (`setattr`, `setitem`, `append`, …) | **graph break** | Data is immutable |
| Passing to any other callable | **graph break** (§5.4) | |

### 5.3 Pytrees

When a structure containing tracers is passed to a call or an op, or
returned from the cell, the tracer:

1. flattens it into leaves and a treedef (the structure),
2. turns concrete leaves into `const` nodes, and
3. emits a `pack` node.

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
   booleans and lengths; anything else breaks.
2. **Tracers are opaque.** A tracer is not a subclass of the wrapped type
   and does not spoof `__class__`. Code that checks
   `isinstance(t, dict)` sees False; it will usually fail fast rather than
   take a silently wrong path.
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

### 5.5 Breaks during tracing

When the tracer hits a break at some point P in the trace:

1. **Stop recording.** Everything recorded before P stays in the graph.
   Append a `deopt` node with the reason and source location.
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
- **User code catching an error during tracing** (a `try/except` around a
  failing call) means the trace saw error handling. That's a graph break
  at the `await`.
- **Errors raised by ops** in compiled mode also deopt. Replay raises the
  same error from the op, because ops are deterministic.

### 5.7 Nested cells and inlining

Each cell is traced separately and gets its own graphs. A `call` to a
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

A `guard` checks that a predicate matches the value observed during
tracing. In v0 the predicate is either the truthiness of a value or a
length. It runs as soon as its input is ready, which is often before the
Python program would have reached it.

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
was recorded only on the traced path, and the guards pin that path. They
run in an order consistent with eager's, because of the control edges of
§4.3.

**Claim (with deopt).** Every effectful call issued before the failure
has control edges to all nodes observed before it in program order, and
all of those passed. So eager would also have issued it, with the same
arguments (data edges) and at the same `seq`. Replay therefore hits it in
the journal and does not re-execute it. From then on, execution is plain
eager. So the combined execution is exactly eager's, apart from the pure
calls that were speculated.

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

**Differential validator:**

- Take recorded requests (inputs plus journaled call results) and run each
  one in eager mode and in compiled mode against the recorded results.
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
  - `const`: broadcast.
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
  own graph (endpoint projection).
- **Planner rewrites** as graph-to-graph passes: dedup of identical pure
  calls, caching (wrap a pure call), dropping ordering edges between
  commuting effectful calls, inlining.
- **EXPLAIN** is the printer in §4.5, plus annotations. **EXPLAIN
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

1. **Home page:** the §4.5 example. Fan-out, a branch, parallelism
   recovered from program order.
2. **Feed with ranking:** a fan-out over items (`ctx.map`), a vectorized
   feature lookup and ranker. Cross-request vectorization.
3. **Checkout:** effectful calls (reserve, charge, confirm). Exercises
   ordering, control edges, and deopt correctness in the presence of
   effects. Later, sagas.

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
   op plus a boolean guard in §4.5 already covers the common case.
5. **Errors of unawaited calls.** The rule in §1.4 is a first cut.
6. **Where do op code hashes come from:** the op's source only, or its
   transitive dependencies? Content-addressing (Unison-style) is the
   principled answer; source-plus-deploy-version is the practical one.
7. **Trace sampling policy and trace storage format,** including how much
   argument data to keep.
