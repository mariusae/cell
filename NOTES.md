# Cells: brainstorm notes

These notes collect a design conversation, grouped by theme. They record
ideas, arguments and open problems; they are not a specification. The
graph and tracing design that comes out of them is in [DESIGN.md](DESIGN.md).

---

## 1. The core idea

A system is split into **cells**: functions that take arguments and call
other cells. A cell never calls another cell directly. Every call goes
through a **context** (`ctx`), which decides how the call is carried out.

```python
@cell
async def my_func(ctx, arg):
    sub_result = await some_other_cell(ctx, arg)
    return process_sub_result(sub_result)
```

Business logic (what the code computes) is separated from execution
policy (how it runs): scheduling, caching, durability, retries, version
rollout, consistency, placement.

**Framing.** This is effect handlers for distributed calls. The handler
is installed by whoever operates the system, not by the caller. The SQL
analogy goes deep:

- Cells, together with their declared semantics, are the **logical plan**.
- The chosen execution mechanisms are the **physical plan**.
- A planner picks the physical plan; tooling (EXPLAIN) makes it visible.

**Summary.** Cells declare what they are; policy decides how they run; a
checker makes sure the how is legal for the what.

**Prior art, each covering part of this:**
- Finagle: services and filters.
- Haxl / DataLoader: batching discovered at run time.
- Temporal / Restate / durable functions: journaled replay.
- Orleans: grains, affinity.
- Service meshes: policy outside the code, but it can't see what calls mean.
- Dapr.
- Unison: content-addressed code.
- Reflow / Bazel: incremental computation.
- Algebraic effects.
- Pathways, TensorFlow 1 distributed graphs, choreographic programming:
  distributed dataflow.
- Physical design advisors in databases (e.g. AutoAdmin).

None of them makes this separation the single organizing idea.

---

## 2. Why policy isn't orthogonal, and why that's fine

Most policies are only legal if the cell has certain properties:

| Policy | Requires of the cell |
|---|---|
| Cache keyed by args | Deterministic within a time window, no side effects |
| Retry freely | Idempotent |
| Durable queue | Serializable args, caller can wait for a result much later, at-least-once delivery acceptable |
| Hedged requests | Safe to run concurrently twice |
| Vectorizing | Has a declared vector form |
| Strong consistency | Known read/write sets |

Retrying `charge_card` double-charges. Caching `get_balance` serves stale
balances. The operator can't know these properties from outside; the
author has to declare them, and the system checks every policy against
the declarations.

This is the real payoff rather than a problem. Declarations tell an
optimizer which rewrites are safe, the way relational algebra supplies
the equivalences a SQL optimizer searches. Separation of concerns becomes
three roles:

- **Cell authors:** logic, plus declared semantics.
- **Operators:** objectives (SLOs, cost, staleness, durability per
  traffic class).
- **Planner:** the physical plan. Hand-written policy shrinks to *hints*,
  like SQL hints.

---

## 3. Cell semantics vocabulary

A candidate minimal set of effects:

- `pure`
- `reads(R)`
- `writes(W)`
- `idempotent_by(key)`
- `external`: opaque effects; the conservative default.

A set of semantic classes that are closed under composition, from the
saga / flexible-transaction literature:

| Class | Meaning |
|---|---|
| Pure | No effects |
| Retriable | Idempotent, and succeeds eventually if retried |
| Compensable | Declares an inverse cell |
| Pivot | Can't be undone, and may fail |

Composition rules the framework can infer from a composite body:

- A sequence of compensables is compensable: undo them in reverse order.
- A sequence of retriables is retriable, provided the body is deterministic.
- Compensables, then one pivot, then retriables is a **well-formed saga**:
  semantically atomic, and a pivot as a whole.
- Anything else (two pivots; a non-retriable after the pivot) is rejected
  when the plan is checked.

The author declares an inverse on each leaf. The framework assembles the
saga; there is no compensation logic in the composite.

**Vector forms** also belong in the semantic vocabulary (see §9).

**Declarations can be wrong**, so check them at run time:

- Re-run a sample of `pure` calls and compare results.
- Shadow-compare cached results against fresh ones.
- Have storage enforce `writes(W)` as a capability.
- Use replay to test `idempotent_by`.
- Property-test that `inverse ∘ step` leaves state unchanged.
- Check sampled vector calls against their scalar form.

---

## 4. The context owns all nondeterminism

Durability, replay, deterministic retries, deopt and simulation all
require the cell body to be deterministic given its inputs and call
results. So `ctx` has to provide time, randomness, config and all I/O to
state. This constrains how code is written, and in return makes the
following almost free:

- Record/replay of production requests.
- Deterministic simulation with injected faults (FoundationDB-style).
- Journaled durable execution (Temporal-style).
- Idempotency keys derived from call paths (§7).
- Deopt from compiled plans back to eager execution (DESIGN.md).

---

## 5. Requirements vs. mechanisms

- **Semantics:** what the cell *is* (author).
- **Requirements:** what the business *needs* from a call, e.g.
  `requires=completes` or `atomic` (declared in the logic).
- **Mechanism:** how the need is met (chosen by the planner).

Choosing a mechanism from semantics and requirement:

| Semantics | Requirement | Mechanism |
|---|---|---|
| Idempotent | Best effort | Inline, retry from budget |
| Idempotent | Completes | Durable queue, at least once |
| Deterministic composite | Completes | Journaled replay |
| Well-formed saga | Atomic (semantically) | Journaled replay plus automatic compensation |
| Storage cells on one shard | Atomic | Local transaction |
| Storage cells across shards | Atomic + isolated | 2PC or deterministic ordering (Calvin-style) |

The same `atomic` requirement can be met by different mechanisms,
depending on where the data lives. After a reshard, the planner can
downgrade a saga to a local transaction.

**Sync vs. async can't be fully hidden.** An inline call and a six-hour
durable one look the same at an `await`, but latency is part of the
contract. `requires=completes` should change the call's type: the caller
gets a handle (poll it, register a continuation cell, or subscribe to an
event) instead of a value.

---

## 6. Global budgets

Centralizing policy makes properties of the whole graph enforceable:
total work, total deadline, total staleness. `ctx` carries the budgets,
and each call spends from them.

Each budget type needs three operations: **split** (divide among
children), **consume**, and **combine** (merge the children back).

| Budget | Split | Combine |
|---|---|---|
| Deadline | Pass the absolute time down | min |
| Work / retries | Divide or share a pool | sum |
| Cost ($) | Like work | sum |
| Staleness | Carry an "as-of" time on results | min of input times |
| Success probability | — | product |

What this enables:

- **Retries stop multiplying.** Three retries at each of four levels no
  longer means 81 calls; every level spends from the same pool.
- **Retries happen in one place.** The planner picks the level, usually
  the lowest idempotent cell nearest the failure.
- **Admission control** uses the plan's predicted cost: reject at the
  edge rather than time out after doing 80% of the work.
- **Load is shed a whole request at a time,** using priority carried in
  ctx.
- **Cancellation cancels the whole subtree:** deadlines, callers that
  gave up, losing hedges.
- **Snapshot reads for a request.** One read timestamp for the whole
  request (Spanner-style) gives consistent reads across storage cells.
- **Fleet-wide retry ratio** (Finagle's `RetryBudget`, applied to the
  whole graph). This is the setting that prevents metastable failures.

Hard parts:

- **Splitting work budgets across machines.** A shared pool is exact but
  expensive; pre-splitting is cheap but can starve one branch while its
  siblings have budget to spare. Middle ground: credit-based flow control.
  The planner pre-splits from profiles, children return unused budget,
  and a child can ask its parent for more.
- **Attribution.** A leaf fails because of spending in another branch.
  EXPLAIN has to show where the budget went.
- **Fairness between branches.** Split ratios are a fairness policy.

---

## 7. Reliable execution

- **Idempotency keys derived from call paths.** Since ctx controls
  nondeterminism, the position of a call in the tree is deterministic,
  e.g. `request_id / checkout / call#2`. It serves as the idempotency key
  for effectful calls. After a crash, replay of a journaled cell
  deduplicates its children, and nobody threads keys by hand.
- **Durable execution** means journaling each cell's calls and replaying
  deterministically. It is the same machinery as deopt in DESIGN.md.
- **Isolation** is the gap sagas leave: other requests see intermediate
  states. Handle it with semantic locks declared by storage cells, or
  accept it under per-request snapshot reads.
- **Versioning of journals.** A journal written under v1 may be replayed
  under v2. Either pin the version for the life of the journal
  (content-addressed code helps), or version the code paths explicitly.
- **Consistency belongs to state, not calls.** Make state access a cell
  (storage cells). Transactions across cells are an explicit construct,
  not a policy setting.

---

## 8. Capabilities and bindings

**ctx is a capability.** A cell can call only what its context grants,
which gives least privilege and an audit trail by construction. Use typed
references (the imported cell object), not strings, so the call graph can
be extracted statically.

**Bindings.** A cell can install inner handlers for its subtree, e.g.
"route calls to x, y and z to this instance". This brings in application
knowledge the planner can't infer: warm in-process caches, affinity,
shard ownership, request-scoped memoization, test doubles.

- **Strength.** A binding is a *fact* ("I hold warm state for users 1–N")
  that the planner weighs, a *preference* (use this if possible, else
  fall back), or a *mandate* (route here or fail). Default to
  preferences; encourage facts.
- **Bindings only narrow.** They attenuate a capability and never widen
  it, and the target must satisfy the bound cell's declared semantics.
- **Bindings are checked against requirements.** For a
  `requires=completes` call, an in-process binding has to be wrapped
  (journal the call). Durable cells accept only preferences, and replay
  falls back if the instance is gone.
- **Scope.** A lexical binding covers this cell's direct calls; a dynamic
  one covers the whole subtree, across hops. Dynamic is where the
  locality wins are. Bindings travel in ctx with a lease and show their
  origin in EXPLAIN.
- **Liveness.** When a lease expires, preferences fall back and mandates
  fail.
- **The callee side.** An instance can *advertise* "I'm a good target for
  x(k)", and the planner routes by affinity.

---

## 9. Vectorization

**Batching and vectorizing are different things:**

- *Batching* is a transport optimization: many calls travel in one
  message, but each still runs separately. It needs no declaration.
- *Vectorizing* is semantic: the cell has a real `[A] → [B]`
  implementation whose cost grows sub-linearly (a `WHERE id IN (…)`
  query, a GPU batch). It has to be declared.

```python
@cell(pure)
async def get_user(ctx, uid: int) -> User: ...

@get_user.vectorized
async def get_users(ctx, uids: list[int]) -> list[User | Error]: ...
```

What the declaration promises:

- **Element-wise equivalence:** `vec(xs)[i] == scalar(xs[i])`.
- **Per-element failures.**
- **A cost curve `cost(n)`,** learned from telemetry, which the planner
  uses to choose window sizes.

**Each element carries its own context:**

- **Capability.** The call must be authorized for every element; group
  by tenant or security domain if the callee can't check per element.
- **Deadline.** The vector's deadline is the earliest among its
  elements. Elements whose deadlines are too tight run alone.
- **Budget, results, retries, durability:** all per element.
- **Trace identity.** EXPLAIN ANALYZE shows which vector a call joined.

**Where N comes from:**

- Fan-out within one request (`ctx.map`).
- **Across independent requests.** This is the bigger win, analogous to
  continuous batching in LLM serving (Orca/vLLM). With iteration-level
  scheduling, new requests can join a multi-step composite at any step.

**Vectorizing composites (`vmap` for services).** A composite whose
leaves are vectorized can itself be vectorized, if its elements don't
interact:

- *At run time (Haxl-style):* group pending calls by callee each round.
  Branching is handled for free, but alignment across rounds is poor.
- *Compiled from traces (vmap-style):* at a branch, split the vector by
  the branch condition and merge afterwards. Vector widths are
  predictable, and the composite gets a declared vector form the planner
  can use one level up.

---

## 10. Tracing

Recover the logical plan by tracing, as PyTorch and JAX do. Engineers
are already used to reasoning about it. Use static extraction for the
safety checks. Optimize what does happen; check against everything that
could.

Lessons carried over from ML frameworks:

- **Eager is always correct.** A compiled plan is a speculative speed-up;
  when in doubt, fall back to eager.
- **Guards and specialization.** Record what the trace assumed (branch
  taken, versions resolved, binding facts). A failing guard means deopt,
  and possibly a new specialization.
- **Graph breaks.** Code the tracer can't see through splits the trace.
  "Too many graph breaks" is a useful diagnostic for engineers.
- **Concrete vs. symbolic.** Concrete tracing (like Dynamo) finds the
  graphs that actually run, so it drives optimization. Symbolic
  extraction (like FX) over-approximates, so it drives the safety checks.

How it differs from ML tracing:

- **Traces come from production and form a distribution.** A service has
  many data-dependent paths, so compile the hot ones, profile-guided, and
  leave the tail eager.
- **The world moves under a plan.** Deploys, reshards and hit rates
  change, so plans need invalidation on cost-model drift, not only on
  guard failure. Recompile continuously and roll out gradually.
- **One mechanism, many uses.** The same traces feed EXPLAIN ANALYZE,
  record/replay and counterfactual replay.

---

## 11. Dataflow execution: vectorizing the whole graph

Haxl finds batching opportunities at run time, inside an orchestration
model: the parent calls, waits, then makes its next call. Instead, the
**root** (e.g. a frontend) collects similar requests, vmaps the whole
compiled plan, and **launches** it. Each stage runs a vector operation
and sends its results directly to the next stage. Serving becomes a
dataflow system.

Compiler passes:

1. **Trace:** recover the dataflow graph, including the glue code between
   awaits.
2. **vmap:** widen the whole graph. Width is chosen once at the root, so
   Haxl's alignment problem disappears.
3. **Partition:** split by location and insert send/receive edges.
   TensorFlow 1 did this; it is also endpoint projection from
   choreographic programming.
4. **Place:** assign stages and glue to locations; ship code to the data.

What this gains:

- **Fewer round trips:** A→B→C→A instead of A→B→A→C→A.
- **Wide vectors everywhere.**
- **Pipelining** between stages.
- **Optimization over the whole graph.**

Closest precedent: Pathways (one controller dispatching compiled, sharded
graphs).

Hard parts:

- **Branching** splits vectors. Merge sub-vectors from several roots at
  popular stages.
- **Deopt per element.** An element that fails a guard leaves the vector
  and continues eagerly; the rest continue.
- **Stragglers.** A stage can split a vector on timeout, driven by
  per-element deadlines.
- **Blast radius.** One stage failure hits N requests. Use per-element,
  per-stage retries and durability.
- **Low load.** The collection window adds latency, so shrink width to 1.
  The compiled plan still saves round trips.
- **Code shipping.** Glue that runs downstream needs a pinned code bundle,
  or portable modules (WASM).
- **Backpressure** between stages uses credits.

---

## 12. A joint controller: capacity, placement and plans

Today the autoscaler, the scheduler and the application's hard-coded
choices run as independent loops that interfere with each other.
Example: a cache is added, the autoscaler scales down, the cache goes
cold after a deploy, and the service falls over.

**Derived demand.** The load on each cell is the edge request mix
multiplied through the call graph under the current plan. The planner can
therefore *predict* per-cell load for a candidate plan before rolling it
out.

**Decision variables:** plans per endpoint and traffic class, placement,
replica counts, cache sizes, vector windows, budget splits, and whether
to materialize a result or compute it on read.

**Constraints:** SLOs, semantic legality, capacity, failure domains.

**Objective:** cost per unit of goodput.

**Example trades:** a cache vs. more replicas; colocating cells vs.
scaling them independently; materializing a result vs. computing it on
read (from the read/write ratio); wider vector windows under load.

**One shared model, loops at different speeds** (SDN traffic engineering
as in B4/SWAN: a central allocator, plus fast local reaction):

| Loop | Timescale | Decides |
|---|---|---|
| Per request | µs–ms | Credits, admission, routing, deopt |
| Budgets | Seconds | Split ratios, retry rates, shedding |
| Plans | Minutes | Recompiles, cache and vector settings |
| Placement / capacity | Minutes–hours | Replicas, colocation, materializations |
| Procurement | Weeks | Hardware |

**Risks:**

- **Correlated mistakes.** Limit change per step, roll out gradually, and
  validate candidate plans with counterfactual replay.
- **Static stability.** If the controller is down, the data plane keeps
  its last plan.
- **Oscillation.** Separate the timescales and add hysteresis.
- **Explaining its decisions.**

The SQL analogy completed: query planner plus physical design advisor
plus resource manager, sharing one model of the workload.

---

## 13. Tooling

Every production system already has a physical plan, but it is implicit
and scattered: client-library retries, hard-coded caches, mesh YAML,
feature flags. Making it explicit is **strictly better**, provided
correctness stays local:

- semantics and requirements are written in the code,
- the planner applies only legal rewrites, and
- declarations are verified at run time.

The code then tells you what a call means and guarantees; tooling tells
you how it runs. Observable behavior (latency, which errors surface,
staleness) still depends on the plan, within the declared bounds.

- **EXPLAIN:** the static plan for an endpoint, and *why* each choice was
  made.
- **EXPLAIN ANALYZE:** one real request's execution tree, including where
  its budget went, which vectors it joined, and compensations that ran.
- **Plan diffs in code review.**
- **Counterfactual replay:** recorded requests under another plan or code
  version.
- **In the editor:** hover over an `await` to see the resolved plan and
  its production stats.
- **Why-not:** "why isn't `get_user` cached?"
- **Time travel:** plans are versioned and immutable.

---

## 14. Other ideas

- **Feature flags, canaries and shadow traffic** as rebinding for a
  request cohort: tee to v1 and v2 and diff the results.
- **Per-request debug overlays:** route one request's calls through a
  recorder, or inject WASM instrumentation.
- **Chaos as a policy,** scoped to cells, versions or cohorts.
- **Differential testing across versions,** cell by cell, on recorded
  inputs.
- **Invalidation driven by dependencies.** A result that is
  `pure-given-reads(R)` is invalidated precisely when R changes. This
  makes caching a safe default rather than an opt-in.
- **Materialized views:** the planner moves work from read time to write
  time when the read/write ratio warrants it.
- **Cost and latency attribution** per cell, for free.
- **Migration:** wrap existing services as leaf cells with `external`
  effects and break them apart over time.
- **Policy is code,** not one giant YAML file: typed, tested, reviewed.
  Layered ownership: authors declare semantics, owners set defaults, the
  platform enforces global invariants.

---

## 15. Open questions

1. What is the minimal effect vocabulary that is still useful?
2. Is a durable cell a separate *kind* of cell (with its own type and the
   replay constraints), or a policy on any cell? We lean toward a separate
   kind.
3. How much is resolved statically and how much per invocation? Likely a
   default compiled plan plus request-scoped overlays.
4. How does a caller see its callee's mechanism, e.g. a handle versus a
   value?
5. How do we avoid building a distributed Haskell runtime nobody can
   operate? Decide early which parts stay boring.
6. How is isolation exposed for sagas: semantic locks, snapshots, or
   both?
7. How are journals and durable work versioned across deploys?
8. Where does trace data live, and how much argument data is retained
   (privacy, cost)?

---

## 16. Roadmap direction

Start local:

1. Graph representation, tracer, and correct graph breaks (DESIGN.md).
2. Example applications run locally, with simulated service latency.
3. Try the ideas on them: planner passes, budgets, vmap over the whole
   graph.
4. Only then, distributed execution.
