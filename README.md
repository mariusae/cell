# cell

Cells separate what a system computes from how it runs. See
[NOTES.md](NOTES.md) for the ideas and [DESIGN.md](DESIGN.md) for the design.

## What is Cell?

Cell is an exploration of a different approach to building and running online serving systems, bringing along ideas from dataflow programming, machine learning frameworks like PyTorch, and database query planners. Traditionally, such systems comprise many different kinds of servers, all communicating via a service mesh. These architectures tend to grow in complexity to meet various needs, and ultimately end up hard-coding many scaling and performance optimizations, like caching, batching, load balancing, admission control, etc. This is because, in this architecture, each server must make local decisions about what to do: where to send a request, whether to cache a response, how concurrent requests can be batched, and so on.

What's more: such systems get more unwieldy over time. The very fact that a user request touches 10s-100s of different kinds of servers, and often many replicas of each, makes it difficult to think of "whole program" optimizations. You can also argue it is indicative of poor separation of concerns: it is hard to disentangle the "business logic" -- what the system should do -- from its "execution plan" -- how to go about it.

Cell represents such serving systems as explicit dataflow graphs whose nodes represent both pure computation, as well as i/o and other effects, like database queries, view materialization, etc. Once you are operating with this representation, all of the above concerns become graph rewrites: we can insert caching nodes where results are profitably (and safely!) cached; we can insert vectorization operations explicitly; by having a view of the whole graph, we can perform admission control at the point of entry, maximizing our overall goodput. Because we have an explicit graph, we can also implement fault handling centrally: retry policies, and even specify how a system should ("gracefully") degrade in response to overload.

In the prototype implementation, the graph is executed by a single node, but it is designed for fully distributed execution: a policy might be applied at the request origin (e.g., at the web server where a user's request lands), and then executed without further coordination.

You can view this approach as being analogous to a database's query planner. Whereas a database encodes the "application" in a query, the Cell application is defined in pure Python; a database uses table statistics to make choices, the Cell runtime might use recent latency and failure data. On it goes.

In order to provide good ergonomics -- explicit graph representations can be illegible -- Cell provides an imperative programming model, acquiring a graph through *tracing*. This is exactly how the PyTorch compiler works: the program is guaranteed to behave-as the original Python program executed eagerly, but in practice the system extracts the corresponding dataflow graph through tracing. This does put some limitations on the model -- for example, any source of non-determinism has to be legible to the framework itself -- but makes the program read as a normal Python program would.

## Status

Currently implemented: milestones M0 (the eager runtime and journals), M1
(the graph IR and the tracer), M2 (compiled execution, deopt and
validation), M3 (static extraction, lint, and strict tracing), M4 (graph
passes, caching, and latency measurement) and M5 (vector forms, ctx.map,
and batches of requests).

## Running it

```
uv sync
uv run pytest                    # tests, including every example scenario
uv run python -m examples        # run the example scenarios and print their journals
uv run python -m examples home   # only scenarios whose name contains "home"
uv run python -m examples --graphs home   # the graphs traced from them
CELL_UPDATE_GOLDEN=1 uv run pytest        # rewrite tests/golden after a deliberate change
uv run python -m cell.static examples.features   # lint the cells in a module
uv run python -m examples --bench          # latency: eager, compiled, cached (simulated time)
uv run python -m demo                      # the demo server, at http://127.0.0.1:5050
```

## Layout


- `src/cell/`: the runtime. `core.py` (cells and ops), `context.py` (ctx
  and handles), `runtime.py` (eager execution, replay), `journal.py`,
  `data.py` (data values as pytrees), `semantics.py`, `graph.py` (the IR),
  `trace.py` (the tracer), `monitor.py` (strict tracing), `compiled.py`
  (running graphs), `validate.py`, `static.py` (call graphs and lint),
  `passes.py` (inline, fold, dedup), `mapping.py` (ctx.map).
- `examples/`: example programs and their scenarios. `harness.py` has the
  fake services (`World`) and the `Scenario` type; `simtime.py` an event
  loop with simulated time; `bench.py` the latency benchmark.
- `demo/`: a Flask server that walks each scenario through the system: its
  source, an eager run, the traced graph, the graph after the rewrites you
  select (as IR and as a Mermaid flowchart), a compiled run, a batch of
  requests sharing vector calls, a load simulation with latency
  distributions and throughput, and latency/throughput curves under load.
- `tests/`: `golden/` holds the traced graph of every example scenario.
