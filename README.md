# cell

Cells separate what a system computes from how it runs. See
[NOTES.md](NOTES.md) for the ideas and [DESIGN.md](DESIGN.md) for the design.

Currently implemented: milestones M0 (the eager runtime and journals), M1
(the graph IR and the tracer) and M2 (compiled execution, deopt and
validation).

```
uv sync
uv run pytest                    # tests, including every example scenario
uv run python -m examples        # run the example scenarios and print their journals
uv run python -m examples home   # only scenarios whose name contains "home"
uv run python -m examples --graphs home   # the graphs traced from them
CELL_UPDATE_GOLDEN=1 uv run pytest        # rewrite tests/golden after a deliberate change
```

Layout:

- `src/cell/`: the runtime. `core.py` (cells and ops), `context.py` (ctx
  and handles), `runtime.py` (eager execution, replay), `journal.py`,
  `data.py` (data values as pytrees), `semantics.py`, `graph.py` (the IR),
  `trace.py` (the tracer), `compiled.py` (running graphs), `validate.py`.
- `examples/`: example programs and their scenarios. `harness.py` has the
  fake services (`World`) and the `Scenario` type.
- `tests/`: `golden/` holds the traced graph of every example scenario.
