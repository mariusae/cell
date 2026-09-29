# cell

Cells separate what a system computes from how it runs. See
[NOTES.md](NOTES.md) for the ideas and [DESIGN.md](DESIGN.md) for the design.

Currently implemented: milestone M0, the eager runtime and journals.

```
uv sync
uv run pytest                    # tests, including every example scenario
uv run python -m examples        # run the example scenarios and print their journals
uv run python -m examples home   # only scenarios whose name contains "home"
```

Layout:

- `src/cell/`: the runtime. `core.py` (cells and ops), `context.py` (ctx
  and handles), `runtime.py` (eager execution, replay), `journal.py`,
  `data.py` (data values as pytrees), `semantics.py`.
- `examples/`: example programs and their scenarios. `harness.py` has the
  fake services (`World`) and the `Scenario` type.
- `tests/`
