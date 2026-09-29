"""Run the example scenarios and print their journals, or their traced graphs.

    uv run python -m examples                  # all scenarios
    uv run python -m examples home             # scenarios whose name contains "home"
    uv run python -m examples --graphs home    # the graphs traced from them
"""

import asyncio
import sys

from . import SCENARIOS


async def main(pattern: str, graphs: bool) -> None:
    for s in SCENARIOS:
        if pattern not in s.name:
            continue
        print(f"== {s.name}")
        if graphs:
            traced, _ = await s.trace()
            print(traced.graph.format())
        else:
            run, world = await s.run()
            print(run.journal.format())
            for name, args in world.effects_in_path_order():
                print(f"   effect {name} {args}")
        print()


if __name__ == "__main__":
    args = sys.argv[1:]
    graphs = "--graphs" in args
    args = [a for a in args if a != "--graphs"]
    asyncio.run(main(args[0] if args else "", graphs))
