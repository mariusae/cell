"""Run the example scenarios and print their journals.

    uv run python -m examples            # all scenarios
    uv run python -m examples home       # scenarios whose name contains "home"
"""

import asyncio
import sys

from . import SCENARIOS


async def main(pattern: str) -> None:
    for s in SCENARIOS:
        if pattern not in s.name:
            continue
        run, world = await s.run()
        print(f"== {s.name}")
        print(run.journal.format())
        for name, args in world.effects_in_path_order():
            print(f"   effect {name} {args}")
        print()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else ""))
