from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from introvertensemble import LibraryWorld, load_layout
from introvertensemble.viewer import LibraryViewer


def seed_demo_occupancy(world: LibraryWorld, seed: int = 7) -> None:
    rng = random.Random(seed)
    target_counts = {
        "z_quiet_room_1": 5,
        "z_open_reading_1": 6,
        "z_entry_lounge_1": 3,
        "z_high_turnover_1": 2,
        "z_discussion_room_1": 4,
        "z_cafe_pantry_1": 2,
    }
    agent_index = 1
    for zone_id, count in target_counts.items():
        seats = world.available_seats(zone_id)
        rng.shuffle(seats)
        for seat in seats[:count]:
            world.occupy_seat(seat.id, f"demo_agent_{agent_index}")
            agent_index += 1


def main() -> None:
    layout_dir = ROOT / "assets" / "layouts" / "library_v1"
    spec = load_layout(layout_dir)
    world = LibraryWorld(spec)
    seed_demo_occupancy(world)
    viewer = LibraryViewer(world)
    viewer.run()


if __name__ == "__main__":
    main()
