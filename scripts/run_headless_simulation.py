from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from introvertensemble import LibrarySimulation, LibraryWorld, load_layout


def main() -> None:
    spec = load_layout(ROOT / "assets" / "layouts" / "library_v1")
    world = LibraryWorld(spec)
    simulation = LibrarySimulation(world)

    print("Running headless library simulation")
    print("-" * 60)
    for _ in range(16):
        summary = simulation.step()
        zone_load = ", ".join(f"{zone}:{count}" for zone, count in sorted(summary.zone_load.items()))
        profile_counts = ", ".join(
            f"{profile}:{count}" for profile, count in sorted(simulation.profile_counts().items())
        ) or "none"
        print(
            f"step={summary.step_index:02d} hour={summary.hour:04.1f} "
            f"arrivals={summary.arrivals} departures={summary.departures} reseats={summary.reseats} "
            f"occupancy={summary.occupancy:02d}"
        )
        print(f"  profiles: {profile_counts}")
        print(f"  zone_load: {zone_load}")


if __name__ == "__main__":
    main()
