from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from introvertensemble import LibrarySimulation, LibraryWorld, load_layout
from introvertensemble.simulation import SimulationConfig
from introvertensemble.viewer import LibraryViewer


def main() -> None:
    layout_dir = ROOT / "assets" / "layouts" / "library_v1"
    spec = load_layout(layout_dir)
    world = LibraryWorld(spec)
    simulation = LibrarySimulation(
        world,
        config=SimulationConfig(
            focal_agent_enabled=True,
            focal_agent_initial_seat_history=("QR-SC-05", "QR-SC-05", "QR-SC-04"),
            focal_agent_never_departs=True,
            events_enabled=True,
        ),
        seed=19,
    )
    viewer = LibraryViewer(world, simulation=simulation)
    viewer.run()


if __name__ == "__main__":
    main()
