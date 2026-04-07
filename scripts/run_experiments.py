from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from introvertensemble import LibrarySimulation, LibraryWorld, load_layout
from introvertensemble.metrics import run_episode
from introvertensemble.simulation import SimulationConfig


def main() -> None:
    spec = load_layout(ROOT / "assets" / "layouts" / "library_v1")
    seeds = [3, 7, 11, 19, 23]
    metrics = []
    for seed in seeds:
        world = LibraryWorld(spec)
        simulation = LibrarySimulation(
            world,
            config=SimulationConfig(
                focal_agent_enabled=True,
                focal_agent_initial_seat_history=("QR-SC-05", "QR-SC-05", "QR-SC-04"),
                focal_agent_session_steps=32,
            ),
            seed=seed,
        )
        metrics.append(run_episode(simulation, steps=40))

    avg_occupancy = sum(item.average_occupancy for item in metrics) / len(metrics)
    avg_reseats = sum(item.total_reseats for item in metrics) / len(metrics)
    avg_focal_score = sum(item.focal_average_score or 0.0 for item in metrics) / len(metrics)
    avg_focal_changes = sum(item.focal_seat_changes for item in metrics) / len(metrics)

    print("Experiment summary")
    print("-" * 60)
    print(f"episodes: {len(metrics)}")
    print(f"avg occupancy: {avg_occupancy:.2f}")
    print(f"avg reseats: {avg_reseats:.2f}")
    print(f"avg focal score: {avg_focal_score:.2f}")
    print(f"avg focal seat changes: {avg_focal_changes:.2f}")
    print()
    print("Per-seed details")
    for seed, metric in zip(seeds, metrics):
        focal_zone = metric.focal_time_in_preferred_zone if metric.focal_time_in_preferred_zone is not None else 0.0
        print(
            f"seed={seed:02d} occ_avg={metric.average_occupancy:.2f} peak={metric.peak_occupancy:02d} "
            f"reseats={metric.total_reseats:02d} focal_score={metric.focal_average_score:.2f} "
            f"focal_changes={metric.focal_seat_changes} preferred_zone_time={focal_zone:.2f}"
        )


if __name__ == "__main__":
    main()
