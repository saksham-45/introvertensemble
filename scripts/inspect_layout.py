from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from introvertensemble import LibraryWorld, load_layout


def main() -> None:
    layout_dir = ROOT / "assets" / "layouts" / "library_v1"
    spec = load_layout(layout_dir)
    world = LibraryWorld(spec)

    seat_types = Counter(seat.seat_type for seat in spec.seats.values())
    zone_counts = Counter(seat.zone_id for seat in spec.seats.values())

    print(f"Layout: {spec.name} ({spec.layout_id} v{spec.version})")
    print(f"Map size: {spec.bounds.width} x {spec.bounds.height} {spec.bounds.unit}")
    print(
        f"Zones: {len(spec.zones)} | Seats: {len(spec.seats)} | Entrances: {len(spec.entrances)} "
        f"| Graph nodes: {len(spec.graph_nodes)} | Edges: {len(spec.graph_edges)}"
    )
    print("Seat types:")
    for seat_type, count in sorted(seat_types.items()):
        print(f"  - {seat_type}: {count}")
    print("Seats per zone:")
    for zone_id, count in sorted(zone_counts.items()):
        zone_name = spec.zones[zone_id].name
        print(f"  - {zone_id} ({zone_name}): {count}")

    print("Spawn windows:")
    for window in spec.spawn_windows:
        weights = world.normalized_entrance_weights(window.start_hour)
        formatted_weights = ", ".join(f"{entrance}={weight:.2f}" for entrance, weight in sorted(weights.items()))
        print(
            f"  - {window.id}: {window.start_hour}:00-{window.end_hour}:00 "
            f"arrivals/h={window.arrival_rate_per_hour:.1f} departures/h={window.departure_rate_per_hour:.1f} "
            f"reseat={window.reseat_pressure:.2f} [{formatted_weights}]"
        )

    sample_seat = "QR-DC-01"
    print(f"Sample feature snapshot for {sample_seat}:")
    for layer_name in sorted(spec.feature_layers):
        value = world.feature_value_for_seat(sample_seat, layer_name)
        print(f"  - {layer_name}: {value:.3f}")

    route_examples = [
        ("E1", "QR-DC-01"),
        ("E2", "OR-ST-05"),
        ("E4", "DH-T1-02"),
        ("F1", "RR-ID-03"),
    ]
    print("Path cost examples:")
    for entrance_id, seat_id in route_examples:
        cost = world.path_cost_from_entrance_to_seat(entrance_id, seat_id)
        print(f"  - {entrance_id} -> {seat_id}: {cost:.2f} m")


if __name__ == "__main__":
    main()
