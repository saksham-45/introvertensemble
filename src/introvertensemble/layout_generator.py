"""Procedural generator for randomized library layouts.

Emits the full asset bundle a layout needs (manifest, zones, seats, entrances,
walk graph, spawn schedule, feature fields) such that ``load_layout`` validates
it. The point is domain randomization: train/validate/test on *disjoint* pools of
generated layouts so generalization can be measured on maps the agent never saw
(ProcGen protocol — see docs/TRAINING_BEST_PRACTICES.md).

Design constraints honored:
- every seat lies inside its zone's bounding box (validator);
- the walk graph is a connected grid so all-pairs shortest paths exist;
- every feature layer has a default for every zone (scorer reads them directly);
- canonical event zone ids (z_quiet_room_1, z_conference_room_1,
  z_high_turnover_1, z_cafe_pantry_1) are emitted for matching zone types so the
  default EventEngine templates actually fire.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Feature layers the scorer reads; every layer needs a default per zone.
FEATURE_LAYERS = (
    "wifi_strength",
    "static_noise",
    "foot_traffic",
    "interruption_risk",
    "privacy",
    "future_crowding_risk",
    "stability",
)

SEAT_TYPE_WEIGHTS = {
    "quiet_carrel": 10,
    "silent_carrel": 9,
    "deep_carrel": 8,
    "individual_desk": 5,
    "shared_table": 5,
    "booth": 5,
    "high_desk": 3,
    "lounge_chair": 2,
    "conference_chair": 2,
}

# Per-zone-type generation profile: seat types drawn here, plus the baseline
# feature field values (privacy/noise/etc.) that make the type behave in
# character. Canonical id ties a type to the default event templates.
@dataclass(frozen=True)
class ZoneKind:
    zone_type: str
    canonical_id: str | None
    seat_types: tuple[str, ...]
    privacy: float
    static_noise: float
    interruption_risk: float
    foot_traffic: float
    future_crowding_risk: float
    stability: float
    wifi_strength: float


ZONE_KINDS = (
    ZoneKind("quiet_study", "z_quiet_room_1",
             ("quiet_carrel", "silent_carrel", "deep_carrel"),
             privacy=0.82, static_noise=0.12, interruption_risk=0.14,
             foot_traffic=0.15, future_crowding_risk=0.20, stability=0.85, wifi_strength=0.7),
    ZoneKind("general_study", None,
             ("individual_desk", "shared_table", "high_desk"),
             privacy=0.55, static_noise=0.35, interruption_risk=0.35,
             foot_traffic=0.4, future_crowding_risk=0.45, stability=0.6, wifi_strength=0.75),
    ZoneKind("collaboration", "z_conference_room_1",
             ("shared_table", "booth", "conference_chair"),
             privacy=0.3, static_noise=0.6, interruption_risk=0.55,
             foot_traffic=0.6, future_crowding_risk=0.6, stability=0.4, wifi_strength=0.7),
    ZoneKind("lounge_casual", "z_cafe_pantry_1",
             ("lounge_chair", "booth"),
             privacy=0.4, static_noise=0.55, interruption_risk=0.45,
             foot_traffic=0.55, future_crowding_risk=0.5, stability=0.45, wifi_strength=0.6),
    ZoneKind("high_disturbance", "z_high_turnover_1",
             ("high_desk", "individual_desk"),
             privacy=0.28, static_noise=0.7, interruption_risk=0.7,
             foot_traffic=0.75, future_crowding_risk=0.7, stability=0.3, wifi_strength=0.65),
)

SEAT_TYPE_ABBR = {
    "quiet_carrel": "QC", "silent_carrel": "SC", "deep_carrel": "DC",
    "individual_desk": "ID", "shared_table": "ST", "high_desk": "HD",
    "booth": "Bo", "conference_chair": "CC", "lounge_chair": "LC",
}


def _jitter(rng: np.random.Generator, value: float, spread: float, lo: float, hi: float) -> float:
    return float(np.clip(value + rng.uniform(-spread, spread), lo, hi))


def generate_layout_dict(seed: int, difficulty: float = 0.5) -> dict[str, object]:
    """Return a dict of {filename: json-serializable} for one procedural layout.

    ``difficulty`` in [0,1] scales up crowd pressure and zone count, giving a
    knob for curriculum / stress evaluation.
    """
    rng = np.random.default_rng(seed)

    # ---- Geometry: a grid of zone cells, each a room of one kind ------------
    n_cols = int(rng.integers(3, 5))
    n_rows = int(rng.integers(2, 4))
    cell_w = _jitter(rng, 6.0, 1.0, 4.5, 7.5)
    cell_h = _jitter(rng, 6.0, 1.0, 4.5, 7.5)
    margin = 0.8
    width = round(n_cols * cell_w + 2 * margin, 2)
    height = round(n_rows * cell_h + 2 * margin, 2)

    # Assign a zone kind to each cell. Guarantee the canonical-id kinds appear so
    # the default events fire; fill the rest randomly.
    cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    rng.shuffle(cells)
    must_have = [k for k in ZONE_KINDS if k.canonical_id is not None]
    assignments: list[tuple[tuple[int, int], ZoneKind, str]] = []
    used_canonical: set[str] = set()
    for i, cell in enumerate(cells):
        kind = must_have[i] if i < len(must_have) else ZONE_KINDS[int(rng.integers(0, len(ZONE_KINDS)))]
        # Canonical id only once; subsequent same-type zones get a unique id.
        if kind.canonical_id and kind.canonical_id not in used_canonical:
            zone_id = kind.canonical_id
            used_canonical.add(kind.canonical_id)
        else:
            zone_id = f"z_{kind.zone_type}_{i}"
        assignments.append((cell, kind, zone_id))

    zones = []
    seats = []
    graph_nodes = []
    graph_edges = []
    zone_defaults: dict[str, dict[str, float]] = {layer: {} for layer in FEATURE_LAYERS}
    cell_center_node: dict[tuple[int, int], str] = {}
    seat_counter: dict[str, int] = {}

    for (r, c), kind, zone_id in assignments:
        x1 = round(margin + c * cell_w, 2)
        y1 = round(margin + r * cell_h, 2)
        x2 = round(x1 + cell_w - 0.4, 2)
        y2 = round(y1 + cell_h - 0.4, 2)
        zones.append({
            "id": zone_id,
            "name": f"{kind.zone_type.replace('_', ' ').title()} {r}{c}",
            "zone_type": kind.zone_type,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "default_privacy": round(kind.privacy, 3),
            "default_noise": round(kind.static_noise, 3),
            "default_traffic": round(kind.foot_traffic, 3),
            "notes": f"Procedural {kind.zone_type} zone (seed {seed}).",
        })
        # Feature-field defaults for this zone (jittered per layer).
        base = {
            "wifi_strength": kind.wifi_strength, "static_noise": kind.static_noise,
            "foot_traffic": kind.foot_traffic, "interruption_risk": kind.interruption_risk,
            "privacy": kind.privacy, "future_crowding_risk": kind.future_crowding_risk,
            "stability": kind.stability,
        }
        for layer in FEATURE_LAYERS:
            zone_defaults[layer][zone_id] = _jitter(rng, base[layer], 0.06, 0.02, 0.98)

        # One graph node at the cell center; seats attach to it.
        cx = round((x1 + x2) / 2, 2)
        cy = round((y1 + y2) / 2, 2)
        node_id = f"n_{r}_{c}"
        graph_nodes.append({"id": node_id, "x": cx, "y": cy, "tags": []})
        cell_center_node[(r, c)] = node_id

        # Seats: a small jittered grid inside the zone box, kept strictly inside.
        n_seat_rows = int(rng.integers(2, 4))
        n_seat_cols = int(rng.integers(2, 4))
        pad = 0.5
        for sr in range(n_seat_rows):
            for sc in range(n_seat_cols):
                fx = (sc + 1) / (n_seat_cols + 1)
                fy = (sr + 1) / (n_seat_rows + 1)
                sx = round(x1 + pad + fx * (x2 - x1 - 2 * pad), 2)
                sy = round(y1 + pad + fy * (y2 - y1 - 2 * pad), 2)
                sx = float(np.clip(sx, x1 + 0.05, x2 - 0.05))
                sy = float(np.clip(sy, y1 + 0.05, y2 - 0.05))
                seat_type = kind.seat_types[int(rng.integers(0, len(kind.seat_types)))]
                abbr = SEAT_TYPE_ABBR[seat_type]
                seat_counter[abbr] = seat_counter.get(abbr, 0) + 1
                seats.append({
                    "id": f"{abbr}-{seat_counter[abbr]:02d}",
                    "zone_id": zone_id,
                    "seat_type": seat_type,
                    "x": round(sx, 2), "y": round(sy, 2),
                    "access_node_id": node_id,
                    "neighbor_radius": round(_jitter(rng, 1.4, 0.2, 1.0, 1.9), 2),
                    "features": {
                        "comfort": round(_jitter(rng, 0.6, 0.2, 0.2, 0.95), 2),
                        "outlet": bool(rng.random() < 0.5),
                        "window_view": round(rng.uniform(0.0, 0.4), 2),
                    },
                })

    # ---- Walk graph: connect the cell grid 4-neighbour so it is connected ----
    def node_xy(nid: str) -> tuple[float, float]:
        for n in graph_nodes:
            if n["id"] == nid:
                return n["x"], n["y"]
        raise KeyError(nid)

    def edge(a: str, b: str) -> None:
        ax, ay = node_xy(a)
        bx, by = node_xy(b)
        cost = round(max(0.5, ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5), 2)
        graph_edges.append({"source": a, "target": b, "cost": cost, "bidirectional": True})

    for r in range(n_rows):
        for c in range(n_cols):
            if c + 1 < n_cols:
                edge(cell_center_node[(r, c)], cell_center_node[(r, c + 1)])
            if r + 1 < n_rows:
                edge(cell_center_node[(r, c)], cell_center_node[(r + 1, c)])

    # ---- Entrances: 2-4 on the perimeter, wired to nearest cell node ---------
    n_entrances = int(rng.integers(2, 5))
    entrances = []
    corners = [(0, 0), (0, n_cols - 1), (n_rows - 1, 0), (n_rows - 1, n_cols - 1)]
    rng.shuffle(corners)
    for i in range(n_entrances):
        r, c = corners[i % len(corners)]
        near_node = cell_center_node[(r, c)]
        nx, ny = node_xy(near_node)
        ent_node_id = f"n_e{i + 1}"
        ex = round(float(np.clip(nx + rng.uniform(-1.0, 1.0), 0.2, width - 0.2)), 2)
        ey = round(float(np.clip(ny + rng.uniform(-1.0, 1.0), 0.2, height - 0.2)), 2)
        graph_nodes.append({"id": ent_node_id, "x": ex, "y": ey, "tags": ["entrance"]})
        edge(ent_node_id, near_node)
        entrances.append({
            "id": f"E{i + 1}",
            "label": f"Gate {i + 1}",
            "x": ex, "y": ey,
            "node_id": ent_node_id,
            "kind": "main" if i == 0 else "side",
            "notes": f"Procedural entrance {i + 1}.",
        })
    entrance_ids = [e["id"] for e in entrances]

    # ---- Spawn schedule: contiguous windows 8..18, crowd scaled by difficulty -
    def entrance_weights() -> dict[str, float]:
        w = rng.uniform(0.5, 1.5, size=len(entrance_ids))
        w = w / w.sum()
        return {eid: round(float(x), 3) for eid, x in zip(entrance_ids, w)}

    crowd = 0.6 + 0.9 * difficulty  # multiplier on arrival rate
    windows = [
        ("morning_open", 8, 10, 12.0), ("midday_peak", 10, 13, 20.0),
        ("afternoon", 13, 16, 16.0), ("evening", 16, 18, 10.0),
    ]
    time_windows = []
    for wid, start, end, base_rate in windows:
        time_windows.append({
            "id": wid, "start_hour": start, "end_hour": end,
            "arrival_rate_per_hour": round(base_rate * crowd * rng.uniform(0.9, 1.1), 2),
            "departure_rate_per_hour": round(base_rate * 0.4 * rng.uniform(0.9, 1.1), 2),
            "reseat_pressure": round(_jitter(rng, 0.25, 0.1, 0.05, 0.5), 3),
            "entrance_weights": entrance_weights(),
        })

    # ---- Feature fields: zone defaults + a few random hotspots ---------------
    layers = []
    for layer in FEATURE_LAYERS:
        hotspots = []
        for h in range(int(rng.integers(0, 3))):
            hx = round(rng.uniform(margin, width - margin), 2)
            hy = round(rng.uniform(margin, height - margin), 2)
            hotspots.append({
                "id": f"{layer}_hs{h}", "x": hx, "y": hy,
                "radius": round(rng.uniform(1.5, 3.5), 2),
                "delta": round(rng.uniform(-0.15, 0.15), 3),
            })
        layers.append({
            "name": layer,
            "zone_defaults": {zid: round(v, 3) for zid, v in zone_defaults[layer].items()},
            "hotspots": hotspots,
        })

    manifest = {
        "layout_id": f"gen_{seed}",
        "name": f"Generated Library (seed {seed})",
        "version": "1.0.0",
        "description": f"Procedurally generated layout, seed {seed}, difficulty {difficulty:.2f}.",
        "map": {"width": width, "height": height, "unit": "meters", "coordinate_origin": "bottom_left"},
        "files": {
            "zones": "zones.json", "seats": "seats.json", "entrances": "entrances.json",
            "walk_graph": "walk_graph.json", "spawn_schedule": "spawn_schedule.json",
            "feature_fields": "feature_fields.json",
        },
        "seat_type_weights": SEAT_TYPE_WEIGHTS,
        "obstacles": [],
    }

    return {
        "layout_manifest.json": manifest,
        "zones.json": zones,
        "seats.json": seats,
        "entrances.json": entrances,
        "walk_graph.json": {"nodes": graph_nodes, "edges": graph_edges},
        "spawn_schedule.json": {"time_windows": time_windows},
        "feature_fields.json": {"layers": layers},
    }


def write_layout(out_dir: Path, seed: int, difficulty: float = 0.5) -> Path:
    """Generate one layout and write its asset bundle under ``out_dir``."""
    bundle = generate_layout_dict(seed, difficulty=difficulty)
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in bundle.items():
        (out_dir / filename).write_text(json.dumps(payload, indent=2))
    return out_dir
