from __future__ import annotations

import json
from pathlib import Path

from .models import (
    Entrance,
    FeatureHotspot,
    FeatureLayer,
    GraphEdge,
    GraphNode,
    LayoutSpec,
    MapBounds,
    Obstacle,
    Seat,
    SpawnWindow,
    Zone,
)


def _read_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_zones(path: Path) -> dict[str, Zone]:
    zones = {}
    for item in _read_json(path):
        zone = Zone(**item)
        if zone.id in zones:
            raise ValueError(f"Duplicate zone id: {zone.id}")
        zones[zone.id] = zone
    return zones


def _load_seats(path: Path) -> dict[str, Seat]:
    seats = {}
    for item in _read_json(path):
        seat = Seat(**item)
        if seat.id in seats:
            raise ValueError(f"Duplicate seat id: {seat.id}")
        seats[seat.id] = seat
    return seats


def _load_entrances(path: Path) -> dict[str, Entrance]:
    entrances = {}
    for item in _read_json(path):
        entrance = Entrance(**item)
        if entrance.id in entrances:
            raise ValueError(f"Duplicate entrance id: {entrance.id}")
        entrances[entrance.id] = entrance
    return entrances


def _load_graph(path: Path) -> tuple[dict[str, GraphNode], tuple[GraphEdge, ...]]:
    payload = _read_json(path)
    nodes: dict[str, GraphNode] = {}
    for item in payload["nodes"]:
        node = GraphNode(id=item["id"], x=item["x"], y=item["y"], tags=tuple(item.get("tags", ())))
        if node.id in nodes:
            raise ValueError(f"Duplicate graph node id: {node.id}")
        nodes[node.id] = node
    edges = tuple(GraphEdge(**item) for item in payload["edges"])
    return nodes, edges


def _load_spawn_windows(path: Path) -> tuple[SpawnWindow, ...]:
    payload = _read_json(path)
    return tuple(SpawnWindow(**item) for item in payload["time_windows"])


def _load_feature_layers(path: Path) -> dict[str, FeatureLayer]:
    payload = _read_json(path)
    layers = {}
    for item in payload["layers"]:
        hotspots = tuple(FeatureHotspot(**hotspot) for hotspot in item.get("hotspots", ()))
        layer = FeatureLayer(
            name=item["name"],
            zone_defaults=item["zone_defaults"],
            hotspots=hotspots,
        )
        if layer.name in layers:
            raise ValueError(f"Duplicate feature layer: {layer.name}")
        layers[layer.name] = layer
    return layers


def _load_obstacles(items: list[dict]) -> tuple[Obstacle, ...]:
    return tuple(Obstacle(**item) for item in items)


def _validate(spec: LayoutSpec) -> None:
    for seat in spec.seats.values():
        if seat.zone_id not in spec.zones:
            raise ValueError(f"Seat {seat.id} references unknown zone {seat.zone_id}")
        if seat.access_node_id not in spec.graph_nodes:
            raise ValueError(f"Seat {seat.id} references unknown graph node {seat.access_node_id}")
        zone = spec.zones[seat.zone_id]
        if not zone.contains(seat.x, seat.y):
            raise ValueError(f"Seat {seat.id} lies outside zone {seat.zone_id}")

    for entrance in spec.entrances.values():
        if entrance.node_id not in spec.graph_nodes:
            raise ValueError(f"Entrance {entrance.id} references unknown node {entrance.node_id}")

    node_ids = set(spec.graph_nodes)
    for edge in spec.graph_edges:
        if edge.source not in node_ids or edge.target not in node_ids:
            raise ValueError(f"Edge {edge.source}->{edge.target} references missing node")
        if edge.cost <= 0:
            raise ValueError(f"Edge {edge.source}->{edge.target} must have positive cost")

    entrance_ids = set(spec.entrances)
    for window in spec.spawn_windows:
        if window.start_hour >= window.end_hour:
            raise ValueError(f"Spawn window {window.id} must have start < end")
        if window.arrival_rate_per_hour < 0 or window.departure_rate_per_hour < 0:
            raise ValueError(f"Spawn window {window.id} has negative rates")
        weight_total = 0.0
        for entrance_id, weight in window.entrance_weights.items():
            if entrance_id not in entrance_ids:
                raise ValueError(f"Spawn window {window.id} references unknown entrance {entrance_id}")
            if weight < 0:
                raise ValueError(f"Spawn window {window.id} has negative weight for {entrance_id}")
            weight_total += weight
        if weight_total <= 0:
            raise ValueError(f"Spawn window {window.id} must have positive total entrance weight")

    zone_ids = set(spec.zones)
    for layer in spec.feature_layers.values():
        for zone_id in layer.zone_defaults:
            if zone_id not in zone_ids:
                raise ValueError(f"Feature layer {layer.name} references unknown zone {zone_id}")


def load_layout(layout_dir: str | Path) -> LayoutSpec:
    layout_root = Path(layout_dir)
    manifest = _read_json(layout_root / "layout_manifest.json")
    bounds = MapBounds(**manifest["map"])
    zones = _load_zones(layout_root / manifest["files"]["zones"])
    seats = _load_seats(layout_root / manifest["files"]["seats"])
    entrances = _load_entrances(layout_root / manifest["files"]["entrances"])
    graph_nodes, graph_edges = _load_graph(layout_root / manifest["files"]["walk_graph"])
    spawn_windows = _load_spawn_windows(layout_root / manifest["files"]["spawn_schedule"])
    feature_layers = _load_feature_layers(layout_root / manifest["files"]["feature_fields"])
    obstacles = _load_obstacles(manifest.get("obstacles", []))

    spec = LayoutSpec(
        layout_id=manifest["layout_id"],
        name=manifest["name"],
        version=manifest["version"],
        description=manifest["description"],
        bounds=bounds,
        seat_type_weights=manifest.get("seat_type_weights", {}),
        zones=zones,
        seats=seats,
        entrances=entrances,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        spawn_windows=spawn_windows,
        feature_layers=feature_layers,
        obstacles=obstacles,
    )
    _validate(spec)
    return spec

