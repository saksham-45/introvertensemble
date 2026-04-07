from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MapBounds:
    width: float
    height: float
    unit: str
    coordinate_origin: str


@dataclass(frozen=True)
class Zone:
    id: str
    name: str
    zone_type: str
    x1: float
    y1: float
    x2: float
    y2: float
    default_privacy: float
    default_noise: float
    default_traffic: float
    notes: str = ""

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass(frozen=True)
class Seat:
    id: str
    zone_id: str
    seat_type: str
    x: float
    y: float
    access_node_id: str
    neighbor_radius: float
    features: dict[str, float | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class Entrance:
    id: str
    label: str
    x: float
    y: float
    node_id: str
    kind: str
    notes: str = ""


@dataclass(frozen=True)
class GraphNode:
    id: str
    x: float
    y: float
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    cost: float
    bidirectional: bool = True


@dataclass(frozen=True)
class SpawnWindow:
    id: str
    start_hour: int
    end_hour: int
    arrival_rate_per_hour: float
    departure_rate_per_hour: float
    reseat_pressure: float
    entrance_weights: dict[str, float]

    def contains_hour(self, hour: float) -> bool:
        return self.start_hour <= hour < self.end_hour


@dataclass(frozen=True)
class FeatureHotspot:
    id: str
    x: float
    y: float
    radius: float
    delta: float


@dataclass(frozen=True)
class FeatureLayer:
    name: str
    zone_defaults: dict[str, float]
    hotspots: tuple[FeatureHotspot, ...] = ()


@dataclass(frozen=True)
class Obstacle:
    id: str
    label: str
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class LayoutSpec:
    layout_id: str
    name: str
    version: str
    description: str
    bounds: MapBounds
    seat_type_weights: dict[str, int]
    zones: dict[str, Zone]
    seats: dict[str, Seat]
    entrances: dict[str, Entrance]
    graph_nodes: dict[str, GraphNode]
    graph_edges: tuple[GraphEdge, ...]
    spawn_windows: tuple[SpawnWindow, ...]
    feature_layers: dict[str, FeatureLayer]
    obstacles: tuple[Obstacle, ...]


@dataclass(frozen=True)
class EventTemplate:
    id: str
    name: str
    zone_ids: tuple[str, ...]
    probability_per_step: float
    min_duration_steps: int
    max_duration_steps: int
    layer_deltas: dict[str, float]
    max_active_instances: int = 1


@dataclass
class ActiveEvent:
    id: str
    template_id: str
    name: str
    zone_id: str
    remaining_steps: int
    layer_deltas: dict[str, float]
