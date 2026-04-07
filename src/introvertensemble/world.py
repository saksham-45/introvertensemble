from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict

from .models import LayoutSpec, Seat, SpawnWindow


class LibraryWorld:
    def __init__(self, spec: LayoutSpec):
        self.spec = spec
        self.occupancy: dict[str, str | None] = {seat_id: None for seat_id in spec.seats}
        self._adjacency = self._build_adjacency()

    def _build_adjacency(self) -> dict[str, list[tuple[str, float]]]:
        adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for edge in self.spec.graph_edges:
            adjacency[edge.source].append((edge.target, edge.cost))
            if edge.bidirectional:
                adjacency[edge.target].append((edge.source, edge.cost))
        return dict(adjacency)

    def available_seats(self, zone_id: str | None = None) -> list[Seat]:
        seats = []
        for seat_id, seat in self.spec.seats.items():
            if self.occupancy[seat_id] is not None:
                continue
            if zone_id is not None and seat.zone_id != zone_id:
                continue
            seats.append(seat)
        return seats

    def occupy_seat(self, seat_id: str, agent_id: str) -> None:
        if seat_id not in self.occupancy:
            raise KeyError(f"Unknown seat id {seat_id}")
        if self.occupancy[seat_id] is not None:
            raise ValueError(f"Seat {seat_id} is already occupied")
        self.occupancy[seat_id] = agent_id

    def release_seat(self, seat_id: str) -> None:
        if seat_id not in self.occupancy:
            raise KeyError(f"Unknown seat id {seat_id}")
        self.occupancy[seat_id] = None

    def current_zone_load(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for seat_id, agent_id in self.occupancy.items():
            if agent_id is None:
                continue
            counts[self.spec.seats[seat_id].zone_id] += 1
        return dict(counts)

    def shortest_path_cost(self, start_node_id: str, goal_node_id: str) -> float:
        if start_node_id not in self.spec.graph_nodes:
            raise KeyError(f"Unknown graph node {start_node_id}")
        if goal_node_id not in self.spec.graph_nodes:
            raise KeyError(f"Unknown graph node {goal_node_id}")
        frontier: list[tuple[float, str]] = [(0.0, start_node_id)]
        best_cost: dict[str, float] = {start_node_id: 0.0}

        while frontier:
            cost, node_id = heapq.heappop(frontier)
            if node_id == goal_node_id:
                return cost
            if cost > best_cost[node_id]:
                continue
            for next_node_id, edge_cost in self._adjacency.get(node_id, []):
                next_cost = cost + edge_cost
                if next_cost < best_cost.get(next_node_id, math.inf):
                    best_cost[next_node_id] = next_cost
                    heapq.heappush(frontier, (next_cost, next_node_id))
        raise ValueError(f"No path between {start_node_id} and {goal_node_id}")

    def path_cost_from_entrance_to_seat(self, entrance_id: str, seat_id: str) -> float:
        entrance = self.spec.entrances[entrance_id]
        seat = self.spec.seats[seat_id]
        return self.shortest_path_cost(entrance.node_id, seat.access_node_id)

    def seat_neighbors(self, seat_id: str, radius: float | None = None) -> list[Seat]:
        seat = self.spec.seats[seat_id]
        threshold = radius if radius is not None else seat.neighbor_radius
        neighbors: list[Seat] = []
        for other in self.spec.seats.values():
            if other.id == seat.id:
                continue
            distance = math.dist((seat.x, seat.y), (other.x, other.y))
            if distance <= threshold:
                neighbors.append(other)
        return neighbors

    def occupied_neighbor_count(self, seat_id: str, radius: float | None = None) -> int:
        return sum(1 for seat in self.seat_neighbors(seat_id, radius=radius) if self.occupancy[seat.id] is not None)

    def local_crowding_ratio(self, seat_id: str, radius: float | None = None) -> float:
        neighbors = self.seat_neighbors(seat_id, radius=radius)
        if not neighbors:
            return 0.0
        occupied = sum(1 for seat in neighbors if self.occupancy[seat.id] is not None)
        return occupied / len(neighbors)

    def active_spawn_window(self, hour: float) -> SpawnWindow:
        hour = hour % 24.0
        for window in self.spec.spawn_windows:
            if window.contains_hour(hour):
                return window
        if self.spec.spawn_windows:
            earliest = min(self.spec.spawn_windows, key=lambda window: window.start_hour)
            latest = max(self.spec.spawn_windows, key=lambda window: window.end_hour)
            if hour < earliest.start_hour:
                return earliest
            if hour >= latest.end_hour:
                return earliest if hour > latest.end_hour else latest
            if hour == latest.end_hour:
                return latest
        raise ValueError(f"No spawn window covers hour {hour}")

    def normalized_entrance_weights(self, hour: float) -> dict[str, float]:
        window = self.active_spawn_window(hour)
        total = sum(window.entrance_weights.values())
        return {entrance_id: weight / total for entrance_id, weight in window.entrance_weights.items()}

    def feature_value_for_seat(self, seat_id: str, layer_name: str) -> float:
        seat = self.spec.seats[seat_id]
        layer = self.spec.feature_layers[layer_name]
        value = layer.zone_defaults[seat.zone_id]
        for hotspot in layer.hotspots:
            distance = math.dist((seat.x, seat.y), (hotspot.x, hotspot.y))
            if distance <= hotspot.radius:
                falloff = 1.0 - (distance / hotspot.radius)
                value += hotspot.delta * falloff
        return max(0.0, min(1.0, value))
