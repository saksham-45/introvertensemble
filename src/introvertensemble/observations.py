from __future__ import annotations

from dataclasses import dataclass

from .scoring import SeatScorer
from .simulation import LibrarySimulation


@dataclass(frozen=True)
class SeatObservation:
    seat_id: str
    zone_id: str
    seat_type: str
    score: float
    privacy: float
    noise: float
    interruption: float
    immediate_neighbor_ratio: float
    local_cluster_ratio: float
    zone_density: float
    occupied: bool


@dataclass(frozen=True)
class FocalObservation:
    step_index: int
    hour: float
    current_seat_id: str | None
    current_zone_id: str | None
    current_score: float | None
    preferred_seat_id: str | None
    preferred_zone_id: str | None
    cooldown_steps_remaining: int
    active_events: tuple[tuple[str, str], ...]
    current_seat: SeatObservation | None
    nearby_candidates: tuple[SeatObservation, ...]
    global_candidates: tuple[SeatObservation, ...]


class ObservationBuilder:
    def __init__(self, simulation: LibrarySimulation, top_k: int = 5):
        self.simulation = simulation
        self.top_k = top_k
        self.scorer = SeatScorer(simulation.world)

    def build_focal_observation(self) -> FocalObservation:
        focal_id = self.simulation.focal_agent_id
        if focal_id is None or focal_id not in self.simulation.agents:
            return FocalObservation(
                step_index=self.simulation.step_index,
                hour=self.simulation.current_hour,
                current_seat_id=None,
                current_zone_id=None,
                current_score=None,
                preferred_seat_id=None,
                preferred_zone_id=None,
                cooldown_steps_remaining=0,
                active_events=(),
                current_seat=None,
                nearby_candidates=(),
                global_candidates=(),
            )

        focal = self.simulation.agents[focal_id]
        current_seat = None
        current_score = None
        current_zone_id = None
        if focal.current_seat_id is not None:
            current_zone_id = self.simulation.world.spec.seats[focal.current_seat_id].zone_id
            current_score = self.scorer.score_seat(focal, focal.current_seat_id).total
            current_seat = self._seat_observation(focal.current_seat_id, focal)

        nearby = []
        for seat_id in self.simulation._focal_local_candidates(focal):
            if seat_id == focal.current_seat_id:
                continue
            nearby.append(self._seat_observation(seat_id, focal))
        nearby.sort(key=lambda item: item.score, reverse=True)

        global_candidates = []
        for seat in self.simulation.world.available_seats():
            global_candidates.append(self._seat_observation(seat.id, focal))
        global_candidates.sort(key=lambda item: item.score, reverse=True)

        active_events = ()
        if self.simulation.event_engine is not None:
            active_events = tuple(
                sorted((event.zone_id, event.name) for event in self.simulation.event_engine.active_events)
            )

        return FocalObservation(
            step_index=self.simulation.step_index,
            hour=self.simulation.current_hour,
            current_seat_id=focal.current_seat_id,
            current_zone_id=current_zone_id,
            current_score=current_score,
            preferred_seat_id=focal.dominant_seat_id(),
            preferred_zone_id=focal.dominant_zone_id(),
            cooldown_steps_remaining=focal.cooldown_steps_remaining,
            active_events=active_events,
            current_seat=current_seat,
            nearby_candidates=tuple(nearby[: self.top_k]),
            global_candidates=tuple(global_candidates[: self.top_k]),
        )

    def _seat_observation(self, seat_id: str, agent) -> SeatObservation:
        seat = self.simulation.world.spec.seats[seat_id]
        score = self.scorer.score_seat(agent, seat_id).total
        zone_id = seat.zone_id
        return SeatObservation(
            seat_id=seat_id,
            zone_id=zone_id,
            seat_type=seat.seat_type,
            score=score,
            privacy=self.simulation.world.feature_value_for_seat(seat_id, "privacy"),
            noise=self.simulation.world.feature_value_for_seat(seat_id, "static_noise"),
            interruption=self.simulation.world.feature_value_for_seat(seat_id, "interruption_risk"),
            immediate_neighbor_ratio=self.simulation.world.immediate_neighbor_ratio(seat_id),
            local_cluster_ratio=self.simulation.world.local_crowding_ratio(seat_id),
            zone_density=self.simulation.world.zone_density(zone_id),
            occupied=self.simulation.world.occupancy[seat_id] is not None,
        )
