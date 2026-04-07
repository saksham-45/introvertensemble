from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from .simulation import LibrarySimulation, StepSummary


@dataclass(frozen=True)
class EpisodeMetrics:
    steps: int
    total_arrivals: int
    total_departures: int
    total_reseats: int
    average_occupancy: float
    peak_occupancy: int
    average_zone_load: dict[str, float]
    final_profile_counts: dict[str, int]
    focal_present: bool
    focal_seat_changes: int
    focal_unique_seats: int
    focal_average_score: float | None
    focal_min_score: float | None
    focal_average_crowding: float | None
    focal_steps_below_stay_threshold: int
    focal_steps_below_leave_threshold: int
    focal_time_in_preferred_zone: float | None


class MetricsTracker:
    def __init__(self, simulation: LibrarySimulation):
        self.simulation = simulation
        self.step_count = 0
        self.total_arrivals = 0
        self.total_departures = 0
        self.total_reseats = 0
        self.occupancy_sum = 0
        self.peak_occupancy = 0
        self.zone_load_sum: dict[str, int] = defaultdict(int)
        self.focal_scores: list[float] = []
        self.focal_crowding: list[float] = []
        self.focal_seat_sequence: list[str] = []
        self.focal_steps_below_stay_threshold = 0
        self.focal_steps_below_leave_threshold = 0
        self.focal_steps_in_preferred_zone = 0

    def observe(self, summary: StepSummary) -> None:
        self.step_count += 1
        self.total_arrivals += summary.arrivals
        self.total_departures += summary.departures
        self.total_reseats += summary.reseats
        self.occupancy_sum += summary.occupancy
        self.peak_occupancy = max(self.peak_occupancy, summary.occupancy)
        for zone_id, load in summary.zone_load.items():
            self.zone_load_sum[zone_id] += load

        focal = self._focal_agent()
        if focal is None or focal.current_seat_id is None:
            return
        seat_id = focal.current_seat_id
        score = self.simulation.scorer.score_seat(focal, seat_id).total
        crowding = self.simulation.world.local_crowding_ratio(seat_id)
        self.focal_scores.append(score)
        self.focal_crowding.append(crowding)
        self.focal_seat_sequence.append(seat_id)
        if score < focal.stay_threshold:
            self.focal_steps_below_stay_threshold += 1
        if score < focal.leave_threshold:
            self.focal_steps_below_leave_threshold += 1
        preferred_zone_id = focal.dominant_zone_id()
        if preferred_zone_id is not None and self.simulation.world.spec.seats[seat_id].zone_id == preferred_zone_id:
            self.focal_steps_in_preferred_zone += 1

    def finalize(self) -> EpisodeMetrics:
        average_zone_load = {
            zone_id: load / self.step_count
            for zone_id, load in sorted(self.zone_load_sum.items())
        } if self.step_count else {}
        focal_present = bool(self.focal_scores)
        unique_focal_seats = len(set(self.focal_seat_sequence))
        focal_seat_changes = sum(
            1 for prev, curr in zip(self.focal_seat_sequence, self.focal_seat_sequence[1:]) if prev != curr
        )
        focal_avg_score = sum(self.focal_scores) / len(self.focal_scores) if self.focal_scores else None
        focal_min_score = min(self.focal_scores) if self.focal_scores else None
        focal_avg_crowding = sum(self.focal_crowding) / len(self.focal_crowding) if self.focal_crowding else None
        focal_time_in_preferred_zone = (
            self.focal_steps_in_preferred_zone / len(self.focal_seat_sequence)
            if self.focal_seat_sequence
            else None
        )
        return EpisodeMetrics(
            steps=self.step_count,
            total_arrivals=self.total_arrivals,
            total_departures=self.total_departures,
            total_reseats=self.total_reseats,
            average_occupancy=(self.occupancy_sum / self.step_count) if self.step_count else 0.0,
            peak_occupancy=self.peak_occupancy,
            average_zone_load=average_zone_load,
            final_profile_counts=self.simulation.profile_counts(),
            focal_present=focal_present,
            focal_seat_changes=focal_seat_changes,
            focal_unique_seats=unique_focal_seats,
            focal_average_score=focal_avg_score,
            focal_min_score=focal_min_score,
            focal_average_crowding=focal_avg_crowding,
            focal_steps_below_stay_threshold=self.focal_steps_below_stay_threshold,
            focal_steps_below_leave_threshold=self.focal_steps_below_leave_threshold,
            focal_time_in_preferred_zone=focal_time_in_preferred_zone,
        )

    def _focal_agent(self):
        if self.simulation.focal_agent_id is None:
            return None
        return self.simulation.agents.get(self.simulation.focal_agent_id)


def run_episode(simulation: LibrarySimulation, steps: int) -> EpisodeMetrics:
    tracker = MetricsTracker(simulation)
    for _ in range(steps):
        tracker.observe(simulation.step())
    return tracker.finalize()
