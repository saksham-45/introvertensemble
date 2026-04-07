from __future__ import annotations

import math
from dataclasses import dataclass

from .agents import SimAgent
from .world import LibraryWorld


@dataclass(frozen=True)
class SeatScoreBreakdown:
    total: float
    privacy: float
    wifi: float
    comfort: float
    outlet: float
    stability: float
    familiarity: float
    seat_type: float
    crowd_penalty: float
    noise_penalty: float
    interruption_penalty: float
    future_crowding_penalty: float
    movement_penalty: float
    turnover_penalty: float


class SeatScorer:
    def __init__(self, world: LibraryWorld):
        self.world = world
        self._max_seat_weight = max(world.spec.seat_type_weights.values(), default=1)
        self._max_path_cost = math.dist((0.0, 0.0), (world.spec.bounds.width, world.spec.bounds.height))

    def score_seat(self, agent: SimAgent, seat_id: str, origin_entrance_id: str | None = None) -> SeatScoreBreakdown:
        seat = self.world.spec.seats[seat_id]
        profile = agent.profile
        zone_id = seat.zone_id

        privacy_value = self.world.feature_value_for_seat(seat_id, "privacy")
        wifi_value = self.world.feature_value_for_seat(seat_id, "wifi_strength")
        stability_value = self.world.feature_value_for_seat(seat_id, "stability")
        noise_value = self.world.feature_value_for_seat(seat_id, "static_noise")
        interruption_value = self.world.feature_value_for_seat(seat_id, "interruption_risk")
        future_crowding_value = self.world.feature_value_for_seat(seat_id, "future_crowding_risk")

        comfort_value = float(seat.features.get("comfort", 0.5))
        outlet_value = 1.0 if seat.features.get("outlet", False) else 0.0
        seat_type_value = self.world.spec.seat_type_weights.get(seat.seat_type, 0) / self._max_seat_weight

        seat_visits = agent.seat_history.get(seat_id, 0)
        zone_visits = agent.zone_history.get(zone_id, 0)
        familiarity_value = min(1.0, 0.25 * seat_visits + 0.12 * zone_visits)

        crowd_ratio = self.world.local_crowding_ratio(seat_id)
        turnover_penalty_value = 1.0 - stability_value

        move_origin = origin_entrance_id or agent.entrance_id
        movement_cost = self.world.path_cost_from_entrance_to_seat(move_origin, seat_id) / self._max_path_cost
        if agent.current_seat_id is not None:
            current_access = self.world.spec.seats[agent.current_seat_id].access_node_id
            movement_cost = self.world.shortest_path_cost(current_access, seat.access_node_id) / self._max_path_cost

        privacy = profile.privacy_weight * privacy_value
        wifi = profile.wifi_weight * wifi_value
        comfort = profile.comfort_weight * comfort_value
        outlet = profile.outlet_weight * outlet_value
        stability = profile.stability_weight * stability_value
        familiarity = profile.familiarity_weight * familiarity_value
        seat_type = profile.seat_type_bias * seat_type_value

        crowd_penalty = profile.crowd_intolerance * crowd_ratio
        noise_penalty = profile.noise_sensitivity * noise_value
        interruption_penalty = profile.interruption_sensitivity * interruption_value
        future_crowding_penalty = profile.future_crowding_sensitivity * future_crowding_value
        movement_penalty = profile.movement_aversion * movement_cost
        turnover_penalty = profile.turnover_sensitivity * turnover_penalty_value

        total = (
            privacy
            + wifi
            + comfort
            + outlet
            + stability
            + familiarity
            + seat_type
            - crowd_penalty
            - noise_penalty
            - interruption_penalty
            - future_crowding_penalty
            - movement_penalty
            - turnover_penalty
        )
        return SeatScoreBreakdown(
            total=total,
            privacy=privacy,
            wifi=wifi,
            comfort=comfort,
            outlet=outlet,
            stability=stability,
            familiarity=familiarity,
            seat_type=seat_type,
            crowd_penalty=crowd_penalty,
            noise_penalty=noise_penalty,
            interruption_penalty=interruption_penalty,
            future_crowding_penalty=future_crowding_penalty,
            movement_penalty=movement_penalty,
            turnover_penalty=turnover_penalty,
        )

