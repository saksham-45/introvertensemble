from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentProfile:
    name: str
    privacy_weight: float
    wifi_weight: float
    comfort_weight: float
    outlet_weight: float
    familiarity_weight: float
    stability_weight: float
    crowd_intolerance: float
    noise_sensitivity: float
    interruption_sensitivity: float
    future_crowding_sensitivity: float
    movement_aversion: float
    switching_aversion: float
    turnover_sensitivity: float
    seat_type_bias: float = 0.10

    @classmethod
    def introvert(cls) -> "AgentProfile":
        return cls(
            name="introvert",
            privacy_weight=1.45,
            wifi_weight=0.70,
            comfort_weight=0.60,
            outlet_weight=0.25,
            familiarity_weight=0.80,
            stability_weight=0.95,
            crowd_intolerance=1.55,
            noise_sensitivity=1.25,
            interruption_sensitivity=1.45,
            future_crowding_sensitivity=1.15,
            movement_aversion=0.35,
            switching_aversion=0.85,
            turnover_sensitivity=1.10,
            seat_type_bias=0.18,
        )

    @classmethod
    def focal_introvert(cls) -> "AgentProfile":
        return cls(
            name="focal_introvert",
            privacy_weight=1.65,
            wifi_weight=0.72,
            comfort_weight=0.62,
            outlet_weight=0.22,
            familiarity_weight=1.10,
            stability_weight=1.15,
            crowd_intolerance=1.85,
            noise_sensitivity=1.35,
            interruption_sensitivity=1.65,
            future_crowding_sensitivity=1.25,
            movement_aversion=0.48,
            switching_aversion=1.15,
            turnover_sensitivity=1.25,
            seat_type_bias=0.20,
        )

    @classmethod
    def standard(cls) -> "AgentProfile":
        return cls(
            name="standard",
            privacy_weight=0.90,
            wifi_weight=0.65,
            comfort_weight=0.65,
            outlet_weight=0.20,
            familiarity_weight=0.30,
            stability_weight=0.40,
            crowd_intolerance=0.80,
            noise_sensitivity=0.70,
            interruption_sensitivity=0.65,
            future_crowding_sensitivity=0.45,
            movement_aversion=0.20,
            switching_aversion=0.45,
            turnover_sensitivity=0.35,
            seat_type_bias=0.10,
        )


@dataclass
class SimAgent:
    id: str
    profile: AgentProfile
    entrance_id: str
    session_steps_remaining: int
    role: str = "background"
    current_seat_id: str | None = None
    seat_history: dict[str, int] = field(default_factory=dict)
    zone_history: dict[str, int] = field(default_factory=dict)
    preferred_seat_id: str | None = None
    preferred_zone_id: str | None = None
    local_search_radius: float = 2.0
    arrival_acceptability_threshold: float = 0.35
    stay_threshold: float = 0.65
    leave_threshold: float = -0.05
    switching_improvement_threshold: float = 0.55
    total_moves: int = 0
    arrivals_step: int = 0

    def record_seat(self, seat_id: str, zone_id: str) -> None:
        self.seat_history[seat_id] = self.seat_history.get(seat_id, 0) + 1
        self.zone_history[zone_id] = self.zone_history.get(zone_id, 0) + 1
        self.preferred_seat_id = self.dominant_seat_id()
        self.preferred_zone_id = self.dominant_zone_id()
        self.current_seat_id = seat_id

    def dominant_seat_id(self) -> str | None:
        if not self.seat_history:
            return self.preferred_seat_id
        return max(self.seat_history.items(), key=lambda item: (item[1], item[0]))[0]

    def dominant_zone_id(self) -> str | None:
        if not self.zone_history:
            return self.preferred_zone_id
        return max(self.zone_history.items(), key=lambda item: (item[1], item[0]))[0]
