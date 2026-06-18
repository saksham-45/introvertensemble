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
    zone_preferences: dict[str, float] = field(default_factory=dict)
    seat_type_preferences: dict[str, float] = field(default_factory=dict)

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
            future_crowding_sensitivity=0.25,
            movement_aversion=0.35,
            switching_aversion=0.85,
            turnover_sensitivity=1.10,
            seat_type_bias=0.18,
            zone_preferences={
                "quiet_study": 0.35,
                "general_study": 0.10,
                "lounge_casual": -0.10,
                "collaboration": -0.28,
                "high_disturbance": -0.35,
            },
            seat_type_preferences={
                "quiet_carrel": 0.18,
                "silent_carrel": 0.22,
                "deep_carrel": 0.24,
                "lounge_chair": -0.08,
                "shared_table": -0.05,
            },
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
            future_crowding_sensitivity=0.08,
            movement_aversion=0.48,
            switching_aversion=1.15,
            turnover_sensitivity=1.25,
            seat_type_bias=0.20,
            zone_preferences={
                "quiet_study": 0.45,
                "general_study": 0.08,
                "lounge_casual": -0.14,
                "collaboration": -0.40,
                "high_disturbance": -0.48,
            },
            seat_type_preferences={
                "quiet_carrel": 0.22,
                "silent_carrel": 0.26,
                "deep_carrel": 0.30,
                "individual_desk": 0.08,
                "shared_table": -0.10,
                "lounge_chair": -0.12,
            },
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
            future_crowding_sensitivity=0.12,
            movement_aversion=0.20,
            switching_aversion=0.45,
            turnover_sensitivity=0.35,
            seat_type_bias=0.10,
            zone_preferences={
                "general_study": 0.10,
                "lounge_casual": 0.02,
                "collaboration": -0.02,
            },
            seat_type_preferences={
                "individual_desk": 0.05,
                "shared_table": 0.03,
                "high_desk": 0.02,
            },
        )

    @classmethod
    def quiet_seeker(cls) -> "AgentProfile":
        return cls(
            name="quiet_seeker",
            privacy_weight=1.20,
            wifi_weight=0.60,
            comfort_weight=0.58,
            outlet_weight=0.15,
            familiarity_weight=0.40,
            stability_weight=0.75,
            crowd_intolerance=1.15,
            noise_sensitivity=1.10,
            interruption_sensitivity=1.05,
            future_crowding_sensitivity=0.18,
            movement_aversion=0.24,
            switching_aversion=0.52,
            turnover_sensitivity=0.72,
            seat_type_bias=0.14,
            zone_preferences={
                "quiet_study": 0.28,
                "general_study": 0.06,
                "collaboration": -0.18,
                "high_disturbance": -0.25,
            },
            seat_type_preferences={
                "quiet_carrel": 0.16,
                "silent_carrel": 0.18,
                "deep_carrel": 0.20,
                "shared_table": -0.04,
            },
        )

    @classmethod
    def comfort_seeker(cls) -> "AgentProfile":
        return cls(
            name="comfort_seeker",
            privacy_weight=0.62,
            wifi_weight=0.45,
            comfort_weight=1.10,
            outlet_weight=0.10,
            familiarity_weight=0.25,
            stability_weight=0.30,
            crowd_intolerance=0.45,
            noise_sensitivity=0.42,
            interruption_sensitivity=0.40,
            future_crowding_sensitivity=0.10,
            movement_aversion=0.18,
            switching_aversion=0.28,
            turnover_sensitivity=0.20,
            seat_type_bias=0.14,
            zone_preferences={
                "lounge_casual": 0.20,
                "general_study": 0.08,
                "high_disturbance": -0.08,
            },
            seat_type_preferences={
                "lounge_chair": 0.22,
                "booth": 0.16,
                "individual_desk": 0.04,
                "high_desk": -0.08,
            },
        )

    @classmethod
    def outlet_seeker(cls) -> "AgentProfile":
        return cls(
            name="outlet_seeker",
            privacy_weight=0.70,
            wifi_weight=0.82,
            comfort_weight=0.52,
            outlet_weight=0.85,
            familiarity_weight=0.20,
            stability_weight=0.28,
            crowd_intolerance=0.55,
            noise_sensitivity=0.46,
            interruption_sensitivity=0.44,
            future_crowding_sensitivity=0.10,
            movement_aversion=0.16,
            switching_aversion=0.26,
            turnover_sensitivity=0.22,
            seat_type_bias=0.12,
            zone_preferences={
                "general_study": 0.12,
                "lounge_casual": 0.04,
                "collaboration": 0.03,
            },
            seat_type_preferences={
                "individual_desk": 0.12,
                "high_desk": 0.10,
                "shared_table": 0.03,
            },
        )

    @classmethod
    def opportunistic_reseater(cls) -> "AgentProfile":
        return cls(
            name="opportunistic_reseater",
            privacy_weight=0.76,
            wifi_weight=0.60,
            comfort_weight=0.56,
            outlet_weight=0.16,
            familiarity_weight=0.08,
            stability_weight=0.16,
            crowd_intolerance=0.52,
            noise_sensitivity=0.38,
            interruption_sensitivity=0.34,
            future_crowding_sensitivity=0.06,
            movement_aversion=0.04,
            switching_aversion=0.08,
            turnover_sensitivity=0.12,
            seat_type_bias=0.08,
            zone_preferences={
                "general_study": 0.10,
                "lounge_casual": 0.06,
            },
            seat_type_preferences={
                "individual_desk": 0.05,
                "shared_table": 0.05,
                "high_desk": 0.06,
            },
        )

    @classmethod
    def collaborator(cls) -> "AgentProfile":
        return cls(
            name="collaborator",
            privacy_weight=0.25,
            wifi_weight=0.55,
            comfort_weight=0.60,
            outlet_weight=0.12,
            familiarity_weight=0.12,
            stability_weight=0.18,
            crowd_intolerance=0.12,
            noise_sensitivity=0.18,
            interruption_sensitivity=0.20,
            future_crowding_sensitivity=0.02,
            movement_aversion=0.10,
            switching_aversion=0.16,
            turnover_sensitivity=0.10,
            seat_type_bias=0.10,
            zone_preferences={
                "collaboration": 0.34,
                "general_study": 0.08,
                "quiet_study": -0.20,
            },
            seat_type_preferences={
                "shared_table": 0.16,
                "booth": 0.14,
                "conference_chair": 0.10,
                "quiet_carrel": -0.14,
                "silent_carrel": -0.18,
                "deep_carrel": -0.20,
            },
        )

    @classmethod
    def extrovert(cls) -> "AgentProfile":
        return cls(
            name="extrovert",
            privacy_weight=0.22,
            wifi_weight=0.50,
            comfort_weight=0.72,
            outlet_weight=0.10,
            familiarity_weight=0.10,
            stability_weight=0.20,
            crowd_intolerance=0.08,
            noise_sensitivity=0.10,
            interruption_sensitivity=0.12,
            future_crowding_sensitivity=0.02,
            movement_aversion=0.08,
            switching_aversion=0.10,
            turnover_sensitivity=0.08,
            seat_type_bias=0.12,
            zone_preferences={
                "collaboration": 0.38,
                "lounge_casual": 0.22,
                "general_study": 0.10,
                "quiet_study": -0.18,
            },
            seat_type_preferences={
                "shared_table": 0.18,
                "booth": 0.16,
                "conference_chair": 0.12,
                "quiet_carrel": -0.12,
                "silent_carrel": -0.16,
                "deep_carrel": -0.18,
            },
        )

    @classmethod
    def neutral(cls) -> "AgentProfile":
        return cls(
            name="neutral",
            privacy_weight=0.70,
            wifi_weight=0.55,
            comfort_weight=0.62,
            outlet_weight=0.16,
            familiarity_weight=0.24,
            stability_weight=0.34,
            crowd_intolerance=0.55,
            noise_sensitivity=0.48,
            interruption_sensitivity=0.45,
            future_crowding_sensitivity=0.08,
            movement_aversion=0.16,
            switching_aversion=0.32,
            turnover_sensitivity=0.28,
            seat_type_bias=0.08,
            zone_preferences={
                "general_study": 0.06,
                "lounge_casual": 0.02,
                "quiet_study": 0.02,
            },
            seat_type_preferences={
                "individual_desk": 0.04,
                "shared_table": 0.02,
                "high_desk": 0.02,
            },
        )

    @classmethod
    def antisocial(cls) -> "AgentProfile":
        return cls(
            name="antisocial",
            privacy_weight=1.75,
            wifi_weight=0.68,
            comfort_weight=0.58,
            outlet_weight=0.20,
            familiarity_weight=1.20,
            stability_weight=1.20,
            crowd_intolerance=2.10,
            noise_sensitivity=1.45,
            interruption_sensitivity=1.75,
            future_crowding_sensitivity=0.06,
            movement_aversion=0.55,
            switching_aversion=1.25,
            turnover_sensitivity=1.35,
            seat_type_bias=0.22,
            zone_preferences={
                "quiet_study": 0.48,
                "general_study": 0.04,
                "lounge_casual": -0.18,
                "collaboration": -0.48,
                "high_disturbance": -0.55,
            },
            seat_type_preferences={
                "quiet_carrel": 0.24,
                "silent_carrel": 0.28,
                "deep_carrel": 0.32,
                "individual_desk": 0.08,
                "shared_table": -0.12,
                "lounge_chair": -0.14,
            },
        )


PROFILE_REGISTRY = {
    "standard": AgentProfile.standard,
    "introvert": AgentProfile.introvert,
    "focal_introvert": AgentProfile.focal_introvert,
    "quiet_seeker": AgentProfile.quiet_seeker,
    "comfort_seeker": AgentProfile.comfort_seeker,
    "outlet_seeker": AgentProfile.outlet_seeker,
    "opportunistic_reseater": AgentProfile.opportunistic_reseater,
    "collaborator": AgentProfile.collaborator,
    "extrovert": AgentProfile.extrovert,
    "neutral": AgentProfile.neutral,
    "antisocial": AgentProfile.antisocial,
}


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
    cooldown_steps_remaining: int = 0
    steps_in_current_seat: int = 0
    total_moves: int = 0
    arrivals_step: int = 0

    def record_seat(self, seat_id: str, zone_id: str) -> None:
        self.seat_history[seat_id] = self.seat_history.get(seat_id, 0) + 1
        self.zone_history[zone_id] = self.zone_history.get(zone_id, 0) + 1
        self.preferred_seat_id = self.dominant_seat_id()
        self.preferred_zone_id = self.dominant_zone_id()
        self.current_seat_id = seat_id
        self.steps_in_current_seat = 0

    def dominant_seat_id(self) -> str | None:
        if not self.seat_history:
            return self.preferred_seat_id
        return max(self.seat_history.items(), key=lambda item: (item[1], item[0]))[0]

    def dominant_zone_id(self) -> str | None:
        if not self.zone_history:
            return self.preferred_zone_id
        return max(self.zone_history.items(), key=lambda item: (item[1], item[0]))[0]
