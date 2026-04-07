from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from .agents import PROFILE_REGISTRY, AgentProfile, SimAgent
from .models import SpawnWindow
from .scoring import SeatScorer
from .world import LibraryWorld


@dataclass(frozen=True)
class SimulationConfig:
    step_minutes: int = 15
    introvert_share: float = 0.18
    min_session_steps: int = 4
    max_session_steps: int = 16
    reseat_margin: float = 0.20
    background_profile_mix: tuple[tuple[str, float], ...] = (
        ("standard", 0.28),
        ("quiet_seeker", 0.18),
        ("comfort_seeker", 0.16),
        ("outlet_seeker", 0.12),
        ("opportunistic_reseater", 0.14),
        ("collaborator", 0.12),
    )
    focal_agent_enabled: bool = False
    focal_agent_entrance_id: str = "E1"
    focal_agent_session_steps: int = 20
    focal_agent_initial_seat_history: tuple[str, ...] = ()
    focal_move_cooldown_steps: int = 4


@dataclass(frozen=True)
class StepSummary:
    step_index: int
    hour: float
    arrivals: int
    departures: int
    reseats: int
    occupancy: int
    zone_load: dict[str, int]


class LibrarySimulation:
    def __init__(
        self,
        world: LibraryWorld,
        config: SimulationConfig | None = None,
        seed: int = 7,
    ):
        self.world = world
        self.config = config or SimulationConfig()
        self.random = random.Random(seed)
        self.scorer = SeatScorer(world)
        self.agents: dict[str, SimAgent] = {}
        self.step_index = 0
        self.current_hour = 8.0
        self._next_agent_index = 1
        self._schedule_start_hour = min(window.start_hour for window in world.spec.spawn_windows)
        self._schedule_end_hour = max(window.end_hour for window in world.spec.spawn_windows)
        self.focal_agent_id: str | None = None
        if self.config.focal_agent_enabled:
            self._create_focal_agent()

    def step(self) -> StepSummary:
        window = self.world.active_spawn_window(self.current_hour)
        departures = self._process_departures()
        arrivals = self._spawn_arrivals(window)
        reseats = self._process_reseating(window)

        summary = StepSummary(
            step_index=self.step_index,
            hour=self.current_hour,
            arrivals=arrivals,
            departures=departures,
            reseats=reseats,
            occupancy=sum(value is not None for value in self.world.occupancy.values()),
            zone_load=self.world.current_zone_load(),
        )

        self.step_index += 1
        self.current_hour += self.config.step_minutes / 60.0
        if self.current_hour >= self._schedule_end_hour:
            overflow = self.current_hour - self._schedule_end_hour
            self.current_hour = self._schedule_start_hour + overflow
        return summary

    def _process_departures(self) -> int:
        departing_ids: list[str] = []
        for agent in self.agents.values():
            agent.session_steps_remaining -= 1
            if agent.cooldown_steps_remaining > 0:
                agent.cooldown_steps_remaining -= 1
            if agent.session_steps_remaining <= 0:
                departing_ids.append(agent.id)
        for agent_id in departing_ids:
            self._remove_agent(agent_id)
        return len(departing_ids)

    def _spawn_arrivals(self, window: SpawnWindow) -> int:
        arrivals = self._sample_rate(window.arrival_rate_per_hour)
        for _ in range(arrivals):
            agent = self._create_agent()
            seat_id = self._choose_arrival_seat(agent)
            if seat_id is None:
                continue
            self.agents[agent.id] = agent
            self.world.occupy_seat(seat_id, agent.id)
            agent.record_seat(seat_id, self.world.spec.seats[seat_id].zone_id)
        return arrivals

    def _process_reseating(self, window: SpawnWindow) -> int:
        reseat_budget = self._sample_rate(window.reseat_pressure * 6.0)
        candidate_ids = [agent_id for agent_id, agent in self.agents.items() if agent.current_seat_id is not None]
        reseats = 0

        focal_candidate_ids = [agent_id for agent_id in candidate_ids if self.agents[agent_id].role == "focal"]
        background_candidate_ids = [agent_id for agent_id in candidate_ids if self.agents[agent_id].role != "focal"]
        for agent_id in focal_candidate_ids:
            agent = self.agents[agent_id]
            if self._process_focal_reseat(agent):
                reseats += 1

        self.random.shuffle(background_candidate_ids)
        for agent_id in background_candidate_ids[:reseat_budget]:
            agent = self.agents[agent_id]
            if self._process_background_reseat(agent):
                reseats += 1
        return reseats

    def _sample_rate(self, rate_per_hour: float) -> int:
        expected = rate_per_hour * (self.config.step_minutes / 60.0)
        whole = int(expected)
        frac = expected - whole
        return whole + (1 if self.random.random() < frac else 0)

    def _create_agent(self) -> SimAgent:
        if self.random.random() < self.config.introvert_share:
            profile = AgentProfile.introvert()
        else:
            profile = self._sample_background_profile()
        entrance_id = self._sample_entrance_id(self.current_hour)
        session_steps = self.random.randint(self.config.min_session_steps, self.config.max_session_steps)
        agent_id = f"agent_{self._next_agent_index:04d}"
        self._next_agent_index += 1
        agent = SimAgent(
            id=agent_id,
            profile=profile,
            entrance_id=entrance_id,
            session_steps_remaining=session_steps,
            arrivals_step=self.step_index,
        )
        self._apply_background_defaults(agent)
        return agent

    def _create_focal_agent(self) -> SimAgent:
        agent_id = "focal_agent"
        focal = SimAgent(
            id=agent_id,
            profile=AgentProfile.focal_introvert(),
            entrance_id=self.config.focal_agent_entrance_id,
            session_steps_remaining=self.config.focal_agent_session_steps,
            role="focal",
            local_search_radius=2.4,
            arrival_acceptability_threshold=0.20,
            stay_threshold=2.60,
            leave_threshold=1.80,
            switching_improvement_threshold=0.55,
            arrivals_step=self.step_index,
        )
        for seat_id in self.config.focal_agent_initial_seat_history:
            if seat_id not in self.world.spec.seats:
                continue
            zone_id = self.world.spec.seats[seat_id].zone_id
            focal.record_seat(seat_id, zone_id)
            focal.current_seat_id = None
        seat_id = self._choose_arrival_seat(focal)
        if seat_id is not None:
            self.world.occupy_seat(seat_id, focal.id)
            focal.record_seat(seat_id, self.world.spec.seats[seat_id].zone_id)
        self.agents[focal.id] = focal
        self.focal_agent_id = focal.id
        return focal

    def _sample_entrance_id(self, hour: float) -> str:
        weights = self.world.normalized_entrance_weights(hour)
        roll = self.random.random()
        cumulative = 0.0
        last = next(iter(weights))
        for entrance_id, weight in weights.items():
            cumulative += weight
            if roll <= cumulative:
                return entrance_id
            last = entrance_id
        return last

    def _sample_background_profile(self) -> AgentProfile:
        total = sum(weight for _, weight in self.config.background_profile_mix)
        if total <= 0:
            return AgentProfile.standard()
        roll = self.random.random() * total
        cumulative = 0.0
        selected = "standard"
        for profile_name, weight in self.config.background_profile_mix:
            cumulative += weight
            if roll <= cumulative:
                selected = profile_name
                break
        factory = PROFILE_REGISTRY.get(selected, AgentProfile.standard)
        return factory()

    def _apply_background_defaults(self, agent: SimAgent) -> None:
        profile_name = agent.profile.name
        if profile_name == "quiet_seeker":
            agent.stay_threshold = 0.80
            agent.leave_threshold = -0.10
            agent.switching_improvement_threshold = 0.40
        elif profile_name == "comfort_seeker":
            agent.stay_threshold = 0.55
            agent.leave_threshold = -0.18
            agent.switching_improvement_threshold = 0.28
        elif profile_name == "outlet_seeker":
            agent.stay_threshold = 0.52
            agent.leave_threshold = -0.12
            agent.switching_improvement_threshold = 0.30
        elif profile_name == "opportunistic_reseater":
            agent.stay_threshold = 0.30
            agent.leave_threshold = 0.10
            agent.switching_improvement_threshold = 0.12
        elif profile_name == "collaborator":
            agent.stay_threshold = 0.36
            agent.leave_threshold = -0.20
            agent.switching_improvement_threshold = 0.16
        elif profile_name == "introvert":
            agent.stay_threshold = 0.78
            agent.leave_threshold = -0.02
            agent.switching_improvement_threshold = 0.42

    def _best_available_seat(
        self,
        agent: SimAgent,
        origin_entrance_id: str | None = None,
        include_current: bool = False,
        candidate_seat_ids: list[str] | None = None,
    ) -> str | None:
        best_id: str | None = None
        best_score = float("-inf")
        if candidate_seat_ids is None:
            candidate_seat_ids = [seat.id for seat in self.world.available_seats()]
        elif not include_current and agent.current_seat_id in candidate_seat_ids:
            candidate_seat_ids = [seat_id for seat_id in candidate_seat_ids if seat_id != agent.current_seat_id]

        for seat_id in candidate_seat_ids:
            if not include_current and seat_id == agent.current_seat_id:
                continue
            if seat_id != agent.current_seat_id and self.world.occupancy.get(seat_id) is not None:
                continue
            score = self.scorer.score_seat(agent, seat_id, origin_entrance_id=origin_entrance_id).total
            if score > best_score:
                best_id = seat_id
                best_score = score
        return best_id

    def _choose_arrival_seat(self, agent: SimAgent) -> str | None:
        if agent.role != "focal":
            return self._best_available_seat(agent, origin_entrance_id=agent.entrance_id)

        preferred_seat_id = agent.dominant_seat_id()
        if preferred_seat_id is not None and self.world.occupancy.get(preferred_seat_id) is None:
            return preferred_seat_id

        local_candidates = self._focal_local_candidates(agent)
        local_best = self._best_available_seat(
            agent,
            origin_entrance_id=agent.entrance_id,
            candidate_seat_ids=local_candidates,
        )
        if local_best is not None:
            local_score = self.scorer.score_seat(agent, local_best, origin_entrance_id=agent.entrance_id).total
            if local_score >= agent.arrival_acceptability_threshold:
                return local_best

        return self._best_available_seat(agent, origin_entrance_id=agent.entrance_id)

    def _focal_local_candidates(self, agent: SimAgent) -> list[str]:
        candidates: list[str] = []
        preferred_seat_id = agent.dominant_seat_id()
        preferred_zone_id = agent.dominant_zone_id()
        if preferred_seat_id is not None and preferred_seat_id in self.world.spec.seats:
            preferred_seat = self.world.spec.seats[preferred_seat_id]
            for seat in self.world.spec.seats.values():
                if self.world.occupancy[seat.id] is not None:
                    continue
                distance = ((seat.x - preferred_seat.x) ** 2 + (seat.y - preferred_seat.y) ** 2) ** 0.5
                if distance <= agent.local_search_radius:
                    candidates.append(seat.id)
        elif preferred_zone_id is not None:
            candidates.extend(
                seat.id
                for seat in self.world.available_seats(preferred_zone_id)
            )

        deduped: list[str] = []
        seen: set[str] = set()
        for seat_id in candidates:
            if seat_id not in seen:
                deduped.append(seat_id)
                seen.add(seat_id)
        return deduped

    def _process_background_reseat(self, agent: SimAgent) -> bool:
        current_seat_id = agent.current_seat_id
        assert current_seat_id is not None
        current_score = self.scorer.score_seat(agent, current_seat_id).total
        best_alt = self._best_available_seat(agent)
        if best_alt is None:
            return False
        best_score = self.scorer.score_seat(agent, best_alt).total
        threshold = self.config.reseat_margin + agent.profile.switching_aversion * 0.35
        if best_score <= current_score + threshold:
            return False
        self._move_agent(agent, best_alt)
        return True

    def _process_focal_reseat(self, agent: SimAgent) -> bool:
        current_seat_id = agent.current_seat_id
        assert current_seat_id is not None
        if agent.cooldown_steps_remaining > 0:
            return False
        current_score = self.scorer.score_seat(agent, current_seat_id).total
        if current_score >= agent.stay_threshold:
            return False

        local_candidates = self._focal_local_candidates(agent)
        threshold = agent.switching_improvement_threshold
        if current_score <= agent.leave_threshold:
            fallback = self._best_focal_fallback(agent, local_candidates)
            if fallback is None:
                return False
            self._move_agent(agent, fallback)
            return True

        if local_candidates:
            local_best = self._best_available_seat(agent, candidate_seat_ids=local_candidates)
            if local_best is not None:
                local_score = self.scorer.score_seat(agent, local_best).total
                if local_score > current_score + threshold and local_score >= agent.arrival_acceptability_threshold:
                    self._move_agent(agent, local_best)
                    return True

        best_alt = self._best_available_seat(agent)
        if best_alt is None:
            return False
        best_score = self.scorer.score_seat(agent, best_alt).total
        if best_score <= current_score + threshold:
            return False
        self._move_agent(agent, best_alt)
        return True

    def _best_focal_fallback(self, agent: SimAgent, local_candidates: list[str]) -> str | None:
        if local_candidates:
            local_best = self._best_available_seat(agent, candidate_seat_ids=local_candidates)
            if local_best is not None:
                local_score = self.scorer.score_seat(agent, local_best).total
                if local_score >= agent.arrival_acceptability_threshold:
                    return local_best

        best_alt = self._best_available_seat(agent)
        if best_alt is None:
            return None
        best_score = self.scorer.score_seat(agent, best_alt).total
        if best_score >= agent.arrival_acceptability_threshold:
            return best_alt
        return None

    def _move_agent(self, agent: SimAgent, new_seat_id: str) -> None:
        current_seat_id = agent.current_seat_id
        if current_seat_id is not None:
            self.world.release_seat(current_seat_id)
        self.world.occupy_seat(new_seat_id, agent.id)
        agent.record_seat(new_seat_id, self.world.spec.seats[new_seat_id].zone_id)
        agent.total_moves += 1
        if agent.role == "focal":
            agent.cooldown_steps_remaining = self.config.focal_move_cooldown_steps

    def _remove_agent(self, agent_id: str) -> None:
        agent = self.agents.pop(agent_id)
        if agent.current_seat_id is not None:
            self.world.release_seat(agent.current_seat_id)

    def profile_counts(self) -> dict[str, int]:
        return dict(Counter(agent.profile.name for agent in self.agents.values()))
