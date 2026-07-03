from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass

from .agents import PROFILE_REGISTRY, AgentProfile, SimAgent
from .events import EventEngine, EventStepSummary
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
    focal_agent_enabled: bool = True
    focal_agent_count: int = 1
    focal_agent_entrance_id: str = "E1"
    focal_agent_session_steps: int = 20
    focal_agent_initial_seat_history: tuple[str, ...] = ()
    focal_move_cooldown_steps: int = 4
    learning_agent_count: int = 0
    learning_agent_session_steps: int = 16
    events_enabled: bool = True
    focal_agent_external_control: bool = False
    focal_agent_never_departs: bool = False
    # When True the focal agent arrives at a uniformly random free seat instead
    # of the scoring argmax. This makes evaluation discriminative: policies must
    # actually find good seats rather than inheriting the optimum at spawn (B3).
    focal_agent_random_spawn: bool = False
    all_agents_learning: bool = False
    marl_population_mix: tuple[tuple[str, float], ...] = (
        ("standard", 0.70),
        ("extrovert", 0.10),
        ("introvert", 0.10),
        ("neutral", 0.05),
        ("antisocial", 0.05),
    )


@dataclass(frozen=True)
class StepSummary:
    step_index: int
    hour: float
    arrivals: int
    departures: int
    reseats: int
    occupancy: int
    zone_load: dict[str, int]


@dataclass(frozen=True)
class MoveResult:
    """Outcome of an externally-requested relocation.

    Single source of truth for "move the focal/learning agent to a seat",
    shared by LibraryEnv.step, the evaluation baselines, and the oracle so the
    three cannot silently drift apart (B13).
    """

    moved: bool
    agent_id: str
    from_seat_id: str | None
    to_seat_id: str | None
    reason: str  # ok | no_agent | unknown_seat | same_seat | occupied | cooldown


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
        self.focal_agent_ids: list[str] = []
        self.learning_agent_ids: list[str] = []
        self.event_engine: EventEngine | None = EventEngine(seed=seed) if self.config.events_enabled else None
        self.last_event_summary: EventStepSummary | None = None
        if self.config.focal_agent_enabled:
            self._create_focal_agents()
        if self.config.learning_agent_count > 0:
            for _ in range(self.config.learning_agent_count):
                self._create_initial_learning_agent()

    def step(self) -> StepSummary:
        self._step_events()
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

    def _step_events(self) -> None:
        if self.event_engine is None:
            self.last_event_summary = None
            self.world.clear_dynamic_zone_layer_deltas()
            return
        self.last_event_summary = self.event_engine.step()
        self.world.set_dynamic_zone_layer_deltas(self.event_engine.zone_layer_deltas())

    def _process_departures(self) -> int:
        departing_ids: list[str] = []
        for agent in self.agents.values():
            if agent.current_seat_id is not None:
                agent.steps_in_current_seat += 1
            if agent.cooldown_steps_remaining > 0:
                agent.cooldown_steps_remaining -= 1
            if agent.role == "focal" and self.config.focal_agent_never_departs:
                continue
            agent.session_steps_remaining -= 1
            if agent.session_steps_remaining <= 0:
                departing_ids.append(agent.id)
        for agent_id in departing_ids:
            self._remove_agent(agent_id)
        self._refresh_agent_id_lists()
        return len(departing_ids)

    def _spawn_arrivals(self, window: SpawnWindow) -> int:
        arrivals = self._sample_rate(window.arrival_rate_per_hour)
        for _ in range(arrivals):
            agent = self._create_agent()
            seat_id = self._choose_arrival_seat(agent)
            if seat_id is None:
                continue
            self.agents[agent.id] = agent
            self._refresh_agent_id_lists()
            self.world.occupy_seat(seat_id, agent.id)
            agent.record_seat(seat_id, self.world.spec.seats[seat_id].zone_id)
        return arrivals

    def _process_reseating(self, window: SpawnWindow) -> int:
        reseat_budget = self._sample_rate(window.reseat_pressure * 6.0)
        candidate_ids = [agent_id for agent_id, agent in self.agents.items() if agent.current_seat_id is not None]
        reseats = 0

        focal_candidate_ids = [agent_id for agent_id in candidate_ids if self.agents[agent_id].role == "focal"]
        background_candidate_ids = [agent_id for agent_id in candidate_ids if self.agents[agent_id].role != "focal"]
        if not self.config.focal_agent_external_control:
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
        if self.config.all_agents_learning:
            profile = self._sample_profile_from_mix(self.config.marl_population_mix)
        elif self.random.random() < self.config.introvert_share:
            profile = AgentProfile.introvert()
        else:
            profile = self._sample_profile_from_mix(self.config.background_profile_mix)
        entrance_id = self._sample_entrance_id(self.current_hour)
        session_steps = self.random.randint(self.config.min_session_steps, self.config.max_session_steps)
        agent_id = f"agent_{self._next_agent_index:04d}"
        self._next_agent_index += 1
        agent = SimAgent(
            id=agent_id,
            profile=profile,
            entrance_id=entrance_id,
            session_steps_remaining=session_steps,
            role="learning" if self.config.all_agents_learning else "background",
            arrivals_step=self.step_index,
        )
        self._apply_background_defaults(agent)
        return agent

    def _create_focal_agents(self) -> list[SimAgent]:
        count = max(1, self.config.focal_agent_count)
        created: list[SimAgent] = []
        for index in range(count):
            agent_id = "focal_agent" if count == 1 else f"focal_agent_{index + 1:02d}"
            created.append(self._create_focal_agent(agent_id=agent_id))
        self._refresh_agent_id_lists()
        return created

    def _create_focal_agent(self, agent_id: str = "focal_agent") -> SimAgent:
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
        return focal

    def _create_initial_learning_agent(self) -> SimAgent:
        agent = self._create_agent()
        agent.role = "learning"
        agent.session_steps_remaining = self.config.learning_agent_session_steps
        seat_id = self._choose_arrival_seat(agent)
        if seat_id is not None:
            self.world.occupy_seat(seat_id, agent.id)
            agent.record_seat(seat_id, self.world.spec.seats[seat_id].zone_id)
        self.agents[agent.id] = agent
        self._refresh_agent_id_lists()
        return agent

    def _refresh_agent_id_lists(self) -> None:
        self.focal_agent_ids = [agent_id for agent_id, agent in self.agents.items() if agent.role == "focal"]
        self.learning_agent_ids = [agent_id for agent_id, agent in self.agents.items() if agent.role in {"focal", "learning"}]
        self.focal_agent_id = self.focal_agent_ids[0] if self.focal_agent_ids else None

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

    def _sample_profile_from_mix(self, mix: tuple[tuple[str, float], ...]) -> AgentProfile:
        total = sum(weight for _, weight in mix)
        if total <= 0:
            return AgentProfile.standard()
        roll = self.random.random() * total
        cumulative = 0.0
        selected = "standard"
        for profile_name, weight in mix:
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
        elif profile_name == "extrovert":
            agent.stay_threshold = 0.42
            agent.leave_threshold = -0.28
            agent.switching_improvement_threshold = 0.10
        elif profile_name == "neutral":
            agent.stay_threshold = 0.65
            agent.leave_threshold = -0.05
            agent.switching_improvement_threshold = 0.35
        elif profile_name == "antisocial":
            agent.stay_threshold = 0.82
            agent.leave_threshold = 0.10
            agent.switching_improvement_threshold = 0.50

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

        if self.config.focal_agent_random_spawn:
            available = self.world.available_seats()
            if not available:
                return None
            return self.random.choice(available).id

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
        current_breakdown = self.scorer.score_seat(agent, current_seat_id)
        current_score = current_breakdown.total
        current_environment_score = current_score - current_breakdown.dwell_bonus
        if current_environment_score >= agent.stay_threshold:
            return False

        local_candidates = self._focal_local_candidates(agent)
        if current_environment_score <= agent.leave_threshold:
            fallback = self._best_focal_fallback(agent, local_candidates, current_environment_score)
            if fallback is None:
                return False
            self._move_agent(agent, fallback)
            return True

        if local_candidates:
            local_best = self._best_available_seat(agent, candidate_seat_ids=local_candidates)
            if local_best is not None:
                local_score = self.scorer.score_seat(agent, local_best).total
                threshold = self._focal_required_improvement(agent, local_best)
                if local_score > current_score + threshold and local_score >= agent.arrival_acceptability_threshold:
                    self._move_agent(agent, local_best)
                    return True

        best_alt = self._best_available_seat(agent)
        if best_alt is None:
            return False
        best_score = self.scorer.score_seat(agent, best_alt).total
        threshold = self._focal_required_improvement(agent, best_alt)
        if best_score <= current_score + threshold:
            return False
        self._move_agent(agent, best_alt)
        return True

    def _best_focal_fallback(self, agent: SimAgent, local_candidates: list[str], current_score: float) -> str | None:
        del local_candidates
        best_id: str | None = None
        best_acceptability = float("-inf")
        best_score = float("-inf")
        current_acceptability = self.scorer.fallback_acceptability_score(agent, agent.current_seat_id)
        for seat in self.world.available_seats():
            seat_id = seat.id
            if not self._is_focal_fallback_acceptable(agent, seat_id):
                continue
            candidate_acceptability = self.scorer.fallback_acceptability_score(agent, seat_id)
            required_acceptability_gain = self._focal_required_acceptability_gain(agent, seat_id)
            if candidate_acceptability <= current_acceptability + required_acceptability_gain:
                continue
            candidate_score = self.scorer.score_seat(agent, seat_id).total
            threshold = self._focal_required_improvement(agent, seat_id, urgent=True)
            if candidate_score <= current_score + threshold and candidate_acceptability <= current_acceptability + required_acceptability_gain + 0.25:
                continue
            if (candidate_acceptability, candidate_score) > (best_acceptability, best_score):
                best_id = seat_id
                best_acceptability = candidate_acceptability
                best_score = candidate_score
        return best_id

    def _is_focal_fallback_acceptable(self, agent: SimAgent, seat_id: str) -> bool:
        score = self.scorer.score_seat(agent, seat_id).total
        acceptability = self.scorer.fallback_acceptability_score(agent, seat_id)
        seat = self.world.spec.seats[seat_id]
        immediate = self.world.immediate_neighbor_ratio(seat_id)
        interruption = self.world.feature_value_for_seat(seat_id, "interruption_risk")
        local = self.world.local_crowding_ratio(seat_id)
        zone_density = self.world.zone_density(seat.zone_id)
        return (
            score >= agent.arrival_acceptability_threshold
            and acceptability >= 0.10
            and immediate <= 0.67
            and interruption <= 0.80
            and local <= 0.95
            and zone_density <= 0.98
        )

    def _focal_required_improvement(self, agent: SimAgent, candidate_seat_id: str, urgent: bool = False) -> float:
        required = agent.switching_improvement_threshold
        required += min(0.18, 0.02 * agent.steps_in_current_seat)
        required += self._same_zone_micro_move_penalty(agent, candidate_seat_id)
        if urgent:
            return max(0.12, required * 0.40)
        return required

    def _focal_required_acceptability_gain(self, agent: SimAgent, candidate_seat_id: str) -> float:
        required = 0.08 + min(0.10, 0.015 * agent.steps_in_current_seat)
        required += 0.20 * self._same_zone_micro_move_penalty(agent, candidate_seat_id)
        return required

    def _same_zone_micro_move_penalty(self, agent: SimAgent, candidate_seat_id: str) -> float:
        current_seat_id = agent.current_seat_id
        if current_seat_id is None or current_seat_id == candidate_seat_id:
            return 0.0
        current_seat = self.world.spec.seats[current_seat_id]
        candidate_seat = self.world.spec.seats[candidate_seat_id]
        if current_seat.zone_id != candidate_seat.zone_id:
            return 0.0
        distance = math.dist((current_seat.x, current_seat.y), (candidate_seat.x, candidate_seat.y))
        if distance <= 1.6:
            return 0.65
        if distance <= 3.0:
            return 0.30
        return 0.10

    def resolve_seat_claims(self, claims: dict[str, str]) -> dict[str, float]:
        rewards: dict[str, float] = {}
        unoccupied_claims: dict[str, list[str]] = {}

        for agent_id, seat_id in claims.items():
            agent = self.agents.get(agent_id)
            if agent is None or seat_id not in self.world.spec.seats:
                rewards[agent_id] = rewards.get(agent_id, 0.0) - 1.0
                continue
            if agent.cooldown_steps_remaining > 0:
                continue
            if self.world.occupancy.get(seat_id) is not None:
                rewards[agent_id] = rewards.get(agent_id, 0.0) - 1.0
                continue
            if seat_id == agent.current_seat_id:
                continue
            unoccupied_claims.setdefault(seat_id, []).append(agent_id)

        winners: dict[str, str] = {}
        for seat_id, agent_ids in unoccupied_claims.items():
            if len(agent_ids) == 1:
                winners[agent_ids[0]] = seat_id
                continue
            self.random.shuffle(agent_ids)
            winners[agent_ids[0]] = seat_id
            for loser in agent_ids[1:]:
                rewards[loser] = rewards.get(loser, 0.0) - 0.75

        for agent_id, seat_id in winners.items():
            self._move_agent(self.agents[agent_id], seat_id)
        self._refresh_agent_id_lists()
        return rewards

    def apply_external_move(
        self,
        agent_id: str,
        seat_id: str,
        *,
        ignore_cooldown: bool = False,
    ) -> MoveResult:
        """Relocate ``agent_id`` to ``seat_id`` on behalf of an external controller.

        Returns a :class:`MoveResult` describing whether the move happened and,
        if not, why. ``ignore_cooldown`` lets an upper-bound oracle re-seat every
        step; ordinary policies (and the RL env) leave it False so the move
        cooldown is respected.
        """
        agent = self.agents.get(agent_id)
        if agent is None:
            return MoveResult(False, agent_id, None, seat_id, "no_agent")
        from_seat_id = agent.current_seat_id
        if seat_id not in self.world.spec.seats:
            return MoveResult(False, agent_id, from_seat_id, seat_id, "unknown_seat")
        if seat_id == from_seat_id:
            return MoveResult(False, agent_id, from_seat_id, seat_id, "same_seat")
        if self.world.occupancy.get(seat_id) is not None:
            return MoveResult(False, agent_id, from_seat_id, seat_id, "occupied")
        if not ignore_cooldown and agent.cooldown_steps_remaining > 0:
            return MoveResult(False, agent_id, from_seat_id, seat_id, "cooldown")
        self._move_agent(agent, seat_id)
        return MoveResult(True, agent_id, from_seat_id, seat_id, "ok")

    def move_path_cost(self, from_seat_id: str | None, to_seat_id: str, agent: SimAgent) -> float:
        """Graph path cost of a relocation, normalized to [0, 1] by max path cost.

        From an occupied seat we use seat-to-seat access-node cost; when the
        agent is arriving (no current seat) we use entrance-to-seat cost.
        """
        max_cost = self.scorer._max_path_cost or 1.0
        if from_seat_id is not None:
            from_node = self.world.spec.seats[from_seat_id].access_node_id
            to_node = self.world.spec.seats[to_seat_id].access_node_id
            return self.world.shortest_path_cost(from_node, to_node) / max_cost
        return self.world.path_cost_from_entrance_to_seat(agent.entrance_id, to_seat_id) / max_cost

    def _move_agent(self, agent: SimAgent, new_seat_id: str) -> None:
        current_seat_id = agent.current_seat_id
        if current_seat_id is not None:
            self.world.release_seat(current_seat_id)
        self.world.occupy_seat(new_seat_id, agent.id)
        agent.record_seat(new_seat_id, self.world.spec.seats[new_seat_id].zone_id)
        agent.total_moves += 1
        if agent.role in {"focal", "learning"}:
            agent.cooldown_steps_remaining = self.config.focal_move_cooldown_steps

    def _remove_agent(self, agent_id: str) -> None:
        agent = self.agents.pop(agent_id)
        if agent.current_seat_id is not None:
            self.world.release_seat(agent.current_seat_id)

    def profile_counts(self) -> dict[str, int]:
        return dict(Counter(agent.profile.name for agent in self.agents.values()))
