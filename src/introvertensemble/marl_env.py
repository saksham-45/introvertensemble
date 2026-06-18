from __future__ import annotations

import functools
from collections import defaultdict
from dataclasses import replace
from typing import Any, Optional

import numpy as np

try:
    from pettingzoo import ParallelEnv
except ImportError:
    ParallelEnv = object


from .loader import load_layout
from .observations import ObservationBuilder
from .scoring import SeatScorer
from .simulation import LibrarySimulation, SimulationConfig
from .world import LibraryWorld


class LibraryParallelEnv(ParallelEnv if ParallelEnv is not object else object):
    metadata = {
        "render_modes": ["human"],
        "name": "library_marl_v0",
    }

    def __init__(
        self,
        layout_names: str | list[str] = "library_v1",
        config: Optional[SimulationConfig] = None,
        seed: int = 42,
        initial_learning_agent_count: int = 20,
        max_learning_agents: int = 200,
        max_episode_steps: int = 1000,
    ):
        super().__init__()
        if ParallelEnv is object:
            raise ImportError("pettingzoo is required for MARL. Please install it.")

        self.layout_names = layout_names if isinstance(layout_names, list) else [layout_names]
        self._current_layout_name = None
        self.sim_seed = seed
        self.max_episode_steps = max_episode_steps
        self._max_learning_agents = max(1, max_learning_agents)
        self.possible_agents = [f"agent_{index:04d}" for index in range(1, self._max_learning_agents + 1)]
        self.config = self._build_config(config, initial_learning_agent_count)

        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        self._layout_root = root / "assets" / "layouts"

        self._reset_layout(seed)
        self.observation_dim = 7 + 8 + 60 + 60 + 6

        self.world = None
        self.sim = None
        self.obs_builder = None
        self.scorer = None
        self.agents: list[str] = []
        self._action_spaces = {agent: self.action_space(agent) for agent in self.possible_agents}
        self._observation_spaces = {agent: self.observation_space(agent) for agent in self.possible_agents}

        self.reset(seed=seed)

    def _build_config(
        self,
        config: SimulationConfig | None,
        initial_learning_agent_count: int,
    ) -> SimulationConfig:
        if config is None:
            return SimulationConfig(
                focal_agent_enabled=False,
                events_enabled=True,
                focal_agent_external_control=True,
                all_agents_learning=True,
                learning_agent_count=max(1, initial_learning_agent_count),
            )
        return replace(
            config,
            focal_agent_enabled=False,
            focal_agent_external_control=True,
            all_agents_learning=True,
            learning_agent_count=max(config.learning_agent_count, initial_learning_agent_count),
        )

    def _reset_layout(self, seed: Optional[int] = None) -> None:
        selector_seed = seed if seed is not None else self.sim_seed
        local_random = np.random.RandomState(selector_seed)
        if self.layout_names:
            layout_index = local_random.randint(0, len(self.layout_names))
            self._current_layout_name = self.layout_names[layout_index]
        else:
            self._current_layout_name = "library_v1"

        layout_dir = self._layout_root / self._current_layout_name
        self.spec = load_layout(layout_dir)
        self.seat_ids = sorted(list(self.spec.seats.keys()))

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> Any:
        from gymnasium import spaces

        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> Any:
        from gymnasium import spaces

        return spaces.Discrete(11)

    @property
    def observation_spaces(self) -> dict[str, Any]:
        return {agent: self.observation_space(agent) for agent in self.possible_agents}

    @property
    def action_spaces(self) -> dict[str, Any]:
        return {agent: self.action_space(agent) for agent in self.possible_agents}

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        del options
        if seed is not None:
            self.sim_seed = seed
        else:
            self.sim_seed += 1

        if isinstance(self.layout_names, list) and len(self.layout_names) > 1:
            self._reset_layout(seed)

        self.world = LibraryWorld(self.spec)
        self.sim = LibrarySimulation(self.world, config=self.config, seed=self.sim_seed)
        self.obs_builder = ObservationBuilder(self.sim)
        self.scorer = SeatScorer(self.world)
        self._sync_scorer_learning_agents()
        self._update_agents()

        observations = {agent: self._get_agent_observation(agent) for agent in self.agents}
        infos = {agent: self._get_agent_info(agent) for agent in self.agents}
        return observations, infos

    def _update_agents(self) -> None:
        self.agents = [
            agent_id
            for agent_id in self.possible_agents
            if agent_id in self.sim.agents and self.sim.agents[agent_id].role in {"focal", "learning"}
        ]

    def step(
        self,
        actions: dict[str, int],
    ) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, bool], dict[str, bool], dict[str, dict[str, Any]]]:
        observations: dict[str, np.ndarray] = {}
        rewards: dict[str, float] = {}
        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}
        infos: dict[str, dict[str, Any]] = {}

        old_agents = set(self.agents)
        claims: dict[str, list[str]] = defaultdict(list)

        for agent_id in list(self.agents):
            target_seat_id = self._target_seat_for_action(agent_id, int(actions.get(agent_id, 0)))
            if target_seat_id is None:
                continue
            agent = self.sim.agents[agent_id]
            if target_seat_id == agent.current_seat_id or agent.cooldown_steps_remaining > 0:
                continue
            if self.world.occupancy.get(target_seat_id) is not None:
                rewards[agent_id] = rewards.get(agent_id, 0.0) - 1.0
                infos.setdefault(agent_id, {})["occupied_target"] = target_seat_id
                continue
            claims[target_seat_id].append(agent_id)

        winners: dict[str, str] = {}
        for seat_id, contending_agents in claims.items():
            if len(contending_agents) == 1:
                winners[contending_agents[0]] = seat_id
                continue
            np.random.shuffle(contending_agents)
            winners[contending_agents[0]] = seat_id
            for loser in contending_agents[1:]:
                rewards[loser] = rewards.get(loser, 0.0) - 0.75
                infos.setdefault(loser, {})["collision"] = seat_id

        for winner, seat_id in winners.items():
            agent = self.sim.agents[winner]
            if agent.current_seat_id is not None:
                self.world.release_seat(agent.current_seat_id)
            self.world.occupy_seat(seat_id, agent.id)
            agent.record_seat(seat_id, self.spec.seats[seat_id].zone_id)
            agent.total_moves += 1
            agent.cooldown_steps_remaining = self.config.focal_move_cooldown_steps

        summary = self.sim.step()
        self._update_agents()
        self._sync_scorer_learning_agents()

        new_agents = set(self.agents)
        departed_agents = old_agents - new_agents
        report_agents = new_agents | departed_agents
        truncated_flag = self.sim.step_index >= self.max_episode_steps

        for agent_id in sorted(report_agents):
            if agent_id in departed_agents:
                terminations[agent_id] = True
                truncations[agent_id] = truncated_flag
                observations[agent_id] = np.zeros(self.observation_dim, dtype=np.float32)
                rewards[agent_id] = rewards.get(agent_id, 0.0)
                infos.setdefault(agent_id, {})["departed"] = True
                continue

            terminations[agent_id] = False
            truncations[agent_id] = truncated_flag
            observations[agent_id] = self._get_agent_observation(agent_id)
            agent = self.sim.agents[agent_id]
            base_reward = 0.0
            if agent.current_seat_id is not None:
                base_reward = self.scorer.score_seat(agent, agent.current_seat_id).total
            rewards[agent_id] = rewards.get(agent_id, 0.0) + base_reward
            info = self._get_agent_info(agent_id)
            info["step_summary"] = summary
            infos[agent_id] = info

        if truncated_flag:
            for agent_id in self.agents:
                truncations[agent_id] = True

        return observations, rewards, terminations, truncations, infos

    def _target_seat_for_action(self, agent_id: str, action: int) -> str | None:
        if action <= 0:
            return None
        observation = self.obs_builder.build_agent_observation(agent_id)
        if action <= 5:
            candidates = observation.nearby_candidates
            index = action - 1
        else:
            candidates = observation.global_candidates
            index = action - 6
        if index >= len(candidates):
            return None
        return candidates[index].seat_id

    def _sync_scorer_learning_agents(self) -> None:
        if self.scorer is not None:
            self.scorer.set_learning_agents(set(self.agents))

    def _get_agent_observation(self, agent_id: str) -> np.ndarray:
        if agent_id not in self.sim.agents:
            return np.zeros(self.observation_dim, dtype=np.float32)

        obs = self.obs_builder.build_agent_observation(agent_id)
        agent = self.sim.agents[agent_id]
        agent_present = agent_id in self.sim.agents
        obs_list: list[float] = []

        obs_list.append(1.0 if agent_present else 0.0)
        if agent_present:
            obs_list.extend(
                [
                    1.0 if agent.current_seat_id is not None else 0.0,
                    agent.steps_in_current_seat / float(self.config.max_session_steps),
                    agent.cooldown_steps_remaining / float(self.config.focal_move_cooldown_steps),
                    agent.session_steps_remaining / float(self.config.learning_agent_session_steps),
                    obs.current_score if obs.current_score is not None else 0.0,
                ]
            )
            if agent.current_seat_id is not None:
                obs_list.append(self.world.feature_value_for_seat(agent.current_seat_id, "privacy"))
            else:
                obs_list.append(0.0)
        else:
            obs_list.extend([0.0] * 6)

        if obs.current_seat is not None:
            seat_obs = obs.current_seat
            obs_list.extend(
                [
                    seat_obs.privacy,
                    seat_obs.noise,
                    seat_obs.interruption,
                    seat_obs.immediate_neighbor_ratio,
                    seat_obs.local_cluster_ratio,
                    seat_obs.zone_density,
                    float(self.spec.seats[seat_obs.seat_id].features.get("comfort", 0.5)),
                    1.0 if self.spec.seats[seat_obs.seat_id].features.get("outlet", False) else 0.0,
                ]
            )
        else:
            obs_list.extend([0.0] * 8)

        def seat_to_features(seat_obs) -> list[float]:
            seat = self.spec.seats[seat_obs.seat_id]
            distance = 0.0
            if agent.current_seat_id is not None:
                current_node = self.spec.seats[agent.current_seat_id].access_node_id
                distance = self.world.shortest_path_cost(current_node, seat.access_node_id)
            elif agent.entrance_id in self.spec.entrances:
                entrance_node = self.spec.entrances[agent.entrance_id].node_id
                distance = self.world.shortest_path_cost(entrance_node, seat.access_node_id)
            else:
                distance = 20.0

            return [
                seat_obs.score,
                seat_obs.privacy,
                seat_obs.noise,
                seat_obs.interruption,
                seat_obs.immediate_neighbor_ratio,
                seat_obs.local_cluster_ratio,
                seat_obs.zone_density,
                float(seat.features.get("comfort", 0.5)),
                1.0 if seat.features.get("outlet", False) else 0.0,
                1.0 if obs.preferred_seat_id == seat_obs.seat_id else 0.0,
                1.0 if obs.preferred_zone_id == seat_obs.zone_id else 0.0,
                distance / 20.0,
            ]

        for index in range(5):
            if index < len(obs.nearby_candidates):
                obs_list.extend(seat_to_features(obs.nearby_candidates[index]))
            else:
                obs_list.extend([0.0] * 12)

        for index in range(5):
            if index < len(obs.global_candidates):
                obs_list.extend(seat_to_features(obs.global_candidates[index]))
            else:
                obs_list.extend([0.0] * 12)

        obs_list.extend(
            [
                self.sim.current_hour / 24.0,
                sum(value is not None for value in self.world.occupancy.values()) / float(max(1, len(self.spec.seats))),
            ]
        )
        event_names = ["Collaboration Burst", "Printer Queue", "Cafe Rush", "Quiet Room Disturbance"]
        active_event_names = [name for _, name in obs.active_events]
        for name in event_names:
            obs_list.append(1.0 if name in active_event_names else 0.0)

        return np.array(obs_list, dtype=np.float32)

    def _get_agent_info(self, agent_id: str) -> dict[str, Any]:
        if agent_id not in self.sim.agents:
            return {
                "step_index": self.sim.step_index,
                "hour": self.sim.current_hour,
                "seated": False,
                "seat_id": None,
                "profile": None,
            }

        agent = self.sim.agents[agent_id]
        return {
            "step_index": self.sim.step_index,
            "hour": self.sim.current_hour,
            "seated": agent.current_seat_id is not None,
            "seat_id": agent.current_seat_id,
            "profile": agent.profile.name,
            "total_moves": agent.total_moves,
            "cooldown": agent.cooldown_steps_remaining,
        }

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None
