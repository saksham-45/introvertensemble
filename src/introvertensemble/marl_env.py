import functools
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = object
    spaces = None

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
    ):
        super().__init__()
        if ParallelEnv is object:
            raise ImportError("pettingzoo is required for MARL. Please install it.")

        self.layout_names = layout_names if isinstance(layout_names, list) else [layout_names]
        self._current_layout_name = None
        self.sim_seed = seed
        self.config = config or SimulationConfig(
            focal_agent_enabled=False,  # We handle learning agents dynamically
            events_enabled=True,
            focal_agent_external_control=True,
            all_agents_learning=True,
        )

        from pathlib import Path
        ROOT = Path(__file__).resolve().parents[2]
        self._layout_root = ROOT / "assets" / "layouts"

        self._reset_layout(seed)

        self.world = None
        self.sim = None
        self.obs_builder = None
        self.scorer = None

        self.agents = []
        self.possible_agents = []
        
        # Max agents that can exist in the environment
        self.max_num_agents = 200

        # Action space: 0 (Stay), 1-5 (Local Candidates), 6-10 (Global Candidates)
        self._action_spaces = {}
        self._observation_spaces = {}
        
        self.observation_dim = 7 + 8 + 60 + 60 + 6

        self.reset(seed=seed)

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
    def observation_space(self, agent: str) -> spaces.Box:
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Discrete:
        return spaces.Discrete(11)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[dict, dict]:
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

        # In a real setup, we want to inject learning agents. 
        # For PettingZoo, agents arrive and leave.
        self.agents = []
        self._action_spaces = {}
        self._observation_spaces = {}
        
        # Populate initial agents from sim (if any were spawned at step 0)
        self._update_agents()

        observations = {agent: self._get_agent_observation(agent) for agent in self.agents}
        infos = {agent: self._get_agent_info(agent) for agent in self.agents}

        return observations, infos

    def _update_agents(self):
        """Update the list of active agents based on the simulation state."""
        self.agents = [agent_id for agent_id, agent in self.sim.agents.items() if agent.role == "learning"]
        
    def step(self, actions: Dict[str, int]):
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        # 1. Process Actions
        conflicts = defaultdict(list)
        agent_targets = {}
        
        for agent_id, action in actions.items():
            if agent_id not in self.sim.agents:
                continue
            
            # Logic similar to single-agent Env: find target seat
            # We temporarily set focal_agent_id so observation builder works
            self.sim.focal_agent_id = agent_id
            target_seat_id = None
            if action > 0:
                obs = self.obs_builder.build_focal_observation()
                if action <= 5:
                    idx = action - 1
                    if idx < len(obs.nearby_candidates):
                        target_seat_id = obs.nearby_candidates[idx].seat_id
                else:
                    idx = action - 6
                    if idx < len(obs.global_candidates):
                        target_seat_id = obs.global_candidates[idx].seat_id
            
            if target_seat_id is not None:
                agent = self.sim.agents[agent_id]
                is_different = target_seat_id != agent.current_seat_id
                not_in_cooldown = agent.cooldown_steps_remaining <= 0
                
                if is_different and not_in_cooldown:
                    conflicts[target_seat_id].append(agent_id)
                    agent_targets[agent_id] = target_seat_id

        # Resolve conflicts randomly
        for seat_id, contending_agents in conflicts.items():
            if self.world.occupancy.get(seat_id) is not None:
                # Seat already taken by background/other agent
                for a in contending_agents:
                    rewards[a] = -1.0 # Penalty for trying to sit in occupied seat
                continue
                
            np.random.shuffle(contending_agents)
            winner = contending_agents[0]
            losers = contending_agents[1:]
            
            # Winner takes seat
            agent = self.sim.agents[winner]
            if agent.current_seat_id is not None:
                self.world.release_seat(agent.current_seat_id)
            self.world.occupy_seat(seat_id, agent.id)
            agent.record_seat(seat_id, self.spec.seats[seat_id].zone_id)
            agent.total_moves += 1
            agent.cooldown_steps_remaining = self.config.focal_move_cooldown_steps
            
            # Losers get penalty
            for loser in losers:
                rewards[loser] = -0.5

        # 2. Step Simulation
        # This will handle departures, background agent arrivals, and background reseating
        self.sim.step()
        
        # 3. Update active agents
        old_agents = set(self.agents)
        self._update_agents()
        new_agents = set(self.agents)
        
        departed_agents = old_agents - new_agents
        
        # 4. Compute Observations, Rewards, and Dones
        # We must return data for active agents AND agents that departed this step
        report_agents = set(self.agents) | departed_agents
        
        truncated_flag = self.sim.step_index >= 1000
        
        for agent_id in report_agents:
            if agent_id in departed_agents:
                terminations[agent_id] = True
                truncations[agent_id] = truncated_flag
                observations[agent_id] = np.zeros(self.observation_dim, dtype=np.float32)
                
                # Check if it was in rewards from conflict
                if agent_id not in rewards:
                    rewards[agent_id] = 0.0
                infos[agent_id] = {}
            else:
                terminations[agent_id] = False
                truncations[agent_id] = truncated_flag
                
                self.sim.focal_agent_id = agent_id
                observations[agent_id] = self._get_agent_observation(agent_id)
                infos[agent_id] = self._get_agent_info(agent_id)
                
                # Reward is seat score (plus any conflict penalties)
                agent = self.sim.agents[agent_id]
                base_reward = 0.0
                if agent.current_seat_id is not None:
                    base_reward = self.scorer.score_seat(agent, agent.current_seat_id).total
                rewards[agent_id] = rewards.get(agent_id, 0.0) + base_reward

        if truncated_flag:
            for agent_id in self.agents:
                truncations[agent_id] = True

        # PettingZoo standard: agents list is empty when env is completely done.
        # But our simulation runs continuously. We'll leave self.agents as active agents.

        return observations, rewards, terminations, truncations, infos

    def _get_agent_observation(self, agent_id: str) -> np.ndarray:
        # Re-use logic from single-agent env.py by temporarily spoofing focal_agent_id
        original_focal = self.sim.focal_agent_id
        self.sim.focal_agent_id = agent_id
        
        # This is a hacky way to use the existing logic without duplicating 100 lines
        from .env import LibraryEnv
        
        obs_vec = None
        
        # We create a dummy env just to call the method if needed, 
        # but since we can copy the logic, let's just implement a minimal version
        focal_present = agent_id in self.sim.agents
        obs = self.obs_builder.build_focal_observation()
        obs_list = []
        
        obs_list.append(1.0 if focal_present else 0.0)
        
        if focal_present:
            agent = self.sim.agents[agent_id]
            obs_list.append(1.0 if agent.current_seat_id is not None else 0.0)
            obs_list.append(agent.steps_in_current_seat / float(self.config.max_session_steps))
            obs_list.append(agent.cooldown_steps_remaining / float(self.config.focal_move_cooldown_steps))
            obs_list.append(agent.session_steps_remaining / float(self.config.focal_agent_session_steps))
            obs_list.append(obs.current_score if obs.current_score is not None else 0.0)
            
            if agent.current_seat_id is not None:
                current_privacy = self.world.feature_value_for_seat(agent.current_seat_id, "privacy")
                obs_list.append(current_privacy)
            else:
                obs_list.append(0.0)
        else:
            obs_list.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
        if focal_present and obs.current_seat is not None:
            s_obs = obs.current_seat
            obs_list.extend([
                s_obs.privacy, s_obs.noise, s_obs.interruption,
                s_obs.immediate_neighbor_ratio, s_obs.local_cluster_ratio,
                s_obs.zone_density,
                float(self.spec.seats[s_obs.seat_id].features.get("comfort", 0.5)),
                1.0 if self.spec.seats[s_obs.seat_id].features.get("outlet", False) else 0.0
            ])
        else:
            obs_list.extend([0.0] * 8)
            
        def seat_to_features(s_obs) -> list[float]:
            seat = self.spec.seats[s_obs.seat_id]
            dist = 0.0
            if focal_present:
                agent = self.sim.agents[agent_id]
                try:
                    if agent.current_seat_id is not None:
                        curr_node = self.spec.seats[agent.current_seat_id].access_node_id
                        dist = self.world.shortest_path_cost(curr_node, seat.access_node_id)
                    else:
                        ent_node = self.spec.entrances[agent.entrance_id].node_id
                        dist = self.world.shortest_path_cost(ent_node, seat.access_node_id)
                except Exception:
                    dist = 20.0
            is_pref_seat = 1.0 if (obs.preferred_seat_id == s_obs.seat_id) else 0.0
            is_pref_zone = 1.0 if (obs.preferred_zone_id == s_obs.zone_id) else 0.0
            return [
                s_obs.score, s_obs.privacy, s_obs.noise, s_obs.interruption,
                s_obs.immediate_neighbor_ratio, s_obs.local_cluster_ratio,
                s_obs.zone_density, float(seat.features.get("comfort", 0.5)),
                1.0 if seat.features.get("outlet", False) else 0.0,
                is_pref_seat, is_pref_zone, dist / 20.0
            ]

        for idx in range(5):
            if idx < len(obs.nearby_candidates):
                obs_list.extend(seat_to_features(obs.nearby_candidates[idx]))
            else:
                obs_list.extend([0.0] * 12)
                
        for idx in range(5):
            if idx < len(obs.global_candidates):
                obs_list.extend(seat_to_features(obs.global_candidates[idx]))
            else:
                obs_list.extend([0.0] * 12)
                
        obs_list.append(self.sim.current_hour / 24.0)
        total_seats = max(1, len(self.spec.seats))
        current_occupied = sum(v is not None for v in self.world.occupancy.values())
        obs_list.append(current_occupied / float(total_seats))
        
        event_names = ["Collaboration Burst", "Printer Queue", "Cafe Rush", "Quiet Room Disturbance"]
        active_event_names = [name for _, name in obs.active_events]
        for name in event_names:
            obs_list.append(1.0 if name in active_event_names else 0.0)
            
        self.sim.focal_agent_id = original_focal
        return np.array(obs_list, dtype=np.float32)

    def _get_agent_info(self, agent_id: str) -> dict:
        seated_status = False
        seat_id = None
        if agent_id in self.sim.agents:
            agent = self.sim.agents[agent_id]
            seated_status = agent.current_seat_id is not None
            seat_id = agent.current_seat_id
            
        return {
            "step_index": self.sim.step_index,
            "hour": self.sim.current_hour,
            "seated": seated_status,
            "seat_id": seat_id,
        }

    def render(self) -> None:
        pass
