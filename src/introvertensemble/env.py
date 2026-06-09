from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    # Fallback to allow other modules to import without gymnasium installed
    HAS_GYMNASIUM = False
    gym = object
    spaces = None

from .loader import load_layout
from .world import LibraryWorld
from .simulation import LibrarySimulation, SimulationConfig
from .scoring import SeatScorer
from .observations import ObservationBuilder


class LibraryEnv(gym.Env if HAS_GYMNASIUM else object):
    """
    Gymnasium environment wrapper for introvertensemble.
    Allows training a single focal agent using reinforcement learning
    against a population of rule-based background agents.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        layout_names: str | list[str] = "library_v1",
        config: SimulationConfig | None = None,
        seed: int = 42,
    ):
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium is required to use LibraryEnv. Please install it.")
            
        super().__init__()
        
        # Store layout names for randomization
        self.layout_names = layout_names if isinstance(layout_names, list) else [layout_names]
        self._current_layout_name = None
        
        # Load layout spec
        from pathlib import Path
        ROOT = Path(__file__).resolve().parents[2]
        self._layout_root = ROOT / "assets" / "layouts"
        
        # Default config with focal agent enabled and under external control
        if config is None:
            self.config = SimulationConfig(
                focal_agent_enabled=True,
                focal_agent_initial_seat_history=(),
                focal_agent_session_steps=40,
                events_enabled=True,
                focal_agent_external_control=True,
            )
        else:
            # Override to ensure focal agent is enabled and externally controlled
            self.config = SimulationConfig(
                step_minutes=config.step_minutes,
                introvert_share=config.introvert_share,
                min_session_steps=config.min_session_steps,
                max_session_steps=config.max_session_steps,
                reseat_margin=config.reseat_margin,
                background_profile_mix=config.background_profile_mix,
                focal_agent_enabled=True,
                focal_agent_entrance_id=config.focal_agent_entrance_id,
                focal_agent_session_steps=config.focal_agent_session_steps,
                focal_agent_initial_seat_history=config.focal_agent_initial_seat_history,
                focal_move_cooldown_steps=config.focal_move_cooldown_steps,
                events_enabled=config.events_enabled,
                focal_agent_external_control=True,
                focal_agent_never_departs=config.focal_agent_never_departs,
            )
            
        self.initial_seed = seed
        self.sim_seed = seed
        self.world = None
        self.sim = None
        self.obs_builder = None
        self.scorer = None
        
        # Initialize with first layout
        self._reset_layout(seed)
        
        # Sort seat IDs deterministically
        self.seat_ids = sorted(list(self.spec.seats.keys()))
        self.num_seats = len(self.seat_ids)
        self.seat_to_index = {seat_id: idx for idx, seat_id in enumerate(self.seat_ids)}
        
        # Define action space:
        # 0: Stay put / No-op
        # 1 to 5: Relocate to nearby candidate 1 to 5
        # 6 to 10: Relocate to global candidate 1 to 5
        self.action_space = spaces.Discrete(11)
        
        # Define observation space:
        # Flat continuous vector of size 141:
        # - Focal agent state (7 features)
        # - Current seat quality (8 features)
        # - Nearby candidates (5 seats * 12 features = 60 features)
        # - Global candidates (5 seats * 12 features = 60 features)
        # - General environment/time context (6 features)
        self.observation_dim = 7 + 8 + 60 + 60 + 6
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )
        
        self.reset(seed=seed)

    def _reset_layout(self, seed: int | None = None) -> None:
        """Reset to a layout, optionally using seed for deterministic selection."""
        # Select layout randomly from the list, using seed for reproducibility
        if self.layout_names:
            # Use the provided seed or sim_seed for layout selection to ensure reproducibility
            selector_seed = seed if seed is not None else self.sim_seed
            local_random = np.random.RandomState(selector_seed)
            layout_index = local_random.randint(0, len(self.layout_names))
            self._current_layout_name = self.layout_names[layout_index]
        else:
            # Fallback to first layout if list is empty
            self._current_layout_name = self.layout_names[0] if self.layout_names else "library_v1"
        
        # Load layout spec
        layout_dir = self._layout_root / self._current_layout_name
        self.spec = load_layout(layout_dir)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.sim_seed = seed
        else:
            self.sim_seed += 1
            
        # Reset layout if we're using multiple layouts
        if isinstance(self.layout_names, list) and len(self.layout_names) > 1:
            self._reset_layout(seed)
        
        self.world = LibraryWorld(self.spec)
        self.sim = LibrarySimulation(self.world, config=self.config, seed=self.sim_seed)
        self.obs_builder = ObservationBuilder(self.sim)
        self.scorer = SeatScorer(self.world)
        
        obs_vec = self._get_observation()
        info = self._get_info()
        return obs_vec, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        focal_id = self.sim.focal_agent_id
        focal_present = focal_id is not None and focal_id in self.sim.agents
        
        # Determine target seat based on candidate lists
        target_seat_id = None
        if focal_present and action > 0:
            obs = self.obs_builder.build_focal_observation()
            if action <= 5:
                # Nearby candidate (1-based index)
                idx = action - 1
                if idx < len(obs.nearby_candidates):
                    target_seat_id = obs.nearby_candidates[idx].seat_id
            else:
                # Global candidate (1-based index)
                idx = action - 6
                if idx < len(obs.global_candidates):
                    target_seat_id = obs.global_candidates[idx].seat_id
                    
        # Apply external action to the focal agent if a valid unoccupied target seat was selected
        if focal_present and target_seat_id is not None:
            agent = self.sim.agents[focal_id]
            is_different = target_seat_id != agent.current_seat_id
            is_unoccupied = self.world.occupancy.get(target_seat_id) is None
            not_in_cooldown = agent.cooldown_steps_remaining <= 0
            
            if is_different and is_unoccupied and not_in_cooldown:
                # Release current seat if seated
                if agent.current_seat_id is not None:
                    self.world.release_seat(agent.current_seat_id)
                
                # Occupy target seat
                self.world.occupy_seat(target_seat_id, agent.id)
                agent.record_seat(target_seat_id, self.spec.seats[target_seat_id].zone_id)
                agent.total_moves += 1
                agent.cooldown_steps_remaining = self.config.focal_move_cooldown_steps
        
        # Advance simulation one step
        summary = self.sim.step()
        
        # Get new observation
        obs_vec = self._get_observation()
        
        # Determine reward
        reward = 0.0
        terminated = False
        
        # Check if the focal agent is still in the library
        focal_active = focal_id is not None and focal_id in self.sim.agents
        if focal_active:
            agent = self.sim.agents[focal_id]
            if agent.current_seat_id is not None:
                # The reward is the current step seat score of the agent
                reward = self.scorer.score_seat(agent, agent.current_seat_id).total
        else:
            # Focal agent departed (session ended)
            terminated = True
            
        truncated = False
        if self.sim.step_index >= 1000:  # Safety limit for truncation
            truncated = True
            
        info = self._get_info()
        info["step_summary"] = summary
        
        return obs_vec, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        focal_id = self.sim.focal_agent_id
        focal_present = focal_id is not None and focal_id in self.sim.agents
        
        obs = self.obs_builder.build_focal_observation()
        obs_list = []
        
        # 1. Focal Agent State (7 features)
        obs_list.append(1.0 if focal_present else 0.0)
        
        if focal_present:
            agent = self.sim.agents[focal_id]
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
            
        # 2. Current Seat Quality (8 features)
        if focal_present and obs.current_seat is not None:
            s_obs = obs.current_seat
            obs_list.extend([
                s_obs.privacy,
                s_obs.noise,
                s_obs.interruption,
                s_obs.immediate_neighbor_ratio,
                s_obs.local_cluster_ratio,
                s_obs.zone_density,
                float(self.spec.seats[s_obs.seat_id].features.get("comfort", 0.5)),
                1.0 if self.spec.seats[s_obs.seat_id].features.get("outlet", False) else 0.0
            ])
        else:
            obs_list.extend([0.0] * 8)
            
        # Helper to convert SeatObservation to features
        def seat_to_features(s_obs) -> list[float]:
            seat = self.spec.seats[s_obs.seat_id]
            
            # Distance from current seat (or entrance if arriving) to candidate
            dist = 0.0
            if focal_present:
                agent = self.sim.agents[focal_id]
                try:
                    if agent.current_seat_id is not None:
                        curr_node = self.spec.seats[agent.current_seat_id].access_node_id
                        dist = self.world.shortest_path_cost(curr_node, seat.access_node_id)
                    else:
                        ent_node = self.spec.entrances[agent.entrance_id].node_id
                        dist = self.world.shortest_path_cost(ent_node, seat.access_node_id)
                except Exception:
                    dist = 20.0  # Max bounding box fallback
            
            is_pref_seat = 1.0 if (obs.preferred_seat_id == s_obs.seat_id) else 0.0
            is_pref_zone = 1.0 if (obs.preferred_zone_id == s_obs.zone_id) else 0.0
            
            return [
                s_obs.score,
                s_obs.privacy,
                s_obs.noise,
                s_obs.interruption,
                s_obs.immediate_neighbor_ratio,
                s_obs.local_cluster_ratio,
                s_obs.zone_density,
                float(seat.features.get("comfort", 0.5)),
                1.0 if seat.features.get("outlet", False) else 0.0,
                is_pref_seat,
                is_pref_zone,
                dist / 20.0  # Normalized distance
            ]

        # 3. Nearby Candidates (5 seats * 12 features = 60 features)
        for idx in range(5):
            if idx < len(obs.nearby_candidates):
                obs_list.extend(seat_to_features(obs.nearby_candidates[idx]))
            else:
                obs_list.extend([0.0] * 12)
                
        # 4. Global Candidates (5 seats * 12 features = 60 features)
        for idx in range(5):
            if idx < len(obs.global_candidates):
                obs_list.extend(seat_to_features(obs.global_candidates[idx]))
            else:
                obs_list.extend([0.0] * 12)
                
        # 5. General Environment/Time Context (6 features)
        obs_list.append(self.sim.current_hour / 24.0)
        total_seats = max(1, len(self.spec.seats))
        current_occupied = sum(v is not None for v in self.world.occupancy.values())
        obs_list.append(current_occupied / float(total_seats))
        
        # 4 event flags
        event_names = ["Collaboration Burst", "Printer Queue", "Cafe Rush", "Quiet Room Disturbance"]
        active_event_names = [name for _, name in obs.active_events]
        for name in event_names:
            obs_list.append(1.0 if name in active_event_names else 0.0)
            
        return np.array(obs_list, dtype=np.float32)

    def _get_info(self) -> dict[str, Any]:
        focal_id = self.sim.focal_agent_id
        seated_status = False
        seat_id = None
        if focal_id is not None and focal_id in self.sim.agents:
            agent = self.sim.agents[focal_id]
            seated_status = agent.current_seat_id is not None
            seat_id = agent.current_seat_id
            
        return {
            "step_index": self.sim.step_index,
            "hour": self.sim.current_hour,
            "focal_present": focal_id is not None and focal_id in self.sim.agents,
            "focal_seated": seated_status,
            "focal_seat_id": seat_id,
        }

    def render(self) -> None:
        pass
