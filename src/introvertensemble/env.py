from __future__ import annotations

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
from .observations import ObservationBuilder
from .scoring import SeatScorer
from .simulation import LibrarySimulation, SimulationConfig
from .world import LibraryWorld


class LibraryEnv(gym.Env if HAS_GYMNASIUM else object):
    """
    Gymnasium environment wrapper for introvertensemble.
    Allows training a single focal agent using reinforcement learning
    against a population of rule-based background agents.
    """
    metadata = {"render_modes": ["human"]}

    #: Reward specifications. ``legacy`` reproduces the original (gameable)
    #: behaviour where reward == raw seat score, including farmable familiarity
    #: and dwell bonuses on cost-free moves. ``environment`` is the corrected
    #: RL reward: it strips the habit terms (they remain in NPC utility) and
    #: charges the realized movement cost on the step a relocation happens (B4).
    REWARD_MODES = ("legacy", "environment")

    def __init__(
        self,
        layout_names: str | list[str] = "library_v1",
        config: SimulationConfig | None = None,
        seed: int = 42,
        reward_mode: str = "legacy",
        score_in_obs: bool = True,
        move_cost_scale: float = 1.5,
        max_episode_steps: int = 1000,
    ):
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium is required to use LibraryEnv. Please install it.")

        super().__init__()

        if reward_mode not in self.REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {self.REWARD_MODES}, got {reward_mode!r}")
        self.reward_mode = reward_mode
        self.score_in_obs = score_in_obs
        self.move_cost_scale = float(move_cost_scale)
        self.max_episode_steps = int(max_episode_steps)

        # Store layout names for randomization
        self.layout_names = layout_names if isinstance(layout_names, list) else [layout_names]
        self._current_layout_name = None
        # Dedicated RNG for layout selection, independent of the simulation seed
        # so that domain randomization does not couple to sim dynamics (B11).
        self._layout_rng = np.random.default_rng(seed)

        # Layout assets are resolved through the package helper so the env works
        # from any working directory / installed location (B10).
        from .loader import default_layout_root
        self._layout_root = default_layout_root()

        # The env always drives the focal agent externally; every other config
        # field (including future ones) is preserved via replace() rather than a
        # lossy field-by-field copy.
        from dataclasses import replace
        if config is None:
            self.config = SimulationConfig(
                focal_agent_enabled=True,
                focal_agent_initial_seat_history=(),
                focal_agent_session_steps=40,
                events_enabled=True,
                focal_agent_external_control=True,
            )
        else:
            self.config = replace(
                config,
                focal_agent_enabled=True,
                focal_agent_external_control=True,
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
        self.seat_ids = sorted(list(self.layout_spec.seats.keys()))
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
        # Finite bounds (B12): features are normalized ratios in [0, 1] and small
        # signed scores; observations are clipped to this range in
        # _get_observation so the Gymnasium env checker's containment test holds.
        self.obs_low, self.obs_high = -20.0, 20.0
        self.observation_space = spaces.Box(
            low=self.obs_low,
            high=self.obs_high,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

        self.reset(seed=seed)

    def _reset_layout(self, seed: int | None = None) -> None:
        """Select and load the active layout for the next episode.

        Layout choice is drawn from a dedicated RNG (``self._layout_rng``) so it
        is reproducible from the constructor seed yet independent of the
        simulation's own randomness (B11). A single-element layout list is a
        no-op selection.
        """
        if not self.layout_names:
            raise ValueError("layout_names must be non-empty")
        if seed is not None:
            # A caller-supplied seed re-anchors the layout stream so that
            # reset(seed=s) is fully reproducible.
            self._layout_rng = np.random.default_rng(seed)
        layout_index = int(self._layout_rng.integers(0, len(self.layout_names)))
        self._current_layout_name = self.layout_names[layout_index]
        self.layout_spec = load_layout(self._layout_root / self._current_layout_name)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        # Seed the Gymnasium base RNG (self.np_random) per the API contract (B12).
        super().reset(seed=seed)
        if seed is not None:
            self.sim_seed = seed
        else:
            self.sim_seed += 1

        # Re-draw the layout each episode when randomizing over several layouts.
        if len(self.layout_names) > 1:
            self._reset_layout(seed)

        self.world = LibraryWorld(self.layout_spec)
        self.sim = LibrarySimulation(self.world, config=self.config, seed=self.sim_seed)
        self.obs_builder = ObservationBuilder(self.sim)
        self.scorer = SeatScorer(self.world)
        
        obs_vec = self._get_observation()
        info = self._get_info()
        return obs_vec, info

    def _action_to_seat(self, action: int) -> str | None:
        """Map a discrete action index to a candidate seat id (None == stay)."""
        if action <= 0:
            return None
        obs = self.obs_builder.build_focal_observation()
        if action <= 5:
            idx = action - 1
            if idx < len(obs.nearby_candidates):
                return obs.nearby_candidates[idx].seat_id
        else:
            idx = action - 6
            if idx < len(obs.global_candidates):
                return obs.global_candidates[idx].seat_id
        return None

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        focal_id = self.sim.focal_agent_id
        focal_present = focal_id is not None and focal_id in self.sim.agents

        # Resolve and apply the relocation through the single shared move path
        # (B13). Capture the normalized path cost so we can charge it as a
        # realized movement penalty this step (B4).
        realized_move_cost = 0.0
        target_seat_id = self._action_to_seat(action) if focal_present else None
        if focal_present and target_seat_id is not None:
            agent = self.sim.agents[focal_id]
            move_cost = self.sim.move_path_cost(agent.current_seat_id, target_seat_id, agent)
            result = self.sim.apply_external_move(focal_id, target_seat_id)
            if result.moved:
                realized_move_cost = move_cost

        # Advance simulation one step
        summary = self.sim.step()
        obs_vec = self._get_observation()

        # Compute reward from the resulting seat, according to reward_mode.
        reward = 0.0
        reward_components: dict[str, float] = {}
        terminated = False
        focal_active = focal_id is not None and focal_id in self.sim.agents
        if focal_active:
            agent = self.sim.agents[focal_id]
            if agent.current_seat_id is not None:
                breakdown = self.scorer.score_seat(agent, agent.current_seat_id)
                reward = self._reward_from_breakdown(breakdown, realized_move_cost, reward_components)
        else:
            # Focal agent departed (session ended)
            terminated = True

        truncated = self.sim.step_index >= self.max_episode_steps

        info = self._get_info()
        info["step_summary"] = summary
        info["realized_move_cost"] = realized_move_cost
        info["moved"] = realized_move_cost > 0.0
        info["reward_components"] = reward_components
        info["reward_mode"] = self.reward_mode

        return obs_vec, reward, terminated, truncated, info

    def _reward_from_breakdown(
        self,
        breakdown,
        realized_move_cost: float,
        components: dict[str, Any],
    ) -> float:
        """Turn a SeatScoreBreakdown into the scalar RL reward for this mode.

        - ``legacy``: reward == raw seat score (reproduces the original, gameable
          objective where familiarity/dwell accrue for free — kept for RQ2).
        - ``environment``: strip the habit terms (they still drive NPC utility)
          and subtract the realized movement cost, so relocation is a genuine
          tradeoff and seat-hopping to farm familiarity no longer pays (B4).
        """
        # Expose every utility component for analysis / reward-hacking plots.
        for field_name in breakdown.__dataclass_fields__:
            components[field_name] = float(getattr(breakdown, field_name))
        components["realized_move_cost"] = float(realized_move_cost)
        components["reward_mode"] = self.reward_mode

        if self.reward_mode == "legacy":
            reward = breakdown.total
        else:  # "environment"
            reward = breakdown.total - breakdown.familiarity - breakdown.dwell_bonus
            reward -= self.move_cost_scale * realized_move_cost
        components["reward"] = float(reward)
        return float(reward)

    def _get_observation(self) -> np.ndarray:
        focal_id = self.sim.focal_agent_id
        focal_present = focal_id is not None and focal_id in self.sim.agents
        
        obs = self.obs_builder.build_focal_observation()
        obs_list = []

        # When score_in_obs is False the true seat score (which equals the
        # reward) is withheld so the task is not a disguised bandit (B5); the
        # policy must infer quality from raw features instead.
        score_gate = 1.0 if self.score_in_obs else 0.0

        # 1. Focal Agent State (7 features)
        obs_list.append(1.0 if focal_present else 0.0)
        
        if focal_present:
            agent = self.sim.agents[focal_id]
            obs_list.append(1.0 if agent.current_seat_id is not None else 0.0)
            obs_list.append(agent.steps_in_current_seat / float(self.config.max_session_steps))
            obs_list.append(agent.cooldown_steps_remaining / float(self.config.focal_move_cooldown_steps))
            obs_list.append(agent.session_steps_remaining / float(self.config.focal_agent_session_steps))
            obs_list.append(score_gate * (obs.current_score if obs.current_score is not None else 0.0))
            
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
                float(self.layout_spec.seats[s_obs.seat_id].features.get("comfort", 0.5)),
                1.0 if self.layout_spec.seats[s_obs.seat_id].features.get("outlet", False) else 0.0
            ])
        else:
            obs_list.extend([0.0] * 8)
            
        # Helper to convert SeatObservation to features
        def seat_to_features(s_obs) -> list[float]:
            seat = self.layout_spec.seats[s_obs.seat_id]
            
            # Distance from current seat (or entrance if arriving) to candidate
            dist = 0.0
            if focal_present:
                agent = self.sim.agents[focal_id]
                try:
                    if agent.current_seat_id is not None:
                        curr_node = self.layout_spec.seats[agent.current_seat_id].access_node_id
                        dist = self.world.shortest_path_cost(curr_node, seat.access_node_id)
                    else:
                        ent_node = self.layout_spec.entrances[agent.entrance_id].node_id
                        dist = self.world.shortest_path_cost(ent_node, seat.access_node_id)
                except Exception:
                    dist = 20.0  # Max bounding box fallback
            
            is_pref_seat = 1.0 if (obs.preferred_seat_id == s_obs.seat_id) else 0.0
            is_pref_zone = 1.0 if (obs.preferred_zone_id == s_obs.zone_id) else 0.0
            
            return [
                score_gate * s_obs.score,
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
        total_seats = max(1, len(self.layout_spec.seats))
        current_occupied = sum(v is not None for v in self.world.occupancy.values())
        obs_list.append(current_occupied / float(total_seats))
        
        # 4 event flags
        event_names = ["Collaboration Burst", "Printer Queue", "Cafe Rush", "Quiet Room Disturbance"]
        active_event_names = [name for _, name in obs.active_events]
        for name in event_names:
            obs_list.append(1.0 if name in active_event_names else 0.0)

        vec = np.array(obs_list, dtype=np.float32)
        # Guarantee the observation lies within the declared Box (B12).
        return np.clip(vec, self.obs_low, self.obs_high)

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
