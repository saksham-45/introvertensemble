from __future__ import annotations

import unittest
import numpy as np
import gymnasium as gym

from introvertensemble import LibraryEnv
from introvertensemble.simulation import SimulationConfig

class TestLibraryEnv(unittest.TestCase):
    def test_env_initialization(self) -> None:
        env = LibraryEnv(seed=42)
        self.assertEqual(env.observation_space.shape, (141,))
        self.assertEqual(env.action_space.n, 11)
        
    def test_env_reset(self) -> None:
        env = LibraryEnv(seed=42)
        obs, info = env.reset(seed=100)
        
        self.assertEqual(obs.shape, (141,))
        self.assertTrue(isinstance(info, dict))
        self.assertTrue(info["focal_present"])
        self.assertTrue(info["focal_seated"])
        self.assertIsNotNone(info["focal_seat_id"])
        
    def test_env_step_noop(self) -> None:
        env = LibraryEnv(seed=42)
        obs, info = env.reset(seed=100)
        initial_seat_id = info["focal_seat_id"]
        
        # Take action 0 (No-op)
        next_obs, reward, terminated, truncated, next_info = env.step(0)
        
        self.assertEqual(next_obs.shape, (141,))
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(next_info["focal_seat_id"], initial_seat_id)
        # Verify that reward is greater than 0 since the agent was seated in a good quiet seat
        self.assertGreater(reward, 0.0)

    def test_env_step_relocate(self) -> None:
        env = LibraryEnv(seed=42)
        obs, info = env.reset(seed=100)
        initial_seat_id = info["focal_seat_id"]
        
        # Find first nearby candidate from observation builder
        focal_obs = env.obs_builder.build_focal_observation()
        self.assertGreater(len(focal_obs.nearby_candidates), 0, "No nearby candidates found")
        target_seat_id = focal_obs.nearby_candidates[0].seat_id
        
        # Step with relocate action (action 1 maps to nearby candidate 0)
        next_obs, reward, terminated, truncated, next_info = env.step(1)
        
        self.assertEqual(next_info["focal_seat_id"], target_seat_id)
        self.assertNotEqual(next_info["focal_seat_id"], initial_seat_id)

    def test_env_termination(self) -> None:
        # Create config with very short session length for focal agent
        config = SimulationConfig(
            focal_agent_session_steps=2,
        )
        env = LibraryEnv(config=config, seed=42)
        obs, info = env.reset()
        
        # Step 1
        _, _, terminated, _, _ = env.step(0)
        self.assertFalse(terminated)
        
        # Step 2: Focal agent session steps run out (total steps = 2)
        _, _, terminated, _, _ = env.step(0)
        self.assertTrue(terminated)

    def test_focal_never_departs(self) -> None:
        config = SimulationConfig(
            focal_agent_session_steps=2,
            focal_agent_never_departs=True,
        )
        env = LibraryEnv(config=config, seed=42)
        env.reset(seed=100)
        focal_id = env.sim.focal_agent_id

        for _ in range(6):
            _, _, terminated, _, _ = env.step(0)
            self.assertFalse(terminated)
            self.assertIn(focal_id, env.sim.agents)

if __name__ == "__main__":
    unittest.main()
