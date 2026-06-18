from __future__ import annotations

import unittest

from introvertensemble.marl_env import LibraryParallelEnv
from introvertensemble.simulation import SimulationConfig


class LibraryParallelEnvTests(unittest.TestCase):
    def test_conflicting_learning_agent_claims_resolve_without_duplicate_occupancy(self) -> None:
        config = SimulationConfig(
            focal_agent_enabled=False,
            events_enabled=False,
            all_agents_learning=True,
            learning_agent_count=2,
            learning_agent_session_steps=16,
            background_profile_mix=(("standard", 1.0),),
            introvert_share=0.0,
        )
        env = LibraryParallelEnv(
            config=config,
            seed=7,
            initial_learning_agent_count=2,
            max_num_agents=20,
        )
        env.reset(seed=11)
        agents = env.agents[:2]
        self.assertGreaterEqual(len(agents), 2)

        first_candidates = list(env.obs_builder.build_agent_observation(agents[0]).nearby_candidates)
        first_candidates.extend(env.obs_builder.build_agent_observation(agents[0]).global_candidates)
        second_candidate_ids = {
            candidate.seat_id
            for candidate in env.obs_builder.build_agent_observation(agents[1]).nearby_candidates
        }
        second_candidate_ids.update(
            candidate.seat_id
            for candidate in env.obs_builder.build_agent_observation(agents[1]).global_candidates
        )
        current_seats = {env.sim.agents[agent_id].current_seat_id for agent_id in agents}

        target_seat_id = None
        for candidate in first_candidates:
            if (
                candidate.seat_id in second_candidate_ids
                and candidate.seat_id not in current_seats
                and env.world.occupancy.get(candidate.seat_id) is None
            ):
                target_seat_id = candidate.seat_id
                break
        self.assertIsNotNone(target_seat_id)

        actions = {
            agents[0]: env._target_seat_action(agents[0], target_seat_id) if hasattr(env, "_target_seat_action") else self._action_for_target(env, agents[0], target_seat_id),
            agents[1]: self._action_for_target(env, agents[1], target_seat_id),
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)

        self.assertEqual(set(observations), set(env.agents))
        self.assertFalse(all(terminations.values()))
        self.assertFalse(all(truncations.values()))
        winners = [agent_id for agent_id in agents if env.sim.agents[agent_id].current_seat_id == target_seat_id]
        losers = [agent_id for agent_id in agents if agent_id not in winners]
        self.assertEqual(len(winners), 1)
        self.assertTrue(any(rewards.get(agent_id, 0.0) < 0.0 for agent_id in losers))
        self.assertEqual(env.world.occupancy[target_seat_id], winners[0])
        occupied = sum(value is not None for value in env.world.occupancy.values())
        seated_learning_agents = sum(1 for agent in env.sim.agents.values() if agent.current_seat_id is not None)
        self.assertEqual(occupied, seated_learning_agents)

    def _action_for_target(self, env: LibraryParallelEnv, agent_id: str, target_seat_id: str) -> int:
        for action in range(1, 11):
            if env._target_seat_for_action(agent_id, action) == target_seat_id:
                return action
        raise AssertionError(f"No action maps {agent_id} to {target_seat_id}")

    def test_parallel_api_smoke(self) -> None:
        try:
            from pettingzoo.test import parallel_api_test
        except ImportError as exc:
            raise unittest.SkipTest("pettingzoo is not installed") from exc

        env = LibraryParallelEnv(seed=3, initial_learning_agent_count=4, max_num_agents=30)
        parallel_api_test(env, num_cycles=20)


if __name__ == "__main__":
    unittest.main()
