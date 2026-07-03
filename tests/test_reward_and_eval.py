"""Regression tests locking in the research-hardening bug fixes.

Each test names the bug id from docs/RESEARCH_ENGINEERING_PLAN.md that it guards.
"""
from __future__ import annotations

import statistics
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import gymnasium  # noqa: F401
    HAS_GYM = True
except ImportError:
    HAS_GYM = False

if HAS_GYM:
    from introvertensemble import LibraryEnv
    from introvertensemble.simulation import SimulationConfig


def _eval_config(session_steps: int = 24) -> SimulationConfig:
    return SimulationConfig(
        focal_agent_enabled=True,
        focal_agent_external_control=True,
        focal_agent_session_steps=session_steps,
        focal_agent_never_departs=True,
        events_enabled=True,
    )


def _run(policy, seed: int, mode: str, steps: int = 24) -> float:
    env = LibraryEnv(seed=seed, config=_eval_config(steps), reward_mode=mode)
    env.reset(seed=seed)
    total = 0.0
    for t in range(steps):
        _, reward, terminated, truncated, _ = env.step(policy(t))
        total += reward
        if terminated or truncated:
            break
    return total


@unittest.skipUnless(HAS_GYM, "gymnasium not installed")
class TestRewardHardening(unittest.TestCase):
    def test_env_checker_passes(self) -> None:
        """B12: env satisfies the Gymnasium API contract."""
        from gymnasium.utils.env_checker import check_env
        env = LibraryEnv(seed=1, reward_mode="environment", score_in_obs=False)
        check_env(env, skip_render_check=True)

    def test_pingpong_farmer_loses_to_noop(self) -> None:
        """B4: seat-hopping to farm familiarity must not beat sitting still.

        The core exploit the corrected reward closes: in environment mode,
        relocation is charged and habit bonuses are excluded, so a hopping
        policy earns strictly less than no-op on average.
        """
        seeds = range(100, 130)
        noop = [_run(lambda t: 0, s, "environment") for s in seeds]
        pingpong = [_run(lambda t: 1 if t % 2 == 0 else 6, s, "environment") for s in seeds]
        self.assertLess(statistics.mean(pingpong), statistics.mean(noop))

    def test_reward_components_logged(self) -> None:
        """B4 instrument: per-component reward breakdown is exposed for analysis."""
        env = LibraryEnv(seed=7, reward_mode="environment")
        env.reset(seed=7)
        _, _, _, _, info = env.step(0)
        comps = info["reward_components"]
        for key in ("privacy", "familiarity", "dwell_bonus", "realized_move_cost", "reward"):
            self.assertIn(key, comps)
        self.assertEqual(info["reward_mode"], "environment")

    def test_environment_mode_excludes_habit_terms(self) -> None:
        """B4: environment reward == total minus familiarity/dwell (no move)."""
        env = LibraryEnv(seed=11, reward_mode="environment")
        env.reset(seed=11)
        _, reward, _, _, info = env.step(0)  # stay: no move cost
        c = info["reward_components"]
        expected = c["total"] - c["familiarity"] - c["dwell_bonus"]
        self.assertAlmostEqual(reward, expected, places=5)

    def test_score_withheld_changes_observation(self) -> None:
        """B5: withholding the true score actually removes it from the obs."""
        env_on = LibraryEnv(seed=5, reward_mode="environment", score_in_obs=True)
        obs_on, _ = env_on.reset(seed=5)
        env_off = LibraryEnv(seed=5, reward_mode="environment", score_in_obs=False)
        obs_off, _ = env_off.reset(seed=5)
        # Same seed => same world; only the score channels should differ.
        self.assertFalse(np.allclose(obs_on, obs_off))

    def test_apply_external_move_semantics(self) -> None:
        """B13: single move path reports outcomes and honours ignore_cooldown."""
        env = LibraryEnv(seed=3, reward_mode="environment")
        env.reset(seed=3)
        sim = env.sim
        focal_id = sim.focal_agent_id
        agent = sim.agents[focal_id]

        # Moving onto your own seat is a no-op with a clear reason.
        same = sim.apply_external_move(focal_id, agent.current_seat_id)
        self.assertFalse(same.moved)
        self.assertEqual(same.reason, "same_seat")

        target = next(s.id for s in env.world.available_seats())
        ok = sim.apply_external_move(focal_id, target)
        self.assertTrue(ok.moved)
        self.assertEqual(agent.cooldown_steps_remaining, env.config.focal_move_cooldown_steps)

        # Cooldown blocks a second move unless explicitly ignored.
        target2 = next(s.id for s in env.world.available_seats())
        blocked = sim.apply_external_move(focal_id, target2)
        self.assertFalse(blocked.moved)
        self.assertEqual(blocked.reason, "cooldown")
        forced = sim.apply_external_move(focal_id, target2, ignore_cooldown=True)
        self.assertTrue(forced.moved)

    def test_config_replace_preserves_new_fields(self) -> None:
        """Lossy config copy fix: env keeps focal_agent_random_spawn."""
        cfg = SimulationConfig(focal_agent_random_spawn=True)
        env = LibraryEnv(seed=1, config=cfg)
        self.assertTrue(env.config.focal_agent_random_spawn)


@unittest.skipUnless(HAS_GYM, "gymnasium not installed")
class TestEvalHarness(unittest.TestCase):
    def test_random_baseline_is_seeded(self) -> None:
        """B7: the random policy is reproducible from the base seed."""
        import evaluate_agent as ev

        def actions(seed: int) -> list[int]:
            rng = np.random.default_rng(seed)
            _, policy = ev.build_policies(None, rng)[0]  # ("Random", fn)
            env = ev.make_env(seed, 12, ["library_v1"])
            env.reset(seed=seed)
            return [policy(env, None) for _ in range(20)]

        self.assertEqual(actions(42), actions(42))
        self.assertNotEqual(actions(42), actions(43))

    def test_oracle_differs_from_greedy(self) -> None:
        """B3: the best-seat reference is a genuinely different policy.

        The old 'perfect-info oracle' was byte-identical to greedy. The
        replacement re-evaluates every seat and relocates, so across seeds it
        produces a different mean return and moves more than zero on average.
        """
        import evaluate_agent as ev
        seeds = range(200, 215)
        greedy_r, oracle_r, oracle_moves = [], [], []
        for s in seeds:
            genv = ev.make_env(s, 24, ["library_v1"])
            greedy_r.append(ev.run_env_episode(genv, lambda e, o: ev.greedy_candidate_action(e), s).total_reward)
            oenv = ev.make_env(s, 24, ["library_v1"])
            res = ev.run_env_episode(oenv, lambda e, o: (ev.oracle_apply(e), 0)[1], s)
            oracle_r.append(res.total_reward)
            oracle_moves.append(res.moves)
        self.assertGreater(statistics.mean(oracle_moves), 0.0)
        self.assertNotAlmostEqual(statistics.mean(greedy_r), statistics.mean(oracle_r), places=2)


if __name__ == "__main__":
    unittest.main()
