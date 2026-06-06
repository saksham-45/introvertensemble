from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from introvertensemble import LibraryEnv, LibrarySimulation, LibraryWorld, load_layout
from introvertensemble.scoring import SeatScorer
from introvertensemble.simulation import SimulationConfig


@dataclass(frozen=True)
class EpisodeResult:
    total_reward: float
    mean_reward: float
    steps: int
    moves: int
    final_seat: str | None
    final_score: float


def make_env(seed: int, session_steps: int) -> LibraryEnv:
    config = SimulationConfig(
        focal_agent_enabled=True,
        focal_agent_external_control=True,
        focal_agent_session_steps=session_steps,
        events_enabled=True,
    )
    return LibraryEnv(config=config, seed=seed)


def focal_step_reward(env: LibraryEnv) -> tuple[float, str | None]:
    focal_id = env.sim.focal_agent_id
    if focal_id is None or focal_id not in env.sim.agents:
        return 0.0, None
    agent = env.sim.agents[focal_id]
    if agent.current_seat_id is None:
        return 0.0, None
    seat_id = agent.current_seat_id
    return env.scorer.score_seat(agent, seat_id).total, seat_id


def try_move_focal_to_seat(env: LibraryEnv, target_seat_id: str) -> bool:
    focal_id = env.sim.focal_agent_id
    if focal_id is None or focal_id not in env.sim.agents:
        return False

    agent = env.sim.agents[focal_id]
    if target_seat_id == agent.current_seat_id:
        return False
    if env.world.occupancy.get(target_seat_id) is not None:
        return False
    if agent.cooldown_steps_remaining > 0:
        return False

    if agent.current_seat_id is not None:
        env.world.release_seat(agent.current_seat_id)

    env.world.occupy_seat(target_seat_id, agent.id)
    agent.record_seat(target_seat_id, env.spec.seats[target_seat_id].zone_id)
    agent.total_moves += 1
    agent.cooldown_steps_remaining = env.config.focal_move_cooldown_steps
    return True


def greedy_candidate_action(env: LibraryEnv) -> int:
    focal_obs = env.obs_builder.build_focal_observation()
    current_score = focal_obs.current_score if focal_obs.current_score is not None else float("-inf")
    best_action = 0
    best_score = current_score

    for idx, candidate in enumerate(focal_obs.nearby_candidates[:5]):
        if candidate.occupied:
            continue
        if candidate.score > best_score:
            best_score = candidate.score
            best_action = idx + 1

    for idx, candidate in enumerate(focal_obs.global_candidates[:5]):
        if candidate.occupied:
            continue
        if candidate.score > best_score:
            best_score = candidate.score
            best_action = idx + 6

    return best_action


def perfect_info_best_seat(env: LibraryEnv) -> str | None:
    focal_id = env.sim.focal_agent_id
    if focal_id is None or focal_id not in env.sim.agents:
        return None

    agent = env.sim.agents[focal_id]
    current_score = float("-inf")
    if agent.current_seat_id is not None:
        current_score = env.scorer.score_seat(agent, agent.current_seat_id).total

    best_seat_id: str | None = None
    best_score = current_score
    for seat in env.world.available_seats():
        score = env.scorer.score_seat(agent, seat.id).total
        if score > best_score:
            best_score = score
            best_seat_id = seat.id

    return best_seat_id


def run_env_episode(env: LibraryEnv, policy, seed: int) -> EpisodeResult:
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    initial_moves = 0
    last_seat: str | None = info.get("focal_seat_id")
    last_score = 0.0

    focal_id = env.sim.focal_agent_id
    tracked_moves = initial_moves
    if focal_id and focal_id in env.sim.agents:
        initial_moves = env.sim.agents[focal_id].total_moves
        tracked_moves = initial_moves
        last_score, last_seat = focal_step_reward(env)

    terminated = truncated = False
    while not (terminated or truncated):
        action = policy(env, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if info.get("focal_seat_id") is not None:
            last_seat = info["focal_seat_id"]
            last_score = reward
        focal_id = env.sim.focal_agent_id
        if focal_id and focal_id in env.sim.agents:
            tracked_moves = env.sim.agents[focal_id].total_moves

    return EpisodeResult(
        total_reward=total_reward,
        mean_reward=total_reward / max(1, steps),
        steps=steps,
        moves=max(0, tracked_moves - initial_moves),
        final_seat=last_seat,
        final_score=last_score,
    )


def run_rule_based_episode(seed: int, session_steps: int) -> EpisodeResult:
    spec = load_layout(ROOT / "assets" / "layouts" / "library_v1")
    world = LibraryWorld(spec)
    config = SimulationConfig(
        focal_agent_enabled=True,
        focal_agent_external_control=False,
        focal_agent_session_steps=session_steps,
        events_enabled=True,
    )
    sim = LibrarySimulation(world, config=config, seed=seed)
    scorer = SeatScorer(world)

    focal_id = sim.focal_agent_id
    initial_moves = 0
    if focal_id and focal_id in sim.agents:
        initial_moves = sim.agents[focal_id].total_moves

    total_reward = 0.0
    steps = 0
    tracked_moves = initial_moves
    last_seat: str | None = None
    last_score = 0.0

    while focal_id is not None and focal_id in sim.agents:
        focal = sim.agents[focal_id]
        tracked_moves = focal.total_moves
        if focal.current_seat_id is not None:
            last_seat = focal.current_seat_id
            last_score = scorer.score_seat(focal, focal.current_seat_id).total
            total_reward += last_score
        sim.step()
        steps += 1

    return EpisodeResult(
        total_reward=total_reward,
        mean_reward=total_reward / max(1, steps),
        steps=steps,
        moves=max(0, tracked_moves - initial_moves),
        final_seat=last_seat,
        final_score=last_score,
    )


def summarize(name: str, results: list[EpisodeResult]) -> dict[str, float]:
    rewards = [item.total_reward for item in results]
    means = [item.mean_reward for item in results]
    moves = [item.moves for item in results]
    finals = [item.final_score for item in results]
    print(f"{name}")
    print(f"  episodes:       {len(results)}")
    print(f"  total reward:   {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"  mean step rew:  {np.mean(means):.3f} +/- {np.std(means):.3f}")
    print(f"  moves/episode:  {np.mean(moves):.2f} +/- {np.std(moves):.2f}")
    print(f"  final score:    {np.mean(finals):.2f} +/- {np.std(finals):.2f}")
    print(f"  sample seat:    {results[0].final_seat}")
    print()
    return {"name": name, "mean_reward": float(np.mean(rewards))}


def print_ranking(rows: list[dict[str, float]]) -> None:
    print("Ranking (higher total reward is better)")
    print("-" * 60)
    for rank, row in enumerate(sorted(rows, key=lambda item: item["mean_reward"], reverse=True), start=1):
        print(f"{rank}. {row['name']}: {row['mean_reward']:.2f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained and baseline seat-selection policies.")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per policy.")
    parser.add_argument("--seed", type=int, default=42, help="Base evaluation seed.")
    parser.add_argument("--session-steps", type=int, default=24, help="Focal agent session length.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT / "models" / "ppo_library_v1.zip",
        help="Trained PPO model path.",
    )
    args = parser.parse_args()

    def noop_policy(_env: LibraryEnv, _obs: np.ndarray) -> int:
        return 0

    def random_policy(_env: LibraryEnv, _obs: np.ndarray) -> int:
        return random.randint(0, 10)

    def greedy_policy(env: LibraryEnv, _obs: np.ndarray) -> int:
        return greedy_candidate_action(env)

    def perfect_info_policy(env: LibraryEnv, _obs: np.ndarray) -> int:
        best_seat_id = perfect_info_best_seat(env)
        if best_seat_id is not None:
            try_move_focal_to_seat(env, best_seat_id)
        return 0

    env_policies: list[tuple[str, object]] = [
        ("Random", random_policy),
        ("No-op (stay put)", noop_policy),
        ("Greedy (top-10 candidates)", greedy_policy),
        ("Perfect-info oracle (all empty seats)", perfect_info_policy),
    ]

    model_candidates = [
        args.model_path,
        args.model_path.parent / "best_model.zip",
    ]
    model_file = next((path for path in model_candidates if path.exists()), None)

    if model_file is not None:
        from stable_baselines3 import PPO

        model = PPO.load(model_file)
        print(f"Loaded trained model: {model_file}")
        print()

        def trained_policy(_env: LibraryEnv, obs: np.ndarray) -> int:
            action, _ = model.predict(obs, deterministic=True)
            return int(action)

        env_policies.append(("Trained PPO", trained_policy))
    else:
        print(f"No trained model at {args.model_path}; skipping PPO evaluation.")
        print("Train first with: ./run.sh train")
        print()

    print("Evaluating seat-selection policies on library_v1")
    print("-" * 60)
    print(f"Episodes per policy: {args.episodes}")
    print(f"Session length: {args.session_steps} steps")
    print()

    rankings: list[dict[str, float]] = []

    for name, policy in env_policies:
        random.seed(args.seed)
        results = [
            run_env_episode(
                make_env(seed=args.seed, session_steps=args.session_steps),
                policy,
                seed=args.seed + episode,
            )
            for episode in range(args.episodes)
        ]
        rankings.append(summarize(name, results))

    rule_based_results = [
        run_rule_based_episode(seed=args.seed + episode, session_steps=args.session_steps)
        for episode in range(args.episodes)
    ]
    rankings.append(summarize("Rule-based focal agent", rule_based_results))

    print_ranking(rankings)


if __name__ == "__main__":
    main()
