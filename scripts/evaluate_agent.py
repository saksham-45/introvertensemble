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


def make_env(seed: int, session_steps: int, layout_names: list[str]) -> LibraryEnv:
    config = SimulationConfig(
        focal_agent_enabled=True,
        focal_agent_external_control=True,
        focal_agent_session_steps=session_steps,
        events_enabled=True,
    )
    return LibraryEnv(layout_names=layout_names, config=config, seed=seed)


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
        "--eval-layouts",
        type=str,
        nargs="+",
        default=["library_v1"],
        help="List of layout names to evaluate on (default: library_v1)",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of different seeds to sweep for each layout (default: 1)",
    )
    parser.add_argument(
        "--export-csv",
        type=Path,
        help="Export results to CSV file",
    )
    parser.add_argument(
        "--export-json",
        type=Path,
        help="Export results to JSON file",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT / "models" / "ppo_multi_layout.zip",
        help="Trained PPO model path.",
    )
    args = parser.parse_args()

    # Define policies
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

    # Load trained PPO model if available
    model_candidates = [
        args.model_path,
        args.model_path.parent / "best_model.zip",
    ]
    model_file = next((path for path in model_candidates if path.exists()), None)
    trained_policy = None
    if model_file is not None:
        from stable_baselines3 import PPO
        model = PPO.load(model_file)
        print(f"Loaded trained model: {model_file}")
        print()

        def trained_policy_func(_env: LibraryEnv, obs: np.ndarray) -> int:
            action, _ = model.predict(obs, deterministic=True)
            return int(action)

        trained_policy = trained_policy_func
    else:
        print(f"No trained model at {args.model_path}; skipping PPO evaluation.")
        print("Train first with: ./run.sh train")
        print()

    # Build list of policies to evaluate
    env_policies: list[tuple[str, object]] = [
        ("Random", random_policy),
        ("No-op (stay put)", noop_policy),
        ("Greedy (top-10 candidates)", greedy_policy),
        ("Perfect-info oracle (all empty seats)", perfect_info_policy),
    ]
    if trained_policy is not None:
        env_policies.append(("Trained PPO", trained_policy))

    # We'll collect all episode results here
    all_results: list[dict] = []

    # For each layout, for each seed offset, for each policy, run episodes
    for layout_idx, layout_name in enumerate(args.eval_layouts):
        for seed_offset in range(args.num_seeds):
            # Compute a base seed for this layout and seed offset
            # We use a large multiplier to avoid overlap between layout_idx and seed_offset
            base_seed = args.seed + layout_idx * 10000 + seed_offset * 1000
            # Create an environment that uses only this layout (no randomization)
            env_factory = lambda seed_val: make_env(seed_val, args.session_steps, [layout_name])
            
            for policy_name, policy_func in env_policies:
                # For each episode in this configuration
                for episode in range(args.episodes):
                    # Seed for this episode: we want different episodes to have different seeds
                    episode_seed = base_seed + episode
                    env = env_factory(episode_seed)
                    # Run the episode
                    episode_result = run_env_episode(env, policy_func, episode_seed)
                    # Convert to dict for storage
                    result_dict = {
                        "layout": layout_name,
                        "policy": policy_name,
                        "seed_offset": seed_offset,
                        "episode": episode,
                        "total_reward": episode_result.total_reward,
                        "mean_reward": episode_result.mean_reward,
                        "steps": episode_result.steps,
                        "moves": episode_result.moves,
                        "final_seat": episode_result.final_seat,
                        "final_score": episode_result.final_score,
                    }
                    all_results.append(result_dict)

    # Now we can summarize and print
    # We'll summarize by layout and policy (averaging over seed offsets and episodes)
    print("\nEvaluation Results")
    print("=" * 60)
    
    # Group results by layout and policy
    from collections import defaultdict
    grouped = defaultdict(list)
    for res in all_results:
        key = (res["layout"], res["policy"])
        grouped[key].append(res)
    
    # Compute summary statistics for each group
    summary_rows = []
    for (layout_name, policy_name), results in grouped.items():
        rewards = [r["total_reward"] for r in results]
        means = [r["mean_reward"] for r in results]
        moves = [r["moves"] for r in results]
        finals = [r["final_score"] for r in results]
        
        summary_rows.append({
            "layout": layout_name,
            "policy": policy_name,
            "episodes": len(results),
            "mean_total_reward": np.mean(rewards),
            "std_total_reward": np.std(rewards),
            "mean_step_reward": np.mean(means),
            "std_step_reward": np.std(means),
            "mean_moves": np.mean(moves),
            "std_moves": np.std(moves),
            "mean_final_score": np.mean(finals),
            "std_final_score": np.std(finals),
        })
        
        # Print a line for this group
        print(f"{layout_name:15} | {policy_name:30} | "
              f"Reward: {np.mean(rewards):6.2f} ± {np.std(rewards):5.2f} | "
              f"Steps: {np.mean(means):5.3f} ± {np.std(means):4.3f} | "
              f"Moves: {np.mean(moves):5.2f} ± {np.std(moves):4.2f} | "
              f"Final Score: {np.mean(finals):5.2f} ± {np.std(finals):4.2f}")
    
    print("-" * 60)
    
    # Export if requested
    if args.export_csv:
        import csv
        with open(args.export_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys() if all_results else [])
            writer.writeheader()
            writer.writerows(all_results)
        print(f"Exported detailed results to {args.export_csv}")
    
    if args.export_json:
        import json
        with open(args.export_json, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Exported detailed results to {args.export_json}")


if __name__ == "__main__":
    main()