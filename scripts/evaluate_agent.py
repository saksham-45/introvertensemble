from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from introvertensemble import LibraryEnv
from introvertensemble.simulation import SimulationConfig


@dataclass(frozen=True)
class EpisodeResult:
    total_reward: float
    mean_reward: float
    steps: int
    moves: int
    final_seat: str | None
    final_score: float


def make_env(
    seed: int,
    session_steps: int,
    layout_names: list[str],
    *,
    reward_mode: str = "environment",
    score_in_obs: bool = False,
    random_spawn: bool = True,
) -> LibraryEnv:
    """Build an evaluation env.

    Defaults reflect the corrected experimental setup: the environment reward
    (no farmable habit terms, movement charged), no leaked score in the
    observation, and a random spawn seat so policies are measured on their
    ability to *find* good seats rather than inheriting the argmax (B3, B5).
    """
    config = SimulationConfig(
        focal_agent_enabled=True,
        focal_agent_external_control=True,
        focal_agent_session_steps=session_steps,
        events_enabled=True,
        focal_agent_random_spawn=random_spawn,
    )
    return LibraryEnv(
        layout_names=layout_names,
        config=config,
        seed=seed,
        reward_mode=reward_mode,
        score_in_obs=score_in_obs,
    )


def greedy_candidate_action(env: LibraryEnv) -> int:
    """Myopic policy over the discrete candidate actions the env exposes.

    Sees the same information a learned policy does (candidate scores) and picks
    the single best one-step move, subject to the env's action space + cooldown.
    """
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


def _environment_seat_value(env: LibraryEnv, agent, seat_id: str) -> float:
    """Seat value consistent with the environment reward: strip habit terms."""
    breakdown = env.scorer.score_seat(agent, seat_id)
    if env.reward_mode == "legacy":
        return breakdown.total
    return breakdown.total - breakdown.familiarity - breakdown.dwell_bonus


def oracle_apply(env: LibraryEnv) -> None:
    """Strong myopic reference: each step, relocate to the best empty seat.

    Unlike greedy this (a) re-evaluates *every* seat, not just the candidate
    actions, (b) ignores the move cooldown, and (c) weighs the move cost in its
    decision. It is a genuinely different, strong policy — resolving the old
    defect where the "perfect-info oracle" was byte-identical to greedy because
    both inherited the argmax seat at spawn and never moved (B3).

    NB: it is *not* a true upper bound. Choosing the pre-step argmax seat can be
    hurt by post-step crowding changes (myopic thrashing), so it does not
    dominate every policy. A clairvoyant / H-step MPC oracle — feasible because
    utilities are closed-form — is future work (see docs roadmap Phase 5).
    """
    focal_id = env.sim.focal_agent_id
    if focal_id is None or focal_id not in env.sim.agents:
        return
    agent = env.sim.agents[focal_id]
    if agent.current_seat_id is None:
        return

    stay_value = _environment_seat_value(env, agent, agent.current_seat_id)
    best_seat = None
    best_net = stay_value
    for seat in env.world.available_seats():
        value = _environment_seat_value(env, agent, seat.id)
        move_cost = env.move_cost_scale * env.sim.move_path_cost(agent.current_seat_id, seat.id, agent)
        net = value - move_cost
        if net > best_net:
            best_net = net
            best_seat = seat.id

    if best_seat is not None:
        env.sim.apply_external_move(focal_id, best_seat, ignore_cooldown=True)


def run_env_episode(env: LibraryEnv, policy, seed: int) -> EpisodeResult:
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    last_seat: str | None = info.get("focal_seat_id")
    last_score = 0.0

    focal_id = env.sim.focal_agent_id
    initial_moves = 0
    tracked_moves = 0
    if focal_id and focal_id in env.sim.agents:
        initial_moves = env.sim.agents[focal_id].total_moves
        tracked_moves = initial_moves

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


def build_policies(trained_policy, rng: np.random.Generator) -> list[tuple[str, object]]:
    def noop_policy(_env: LibraryEnv, _obs: np.ndarray) -> int:
        return 0

    def random_policy(_env: LibraryEnv, _obs: np.ndarray) -> int:
        # Seeded per run for reproducibility (B7).
        return int(rng.integers(0, 11))

    def greedy_policy(env: LibraryEnv, _obs: np.ndarray) -> int:
        return greedy_candidate_action(env)

    def oracle_policy(env: LibraryEnv, _obs: np.ndarray) -> int:
        oracle_apply(env)
        return 0

    policies: list[tuple[str, object]] = [
        ("Random", random_policy),
        ("No-op (stay put)", noop_policy),
        ("Greedy (candidate actions)", greedy_policy),
        ("Best-seat (myopic, all seats)", oracle_policy),
    ]
    if trained_policy is not None:
        policies.append(("Trained PPO", trained_policy))
    return policies


def summarize_group(results: list[dict]) -> dict:
    rewards = np.array([r["total_reward"] for r in results], dtype=float)
    steps = np.array([r["steps"] for r in results], dtype=float)
    moves = np.array([r["moves"] for r in results], dtype=float)
    finals = np.array([r["final_score"] for r in results], dtype=float)
    n = len(results)
    # 95% CI on the mean via normal approx (rliable's bootstrap CIs are used in
    # the results pipeline; this keeps the CLI dependency-free).
    ci = 1.96 * rewards.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return {
        "episodes": n,
        "mean_total_reward": float(rewards.mean()),
        "std_total_reward": float(rewards.std(ddof=1)) if n > 1 else 0.0,
        "ci95_total_reward": float(ci),
        "mean_steps": float(steps.mean()),
        "mean_moves": float(moves.mean()),
        "std_moves": float(moves.std(ddof=1)) if n > 1 else 0.0,
        "mean_final_score": float(finals.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained and baseline seat-selection policies.")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per policy per (layout, seed).")
    parser.add_argument("--seed", type=int, default=42, help="Base evaluation seed.")
    parser.add_argument("--session-steps", type=int, default=24, help="Focal agent session length.")
    parser.add_argument("--eval-layouts", type=str, nargs="+", default=["library_v1"])
    parser.add_argument("--num-seeds", type=int, default=1, help="Seed offsets swept per layout.")
    parser.add_argument("--reward-mode", choices=list(LibraryEnv.REWARD_MODES), default="environment")
    parser.add_argument("--score-in-obs", dest="score_in_obs", action="store_true", default=False,
                        help="Leak the true seat score into observations (ablation; default off).")
    parser.add_argument("--spawn", choices=["random", "scored"], default="random",
                        help="Focal spawn seat: random (discriminative) or scored argmax.")
    parser.add_argument("--export-csv", type=Path)
    parser.add_argument("--export-json", type=Path)
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "ppo_multi_layout.zip")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    trained_policy = None
    model_candidates = [args.model_path, args.model_path.parent / "best_model.zip"]
    model_file = next((p for p in model_candidates if p.exists()), None)
    if model_file is not None:
        from stable_baselines3 import PPO
        model = PPO.load(model_file)
        print(f"Loaded trained model: {model_file}\n")

        def trained_policy_func(_env: LibraryEnv, obs: np.ndarray) -> int:
            action, _ = model.predict(obs, deterministic=True)
            return int(action)

        trained_policy = trained_policy_func
    else:
        print(f"No trained model at {args.model_path}; skipping PPO. Train with: ./run.sh train\n")

    policies = build_policies(trained_policy, rng)

    all_results: list[dict] = []
    for layout_idx, layout_name in enumerate(args.eval_layouts):
        for seed_offset in range(args.num_seeds):
            base_seed = args.seed + layout_idx * 10_000 + seed_offset * 1_000
            for policy_name, policy_func in policies:
                for episode in range(args.episodes):
                    episode_seed = base_seed + episode
                    env = make_env(
                        episode_seed, args.session_steps, [layout_name],
                        reward_mode=args.reward_mode,
                        score_in_obs=args.score_in_obs,
                        random_spawn=(args.spawn == "random"),
                    )
                    result = run_env_episode(env, policy_func, episode_seed)
                    all_results.append({
                        "layout": layout_name,
                        "policy": policy_name,
                        "reward_mode": args.reward_mode,
                        "spawn": args.spawn,
                        "seed_offset": seed_offset,
                        "episode": episode,
                        "total_reward": result.total_reward,
                        "mean_reward": result.mean_reward,
                        "steps": result.steps,
                        "moves": result.moves,
                        "final_seat": result.final_seat,
                        "final_score": result.final_score,
                    })

    print("Evaluation Results  "
          f"(reward_mode={args.reward_mode}, spawn={args.spawn}, "
          f"score_in_obs={args.score_in_obs})")
    print("=" * 96)
    from collections import defaultdict
    grouped = defaultdict(list)
    for res in all_results:
        grouped[(res["layout"], res["policy"])].append(res)

    for (layout_name, policy_name), results in grouped.items():
        s = summarize_group(results)
        # Column labels are now truthful: Reward (total), StepRew (mean per-step
        # reward), Steps (episode length), Moves (relocations) (B2).
        print(f"{layout_name:15} | {policy_name:34} | "
              f"Reward: {s['mean_total_reward']:7.2f} ± {s['ci95_total_reward']:5.2f} | "
              f"StepRew: {s['mean_final_score']:5.2f} | "
              f"Steps: {s['mean_steps']:4.1f} | "
              f"Moves: {s['mean_moves']:4.2f}")
    print("-" * 96)

    if args.export_csv:
        import csv
        with open(args.export_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys() if all_results else [])
            writer.writeheader()
            writer.writerows(all_results)
        print(f"Exported detailed results to {args.export_csv}")

    if args.export_json:
        import json
        with open(args.export_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Exported detailed results to {args.export_json}")


if __name__ == "__main__":
    main()
