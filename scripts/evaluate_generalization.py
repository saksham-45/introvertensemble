"""Evaluate a trained generalist agent on held-out TEST layouts vs baselines.

Reports the generalization gap: mean return on the TRAIN pool vs the never-seen
TEST pool (ProcGen protocol). The PPO policy's observations are normalized with
the VecNormalize statistics saved at train time; baselines read env internals and
need no normalization.

    python scripts/evaluate_generalization.py --splits assets/generated/splits.json \
        --model models/ppo_generalist/best_model.zip \
        --vecnormalize models/ppo_generalist/vecnormalize.pkl \
        --episodes 60 --export-json results/generalization.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_agent as ev  # reuse baseline policies + episode runner
from introvertensemble import LibraryEnv
from introvertensemble.simulation import SimulationConfig


def split_dirs(splits_path: Path, split: str) -> list[str]:
    index = json.loads(splits_path.read_text())
    root = splits_path.parent / split
    return [str(root / name) for name in index[split]]


def make_pool_env(layout_dirs: list[str], seed: int, session_steps: int) -> LibraryEnv:
    config = SimulationConfig(
        focal_agent_enabled=True, focal_agent_external_control=True,
        focal_agent_session_steps=session_steps, focal_agent_random_spawn=True,
        events_enabled=True,
    )
    return LibraryEnv(
        layout_names=layout_dirs, config=config, seed=seed,
        reward_mode="environment", score_in_obs=False,
    )


def load_ppo_policy(model_path: Path, vecnormalize_path: Path | None):
    from stable_baselines3 import PPO
    model = PPO.load(str(model_path))
    obs_rms = None
    clip_obs = 10.0
    if vecnormalize_path and vecnormalize_path.exists():
        import pickle
        with open(vecnormalize_path, "rb") as f:
            vec = pickle.load(f)
        obs_rms = vec.obs_rms
        clip_obs = getattr(vec, "clip_obs", 10.0)

    def _normalize(obs: np.ndarray) -> np.ndarray:
        if obs_rms is None:
            return obs
        return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -clip_obs, clip_obs)

    def policy(_env: LibraryEnv, obs: np.ndarray) -> int:
        action, _ = model.predict(_normalize(obs).astype(np.float32), deterministic=True)
        return int(action)

    return policy


def run_split(name: str, layout_dirs: list[str], policies, episodes: int, base_seed: int,
              session_steps: int) -> list[dict]:
    records: list[dict] = []
    for policy_name, policy_func in policies:
        for ep in range(episodes):
            seed = base_seed + ep
            env = make_pool_env(layout_dirs, seed, session_steps)
            res = ev.run_env_episode(env, policy_func, seed)
            records.append({
                "layout": name, "policy": policy_name,
                "reward_mode": "environment", "spawn": "random",
                "episode": ep, "total_reward": res.total_reward,
                "mean_reward": res.mean_reward, "steps": res.steps,
                "moves": res.moves, "final_seat": res.final_seat,
                "final_score": res.final_score,
            })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generalist agent on held-out layouts.")
    parser.add_argument("--splits", type=Path, default=ROOT / "assets" / "generated" / "splits.json")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "ppo_generalist" / "best_model.zip")
    parser.add_argument("--vecnormalize", type=Path, default=ROOT / "models" / "ppo_generalist" / "vecnormalize.pkl")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--session-steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-json", type=Path)
    parser.add_argument("--train-sample", type=int, default=16,
                        help="How many train layouts to sample for the gap measurement.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    trained = None
    model_path = args.model
    if not model_path.exists():
        # Fall back to the final model if the best-by-validation checkpoint is
        # absent (e.g. a run too short to trigger an evaluation).
        fallback = model_path.parent / "ppo_generalist.zip"
        if fallback.exists():
            model_path = fallback
    if model_path.exists():
        trained = load_ppo_policy(model_path, args.vecnormalize)
        print(f"Loaded PPO model: {model_path}")
    else:
        print(f"No model at {args.model}; baselines only.")

    policies = ev.build_policies(trained, rng)

    all_train = split_dirs(args.splits, "train")
    train_sample = list(rng.choice(all_train, size=min(args.train_sample, len(all_train)), replace=False))
    test_dirs = split_dirs(args.splits, "test")

    print(f"Evaluating on {len(train_sample)} train (seen) and {len(test_dirs)} test (unseen) layouts, "
          f"{args.episodes} episodes/policy each...")
    records = []
    records += run_split("train_pool (seen)", train_sample, policies, args.episodes, args.seed, args.session_steps)
    records += run_split("test_pool (unseen)", test_dirs, policies, args.episodes, args.seed + 500, args.session_steps)

    # Console summary + generalization gap for the PPO policy.
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["layout"], r["policy"])].append(r["total_reward"])
    print("\nGeneralization results (total reward, mean ± 95% CI)")
    print("=" * 78)
    for (layout, policy), vals in sorted(grouped.items()):
        arr = np.array(vals)
        ci = 1.96 * arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        print(f"{layout:22} | {policy:34} | {arr.mean():7.2f} ± {ci:5.2f}")

    def mean_for(layout, policy):
        vals = grouped.get((layout, policy))
        return float(np.mean(vals)) if vals else float("nan")

    ppo_name = next((p for (_, p) in grouped if p == "Trained PPO"), None)
    if ppo_name:
        seen = mean_for("train_pool (seen)", "Trained PPO")
        unseen = mean_for("test_pool (unseen)", "Trained PPO")
        print("-" * 78)
        print(f"PPO generalization gap (seen - unseen): {seen - unseen:.2f} "
              f"(seen {seen:.2f}, unseen {unseen:.2f})")

    if args.export_json:
        args.export_json.parent.mkdir(parents=True, exist_ok=True)
        args.export_json.write_text(json.dumps(records, indent=2))
        print(f"\nExported {len(records)} records to {args.export_json}")


if __name__ == "__main__":
    main()
