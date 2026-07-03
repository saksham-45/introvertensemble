"""Train a generalizing PPO seat-selection agent on procedural layouts.

Implements the recipe in docs/TRAINING_BEST_PRACTICES.md: domain randomization
over a pool of generated TRAIN layouts, VecNormalize observation normalization,
tuned PPO hyperparameters, and best-model selection on a held-out VAL pool
(never the test pool).

    python scripts/generate_layouts.py --out assets/generated
    python scripts/train_generalization.py --splits assets/generated/splits.json \
        --timesteps 500000 --seed 0 --n-envs 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from introvertensemble import LibraryEnv
from introvertensemble.simulation import SimulationConfig


def _split_dirs(splits_path: Path, split: str) -> list[str]:
    index = json.loads(splits_path.read_text())
    root = splits_path.parent / split
    return [str(root / name) for name in index[split]]


def make_env_fn(layout_dirs: list[str], seed: int, session_steps: int, domain_randomize: bool):
    def _init() -> LibraryEnv:
        config = SimulationConfig(
            focal_agent_enabled=True,
            focal_agent_external_control=True,
            focal_agent_session_steps=session_steps,
            focal_agent_random_spawn=True,
            events_enabled=True,
        )
        return LibraryEnv(
            layout_names=layout_dirs,
            config=config,
            seed=seed,
            reward_mode="environment",
            score_in_obs=False,
            domain_randomize=domain_randomize,
        )
    return _init


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a generalizing PPO agent on generated layouts.")
    parser.add_argument("--splits", type=Path, default=ROOT / "assets" / "generated" / "splits.json")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--session-steps", type=int, default=24)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "ppo_generalist")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs" / "ppo_generalist")
    args = parser.parse_args()

    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
    except ImportError as exc:
        raise SystemExit("RL deps missing. Install with: pip install -e '.[rl]'") from exc

    train_dirs = _split_dirs(args.splits, "train")
    val_dirs = _split_dirs(args.splits, "val")
    print(f"Train layouts: {len(train_dirs)} | Val layouts: {len(val_dirs)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    train_fns = [make_env_fn(train_dirs, args.seed + i, args.session_steps, True) for i in range(args.n_envs)]
    train_env = VecNormalize(
        VecMonitor(vec_cls(train_fns)),
        norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99,
    )
    # Validation env: clean distribution (no domain randomization), reward not
    # normalized so EvalCallback reports true returns; obs stats synced from train.
    eval_env = VecNormalize(
        VecMonitor(DummyVecEnv([make_env_fn(val_dirs, args.seed + 9999, args.session_steps, False)])),
        norm_obs=True, norm_reward=False, clip_obs=10.0, training=False,
    )

    # Tuned PPO (docs/TRAINING_BEST_PRACTICES.md): tanh, wide separate value net,
    # lr 3e-4, gae_lambda 0.95, gamma 0.99, clip 0.2, modest entropy.
    policy_kwargs = dict(activation_fn=torch.nn.Tanh, net_arch=dict(pi=[256, 256], vf=[256, 256]))
    model = PPO(
        "MlpPolicy", train_env, verbose=1, seed=args.seed,
        learning_rate=3e-4, n_steps=2048, batch_size=256, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
        max_grad_norm=0.5, policy_kwargs=policy_kwargs, tensorboard_log=str(args.log_dir),
    )

    eval_callback = EvalCallback(
        eval_env, best_model_save_path=str(args.out.parent),
        log_path=str(args.log_dir / "eval"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=20, deterministic=True,
    )

    run_config = {
        "train_layouts": len(train_dirs), "val_layouts": len(val_dirs),
        "timesteps": args.timesteps, "seed": args.seed, "n_envs": args.n_envs,
        "reward_mode": "environment", "score_in_obs": False, "domain_randomize": True,
        "hyperparams": {"lr": 3e-4, "n_steps": 2048, "batch_size": 256, "n_epochs": 10,
                        "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01,
                        "net_arch": "pi=[256,256] vf=[256,256] tanh"},
    }
    (args.out.parent / "run_config.json").write_text(json.dumps(run_config, indent=2))

    print(f"Training PPO for {args.timesteps:,} steps on {args.n_envs} envs...")
    model.learn(total_timesteps=args.timesteps, callback=eval_callback, progress_bar=True)

    model.save(str(args.out))
    train_env.save(str(args.out.parent / "vecnormalize.pkl"))
    print(f"\nSaved final model to {args.out}.zip")
    print(f"Best-by-validation model at {args.out.parent / 'best_model.zip'}")
    print(f"VecNormalize stats at {args.out.parent / 'vecnormalize.pkl'}")


if __name__ == "__main__":
    main()
