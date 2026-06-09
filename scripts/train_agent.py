from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from introvertensemble import LibraryEnv
from introvertensemble.simulation import SimulationConfig


def make_env(seed: int, session_steps: int, layout_names: list[str]) -> LibraryEnv:
    config = SimulationConfig(
        focal_agent_enabled=True,
        focal_agent_external_control=True,
        focal_agent_session_steps=session_steps,
        events_enabled=True,
    )
    return LibraryEnv(layout_names=layout_names, config=config, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO agent on multiple layouts.")
    parser.add_argument("--timesteps", type=int, default=80_000, help="Total training timesteps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--session-steps", type=int, default=24, help="Focal agent session length in steps.")
    parser.add_argument(
        "--train-layouts",
        type=str,
        nargs="+",
        default=["library_v1", "library_v2_riverside", "library_v3_courtyard"],
        help="List of layout names to train on (default: library_v1 library_v2_riverside library_v3_courtyard)",
    )
    parser.add_argument(
        "--val-layouts",
        type=str,
        nargs="+",
        default=["library_v1"],
        help="List of layout names to validate on (default: library_v1)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT / "models" / "ppo_multi_layout",
        help="Path prefix for saved model (SB3 appends .zip).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT / "logs" / "ppo_multi_layout",
        help="TensorBoard / SB3 log directory.",
    )
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise SystemExit(
            "RL dependencies missing. Install with:\n"
            "  pip install -e '.[rl]'"
        ) from exc

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    train_env = Monitor(make_env(seed=args.seed, session_steps=args.session_steps, layout_names=args.train_layouts))
    # For evaluation, we'll use the first validation layout (or default to library_v1 if none specified)
    val_layouts = args.val_layouts if args.val_layouts else ["library_v1"]
    eval_env = Monitor(make_env(seed=args.seed + 10_000, session_steps=args.session_steps, layout_names=val_layouts))

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=args.seed,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.02,
        clip_range=0.2,
        tensorboard_log=str(args.log_dir),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=str(args.model_path.parent / "checkpoints"),
        name_prefix="ppo_library_v1",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.model_path.parent),
        log_path=str(args.log_dir / "eval"),
        eval_freq=5_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    print(f"Training PPO on library_v1 for {args.timesteps:,} timesteps...")
    print(f"Session length: {args.session_steps} steps (~{args.session_steps * 15 / 60:.1f} hours)")
    print(f"Model will be saved to: {args.model_path}.zip")
    print(f"TensorBoard logs: {args.log_dir}")
    print()

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    final_path = str(args.model_path)
    model.save(final_path)
    print()
    print(f"Training complete. Model saved to {final_path}.zip")
    print(f"Evaluate with: ./run.sh eval")


if __name__ == "__main__":
    main()
