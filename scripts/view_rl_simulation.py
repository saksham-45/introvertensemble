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
from introvertensemble.viewer import LibraryViewer


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize the trained PPO focal agent in the library.")
    parser.add_argument("--seed", type=int, default=42, help="Simulation seed.")
    parser.add_argument("--session-steps", type=int, default=24, help="Focal agent session length.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT / "models" / "ppo_library_v1.zip",
        help="Trained PPO model path.",
    )
    args = parser.parse_args()

    model_candidates = [
        args.model_path,
        args.model_path.parent / "best_model.zip",
    ]
    model_file = next((path for path in model_candidates if path.exists()), None)
    if model_file is None:
        raise SystemExit(
            f"No trained model found at {args.model_path}.\n"
            "Train first with: ./run.sh train"
        )

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit("RL dependencies missing. Install with: pip install -e '.[rl]'") from exc

    model = PPO.load(model_file)
    print(f"Loaded RL policy: {model_file}")

    config = SimulationConfig(
        focal_agent_enabled=True,
        focal_agent_external_control=True,
        focal_agent_session_steps=args.session_steps,
        events_enabled=True,
    )
    env = LibraryEnv(config=config, seed=args.seed)
    obs, _info = env.reset(seed=args.seed)

    state = {
        "obs": obs,
        "reward": 0.0,
        "action": 0,
        "action_label": "stay",
        "episode_reward": 0.0,
        "done": False,
    }

    action_labels = {
        0: "stay",
        **{idx: f"nearby #{idx}" for idx in range(1, 6)},
        **{idx: f"global #{idx - 5}" for idx in range(6, 11)},
    }

    def rl_step():
        if state["done"]:
            return None

        action, _ = model.predict(state["obs"], deterministic=True)
        action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)

        state["obs"] = obs
        state["reward"] = reward
        state["action"] = action
        state["action_label"] = action_labels.get(action, str(action))
        state["episode_reward"] += reward
        state["done"] = terminated or truncated

        summary = info.get("step_summary")
        if state["done"]:
            print(
                f"Episode finished: total_reward={state['episode_reward']:.2f} "
                f"final_seat={info.get('focal_seat_id')}"
            )
        return summary

    viewer = LibraryViewer(
        env.world,
        simulation=env.sim,
        step_hook=rl_step,
    )
    viewer.last_summary = rl_step()
    viewer.run()


if __name__ == "__main__":
    main()
