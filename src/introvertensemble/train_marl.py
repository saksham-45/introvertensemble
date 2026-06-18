from __future__ import annotations

import argparse
from pathlib import Path

from ray.rllib.algorithms.ppo import PPOConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LibraryParallelEnv with Ray RLlib PPO.")
    parser.add_argument("--layout", action="append", default=["library_v1"], help="Layout directory name under assets/layouts.")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--initial-learning-agents", type=int, default=20)
    parser.add_argument("--max-agents", type=int, default=200)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--rollout-fragment-length", type=int, default=200)
    parser.add_argument("--train-batch-size", type=int, default=4000)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--curriculum-start", type=int, default=None)
    parser.add_argument("--curriculum-end", type=int, default=None)
    return parser.parse_args()


def make_env(env_config: dict):
    from introvertensemble.marl_env import LibraryParallelEnv

    return LibraryParallelEnv(
        layout_names=env_config.get("layout_names", "library_v1"),
        initial_learning_agent_count=int(env_config.get("initial_learning_agent_count", 20)),
        max_learning_agents=int(env_config.get("max_learning_agents", 200)),
        max_episode_steps=int(env_config.get("max_episode_steps", 1000)),
        seed=int(env_config.get("seed", 42)),
    )


def build_config(args: argparse.Namespace) -> PPOConfig:
    initial_agents = args.curriculum_start if args.curriculum_start is not None else args.initial_learning_agents
    env_config = {
        "layout_names": args.layout,
        "initial_learning_agent_count": initial_agents,
        "max_learning_agents": args.max_agents,
        "max_episode_steps": args.max_episode_steps,
        "seed": 42,
    }
    probe_env = make_env(env_config)
    try:
        active_agent = probe_env.agents[0]
        observation_space = probe_env.observation_space(active_agent)
        action_space = probe_env.action_space(active_agent)
    finally:
        probe_env.close()

    config = (
        PPOConfig()
        .environment(env=make_env, env_config=env_config)
        .framework("torch")
        .training(
            gamma=0.99,
            lr=3e-4,
            lambda_=0.95,
            kl_coeff=0.2,
            clip_param=0.3,
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,
            train_batch_size=args.train_batch_size,
        )
        .rollouts(
            num_rollout_workers=args.num_workers,
            rollout_fragment_length=args.rollout_fragment_length,
        )
        .resources(num_gpus=args.gpus)
    )

    config["multi_agent"] = {
        "policies": {
            "shared_policy": (None, observation_space, action_space, {}),
        },
        "policy_mapping_fn": lambda agent_id, episode=None, worker=None, **kwargs: "shared_policy",
        "count_steps_by_episode": False,
    }
    return config


def main() -> None:
    import ray
    from ray.rllib.algorithms.ppo import PPO

    args = parse_args()
    ray.init(ignore_reinit_error=True)
    config = build_config(args)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None

    if args.resume_checkpoint:
        algo = PPO.from_checkpoint(args.resume_checkpoint)
    else:
        algo = config.build()

    try:
        progress = 0
        while algo.num_timesteps < args.total_timesteps:
            if args.curriculum_start is not None and args.curriculum_end is not None and args.total_timesteps > 0:
                progress = min(1.0, algo.num_timesteps / args.total_timesteps)
                agent_count = int(
                    args.curriculum_start
                    + (args.curriculum_end - args.curriculum_start) * progress
                )
                print(f"curriculum_learning_agents={agent_count}")
            result = algo.train()
            if checkpoint_dir is not None and result.get("episode_reward_mean") is not None:
                checkpoint = algo.save(checkpoint_dir=str(checkpoint_dir))
                print(f"checkpoint={checkpoint}")
            print(
                "timestep={timesteps} episode_reward_mean={reward:.4f} episode_len_mean={length:.2f}".format(
                    timesteps=algo.num_timesteps,
                    reward=float(result.get("episode_reward_mean", 0.0)),
                    length=float(result.get("episode_len_mean", 0.0)),
                )
            )
    finally:
        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
