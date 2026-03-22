"""Train the Red Team agent with MaskablePPO.

Usage:
    python scripts/train_red.py                            # defaults (500k steps)
    python scripts/train_red.py --timesteps 100000         # shorter run
    python scripts/train_red.py --seed 123 --log-dir logs/run2
    python scripts/train_red.py --timesteps 1000000 --save-dir models/long_run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.red_trainer import evaluate, train
from src.agents.wrappers import make_masked_env
from src.config import (
    DEFAULT_MAX_STEPS,
    RL_EVAL_EPISODES,
    RL_EVAL_FREQ,
    RL_SAVE_FREQ,
    RL_TOTAL_TIMESTEPS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Red Team RL agent with MaskablePPO."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=RL_TOTAL_TIMESTEPS,
        help=f"Total training timesteps (default: {RL_TOTAL_TIMESTEPS})",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--log-dir", type=str, default="logs/", help="TensorBoard log directory (default: logs/)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models/",
        help="Model checkpoint directory (default: models/)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=RL_EVAL_FREQ,
        help=f"Evaluate every N steps (default: {RL_EVAL_FREQ})",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=RL_EVAL_EPISODES,
        help=f"Episodes per evaluation (default: {RL_EVAL_EPISODES})",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=RL_SAVE_FREQ,
        help=f"Checkpoint every N steps (default: {RL_SAVE_FREQ})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Training Red Team agent | timesteps={args.timesteps:,} | seed={args.seed}")
    print(f"Logs -> {args.log_dir}  |  Models -> {args.save_dir}")
    print(f"TensorBoard: tensorboard --logdir {args.log_dir}\n")

    model = train(
        total_timesteps=args.timesteps,
        seed=args.seed,
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        max_steps=DEFAULT_MAX_STEPS,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        save_freq=args.save_freq,
    )

    # Quick post-training evaluation
    print("\nPost-training evaluation (20 episodes)...")
    eval_env = make_masked_env(seed=args.seed + 9999, max_steps=DEFAULT_MAX_STEPS)
    metrics = evaluate(model, eval_env, n_episodes=20, deterministic=True)
    eval_env.close()

    print("\n--- Results ---")
    print(f"  Exfiltration rate:    {metrics['exfiltration_rate']:.1%}")
    print(f"  Detection rate:       {metrics['detection_rate']:.1%}")
    print(f"  Mean reward:          {metrics['mean_episode_reward']:+.1f}")
    print(f"  Mean episode length:  {metrics['mean_episode_length']:.0f} steps")
    print(f"  Mean nodes compromised: {metrics['mean_nodes_compromised']:.1f}")
    print(f"  Mean max suspicion:   {metrics['mean_max_suspicion']:.0f}%")
    print(f"\nFinal model saved to: {args.save_dir}/red_agent_final.zip")


if __name__ == "__main__":
    main()
