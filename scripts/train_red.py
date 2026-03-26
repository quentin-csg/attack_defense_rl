"""Train the Red Team agent with MaskablePPO.

Usage:
    python scripts/train_red.py                               # defaults (500k steps)
    python scripts/train_red.py --timesteps 100000            # shorter run
    python scripts/train_red.py --run-name my_exp             # named run
    python scripts/train_red.py --timesteps 1000000 --run-name long_run
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
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
        "--run-name",
        type=str,
        default=None,
        help="Run name — creates logs/{run_name}/ and models/{run_name}/. "
             "Defaults to a timestamp (run_YYYYMMDD_HHMMSS).",
    )
    parser.add_argument(
        "--log-dir", type=str, default=None,
        help="TensorBoard log directory. Overrides --run-name if set.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Model checkpoint directory. Overrides --run-name if set.",
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
    parser.add_argument(
        "--blue-team",
        action="store_true",
        default=False,
        help="Enable scripted Blue Team defender during training (Phase 4).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve run directories — explicit flags override --run-name
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    log_dir = args.log_dir or f"logs/{run_name}"
    save_dir = args.save_dir or f"models/{run_name}"

    blue_team = None
    if args.blue_team:
        from src.agents.blue_scripted import ScriptedBlueTeam
        blue_team = ScriptedBlueTeam(seed=args.seed)
        print("Blue Team: ENABLED (scripted, noisy thresholds, Poisson patrols)")
    else:
        print("Blue Team: DISABLED (Phase 3 mode)")

    print(f"Training Red Team agent | run={run_name} | timesteps={args.timesteps:,} | seed={args.seed}")
    print(f"Logs -> {log_dir}  |  Models -> {save_dir}")
    print(f"TensorBoard: tensorboard --logdir {log_dir}\n")

    model = train(
        total_timesteps=args.timesteps,
        seed=args.seed,
        log_dir=log_dir,
        save_dir=save_dir,
        max_steps=DEFAULT_MAX_STEPS,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        save_freq=args.save_freq,
        blue_team=blue_team,
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
    print(f"\nFinal model saved to: {save_dir}/red_agent_final.zip")


if __name__ == "__main__":
    main()
