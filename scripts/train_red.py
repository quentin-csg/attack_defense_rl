"""Train the Red Team agent with MaskablePPO.

Usage:
    python scripts/train_red.py                                        # fixed network (Phase 3/4)
    python scripts/train_red.py --blue-team                            # + Blue Team (Phase 4)
    python scripts/train_red.py --pcg small                            # PCG small networks (Phase 5)
    python scripts/train_red.py --pcg medium --blue-team               # PCG medium + Blue Team
    python scripts/train_red.py --pcg curriculum --blue-team           # full curriculum
    python scripts/train_red.py --timesteps 100000 --run-name my_exp   # custom budget/name
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.red_trainer import evaluate, train, train_curriculum
from src.agents.wrappers import make_masked_env, make_pcg_masked_env
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
    parser.add_argument(
        "--pcg",
        type=str,
        default=None,
        choices=["small", "medium", "large", "curriculum"],
        help=(
            "PCG mode (Phase 5): 'small', 'medium', 'large' train on random networks "
            "of that size. 'curriculum' runs the full small→medium→large curriculum."
        ),
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

    pcg_mode = args.pcg  # None | "small" | "medium" | "large" | "curriculum"

    print(f"Training Red Team agent | run={run_name} | seed={args.seed}")
    if pcg_mode:
        print(f"PCG mode: {pcg_mode.upper()}")
    else:
        print(f"Network: fixed 8-node | timesteps={args.timesteps:,}")
    print(f"Logs -> {log_dir}  |  Models -> {save_dir}")
    print(f"TensorBoard: tensorboard --logdir {log_dir}\n")

    # --- Curriculum mode ---
    if pcg_mode == "curriculum":
        from src.pcg.curriculum import CurriculumManager
        curriculum = CurriculumManager(seed=args.seed)
        total = curriculum.total_timesteps
        print(f"Curriculum total timesteps: {total:,}")
        model = train_curriculum(
            curriculum=curriculum,
            seed=args.seed,
            log_dir=log_dir,
            save_dir=save_dir,
            blue_team=blue_team,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            save_freq=args.save_freq,
        )
        eval_env = make_pcg_masked_env(size="small", seed=args.seed + 9999, blue_team=blue_team)
        final_label = f"{save_dir}/red_agent_curriculum_final.zip"

    # --- Single PCG size mode ---
    elif pcg_mode in ("small", "medium", "large"):
        from src.config import PCG_MAX_STEPS_LARGE, PCG_MAX_STEPS_MEDIUM, PCG_MAX_STEPS_SMALL
        _size_steps = {"small": PCG_MAX_STEPS_SMALL, "medium": PCG_MAX_STEPS_MEDIUM, "large": PCG_MAX_STEPS_LARGE}
        pcg_max_steps = _size_steps[pcg_mode]
        print(f"PCG size={pcg_mode} | max_steps/episode={pcg_max_steps} | timesteps={args.timesteps:,}\n")
        model = train(
            total_timesteps=args.timesteps,
            seed=args.seed,
            log_dir=log_dir,
            save_dir=save_dir,
            max_steps=pcg_max_steps,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
            save_freq=args.save_freq,
            blue_team=blue_team,
            pcg_size=pcg_mode,
        )
        eval_env = make_pcg_masked_env(size=pcg_mode, seed=args.seed + 9999, blue_team=blue_team)
        final_label = f"{save_dir}/red_agent_final.zip"

    # --- Fixed network mode (Phase 3/4) ---
    else:
        print(f"timesteps={args.timesteps:,}\n")
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
        eval_env = make_masked_env(seed=args.seed + 9999, max_steps=DEFAULT_MAX_STEPS, blue_team=blue_team)
        final_label = f"{save_dir}/red_agent_final.zip"

    # --- Post-training evaluation ---
    if model is None:
        print("Training produced no model (interrupted before first update). Skipping evaluation.")
        eval_env.close()
        return

    print("\nPost-training evaluation (20 episodes)...")
    metrics = evaluate(model, eval_env, n_episodes=20, deterministic=True)
    eval_env.close()

    print("\n--- Results ---")
    print(f"  Exfiltration rate:    {metrics['exfiltration_rate']:.1%}")
    print(f"  Detection rate:       {metrics['detection_rate']:.1%}")
    print(f"  Mean reward:          {metrics['mean_episode_reward']:+.1f}")
    print(f"  Mean episode length:  {metrics['mean_episode_length']:.0f} steps")
    print(f"  Mean nodes compromised: {metrics['mean_nodes_compromised']:.1f}")
    print(f"  Mean max suspicion:   {metrics['mean_max_suspicion']:.0f}%")
    print(f"\nFinal model saved to: {final_label}")


if __name__ == "__main__":
    main()
