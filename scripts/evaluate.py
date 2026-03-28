"""Evaluate a trained Red Team model over N episodes.

Usage:
    python scripts/evaluate.py models/red_agent_final.zip
    python scripts/evaluate.py models/red_agent_final.zip --episodes 200
    python scripts/evaluate.py models/red_agent_final.zip --seed 0 --stochastic
    python scripts/evaluate.py models/pcg_small_final.zip --pcg small
    python scripts/evaluate.py models/vs_blue_final.zip --blue-team
    python scripts/evaluate.py models/pcg_blue.zip --pcg medium --blue-team
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.red_trainer import evaluate, load_model
from src.agents.wrappers import make_masked_env, make_pcg_masked_env
from src.config import DEFAULT_MAX_STEPS, PCG_MAX_STEPS_SMALL, PCG_MAX_STEPS_MEDIUM, PCG_MAX_STEPS_LARGE

_PCG_MAX_STEPS = {"small": PCG_MAX_STEPS_SMALL, "medium": PCG_MAX_STEPS_MEDIUM, "large": PCG_MAX_STEPS_LARGE}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Red Team model.")
    parser.add_argument("model_path", type=str, help="Path to saved model (.zip)")
    parser.add_argument(
        "--episodes", type=int, default=100, help="Number of episodes (default: 100)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        default=False,
        help="Use stochastic policy (default: deterministic)",
    )
    parser.add_argument(
        "--pcg",
        type=str,
        default=None,
        choices=("small", "medium", "large"),
        help="Evaluate on random PCG networks instead of fixed network",
    )
    parser.add_argument(
        "--blue-team",
        action="store_true",
        default=False,
        help="Add scripted Blue Team defender during evaluation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deterministic = not args.stochastic

    blue_team = None
    if args.blue_team:
        from src.agents.blue_scripted import ScriptedBlueTeam
        blue_team = ScriptedBlueTeam(seed=args.seed)

    if args.pcg is not None:
        max_steps = _PCG_MAX_STEPS[args.pcg]
        eval_env = make_pcg_masked_env(size=args.pcg, seed=args.seed, max_steps=max_steps, blue_team=blue_team)
    else:
        eval_env = make_masked_env(seed=args.seed, max_steps=DEFAULT_MAX_STEPS, blue_team=blue_team)

    mode_label = f"pcg={args.pcg}" if args.pcg else "fixed-network"
    blue_label = " + blue-team" if args.blue_team else ""
    print(f"Evaluating: {args.model_path}")
    print(f"Episodes: {args.episodes} | Seed: {args.seed} | Policy: {'deterministic' if deterministic else 'stochastic'}")
    print(f"Mode: {mode_label}{blue_label}\n")

    model = load_model(args.model_path, env=eval_env)
    metrics = evaluate(model, eval_env, n_episodes=args.episodes, deterministic=deterministic)
    eval_env.close()

    print("--- Evaluation Results ---")
    print(f"  Exfiltration rate:      {metrics['exfiltration_rate']:.1%}")
    print(f"  Detection rate:         {metrics['detection_rate']:.1%}")
    print(f"  Mean episode reward:    {metrics['mean_episode_reward']:+.1f}")
    print(f"  Mean episode length:    {metrics['mean_episode_length']:.0f} steps")
    print(f"  Mean nodes compromised: {metrics['mean_nodes_compromised']:.1f}")
    print(f"  Mean max suspicion:     {metrics['mean_max_suspicion']:.0f}%")


if __name__ == "__main__":
    main()
