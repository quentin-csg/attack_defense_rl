"""Visualize a CyberEnv episode in the Pygame window.

Usage:
    python scripts/visualize.py                          # random actions, seed=42
    python scripts/visualize.py --seed 123               # custom seed
    python scripts/visualize.py --speed 3                # slower (default 10 FPS)
    python scripts/visualize.py --model models/red_agent_final.zip  # trained agent
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to Python path so `src` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import MAX_NODES
from src.environment.actions import ActionType
from src.environment.cyber_env import CyberEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a CyberEnv episode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--speed", type=int, default=10, help="Steps per second (default: 10)")
    parser.add_argument(
        "--max-steps", type=int, default=200, help="Max steps per episode (default: 200)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a trained model (.zip) — uses random actions if not provided",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    delay = 1.0 / max(args.speed, 1)

    # Load trained model if provided
    model = None
    if args.model is not None:
        from src.agents.red_trainer import load_model
        from src.agents.wrappers import make_masked_env

        print(f"Loading model: {args.model}")
        tmp_env = make_masked_env(seed=args.seed, max_steps=args.max_steps)
        model = load_model(args.model, env=tmp_env)
        tmp_env.close()
        print("Model loaded — running trained agent.\n")
    else:
        print("No model provided — running random valid actions.\n")

    env = CyberEnv(seed=args.seed, max_steps=args.max_steps, render_mode="human")
    obs, info = env.reset()
    env.render()

    agent_label = f"model={Path(args.model).name}" if args.model else "random"
    print(f"Episode started  |  seed={args.seed}  |  speed={args.speed} FPS  |  agent={agent_label}")
    print("Close the Pygame window or press Ctrl+C to stop.\n")

    try:
        for step in range(args.max_steps):
            mask = env.action_masks()

            if model is not None:
                action, _ = model.predict(obs, deterministic=True, action_masks=mask)
                action = int(action)
            else:
                # Pick a random valid action
                valid_actions = mask.nonzero()[0]
                action = int(np.random.default_rng(args.seed + step).choice(valid_actions))

            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            action_type, target = action // MAX_NODES, action % MAX_NODES
            action_name = ActionType(action_type).name
            print(
                f"  Step {step + 1:>3d}  |  {action_name:<22s} -> node {target}  |"
                f"  reward={reward:+.1f}  |  suspicion={info['max_suspicion']:.0f}%"
            )

            if terminated:
                if info.get("exfiltrated"):
                    print("\n  >>> EXFILTRATION SUCCESSFUL <<<")
                else:
                    print("\n  >>> DETECTED — Game Over <<<")
                break
            if truncated:
                print("\n  >>> MAX STEPS REACHED <<<")
                break

            time.sleep(delay)

        print(f"\nEpisode finished  |  total reward: {info['episode_reward']:+.1f}")
        print("Window stays open — close it or press Ctrl+C to exit.")

        # Keep the window open until the user closes it
        import pygame

        while True:
            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
