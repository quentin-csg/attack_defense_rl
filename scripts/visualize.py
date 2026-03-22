"""Visualize a random episode in the Pygame window.

Usage:
    python scripts/visualize.py              # random actions, seed=42
    python scripts/visualize.py --seed 123   # custom seed
    python scripts/visualize.py --speed 5    # slower (default 10 FPS)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to Python path so `src` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.environment.actions import ActionType, encode_action
from src.environment.cyber_env import CyberEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a random CyberEnv episode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--speed", type=int, default=10, help="Steps per second (default: 10)")
    parser.add_argument(
        "--max-steps", type=int, default=200, help="Max steps per episode (default: 200)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    delay = 1.0 / max(args.speed, 1)

    env = CyberEnv(seed=args.seed, max_steps=args.max_steps, render_mode="human")
    obs, info = env.reset()
    env.render()

    print(f"Episode started  |  seed={args.seed}  |  speed={args.speed} FPS")
    print(f"Close the Pygame window or press Ctrl+C to stop.\n")

    try:
        for step in range(args.max_steps):
            # Pick a random valid action using the action mask
            mask = env.action_masks()
            valid_actions = mask.nonzero()[0]
            action = int(env._np_rng.choice(valid_actions))

            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            action_type, target = action // 50, action % 50
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
        while True:
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
