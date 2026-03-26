"""Visualize a CyberEnv episode in the Pygame window.

Usage:
    python scripts/visualize.py                                       # random actions, fixed network
    python scripts/visualize.py --seed 123                            # custom seed
    python scripts/visualize.py --speed 3                             # slower (default 10 FPS)
    python scripts/visualize.py --model models/red_agent_final.zip    # trained agent
    python scripts/visualize.py --pcg small                           # random PCG network (small)
    python scripts/visualize.py --pcg medium --model models/...zip    # PCG + trained agent

Controls (keyboard):
    SPACE      pause / resume
    + / UP     speed up
    - / DOWN   slow down
    R          reset speed to default
    ESC        quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to Python path so `src` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.environment.cyber_env import CyberEnv

_PCG_SIZES = ("small", "medium", "large")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a CyberEnv episode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--speed", type=int, default=10, help="Steps per second (default: 10)")
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Max steps per episode (auto if --pcg)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a trained model (.zip) — uses random actions if not provided",
    )
    parser.add_argument(
        "--pcg",
        type=str,
        default=None,
        choices=_PCG_SIZES,
        help="Use a randomly generated PCG network: small / medium / large",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_delay = 1.0 / max(args.speed, 1)

    # Resolve PCG factory and max_steps
    network_factory = None
    if args.pcg is not None:
        from src.pcg.generator import NetworkSize, generate_network
        from src.config import PCG_MAX_STEPS_SMALL, PCG_MAX_STEPS_MEDIUM, PCG_MAX_STEPS_LARGE

        size_enum = NetworkSize(args.pcg)
        _pcg_max_steps = {
            NetworkSize.SMALL: PCG_MAX_STEPS_SMALL,
            NetworkSize.MEDIUM: PCG_MAX_STEPS_MEDIUM,
            NetworkSize.LARGE: PCG_MAX_STEPS_LARGE,
        }
        max_steps = args.max_steps or _pcg_max_steps[size_enum]

        def network_factory(episode_seed: int | None):  # noqa: E306
            net, meta = generate_network(size_enum, seed=episode_seed)
            print(f"  [PCG] new {args.pcg} network — {meta.n_nodes} nodes, "
                  f"{meta.n_subnets} subnets, difficulty={meta.difficulty_score:.1f}, "
                  f"max_steps={meta.recommended_max_steps}")
            return net

        print(f"PCG mode: {args.pcg} networks  |  max_steps={max_steps}\n")
    else:
        max_steps = args.max_steps or 200

    # Load trained model if provided
    model = None
    if args.model is not None:
        from src.agents.red_trainer import load_model
        from src.agents.wrappers import make_masked_env, make_pcg_masked_env

        print(f"Loading model: {args.model}")
        if args.pcg is not None:
            tmp_env = make_pcg_masked_env(size=args.pcg, seed=args.seed, max_steps=max_steps)
        else:
            tmp_env = make_masked_env(seed=args.seed, max_steps=max_steps)
        model = load_model(args.model, env=tmp_env)
        tmp_env.close()
        print("Model loaded — running trained agent.\n")
    else:
        print("No model provided — running random valid actions.\n")

    env = CyberEnv(
        seed=args.seed,
        max_steps=max_steps,
        render_mode="human",
        network_factory=network_factory,
    )
    obs, info = env.reset()
    # First render creates the renderer; controls are owned by it.
    env.render()

    pcg_label = f"pcg={args.pcg}" if args.pcg else "fixed-network"
    agent_label = f"model={Path(args.model).name}" if args.model else "random"
    print(f"Episode started  |  seed={args.seed}  |  speed={args.speed} FPS  |  {pcg_label}  |  agent={agent_label}")
    print("Controls: SPACE=pause  +/-=speed  R=reset  ESC=quit\n")

    try:
        actual_steps = 0  # only count real env steps, not pause frames
        rng = np.random.default_rng(args.seed)
        while actual_steps < args.max_steps:
            # The renderer is the sole consumer of the Pygame event queue.
            # env.render() → renderer.update() → renderer._handle_events() processes
            # all QUIT / KEYDOWN (SPACE, +/-, R, ESC) / MOUSEBUTTONDOWN events.
            # We read the resulting state from env.renderer_controls.
            controls = env.renderer_controls

            # Check if user closed the window
            if controls is not None and not env._renderer.is_open:
                break

            # If paused: redraw without stepping, sleep briefly (do NOT increment actual_steps)
            if controls is not None and controls.paused:
                env.render()
                time.sleep(0.05)
                continue

            mask = env.action_masks()

            if model is not None:
                action, _ = model.predict(obs, deterministic=True, action_masks=mask)
                action = int(action)
            else:
                # Pick a random valid action
                valid_actions = mask.nonzero()[0]
                action = int(rng.choice(valid_actions))

            obs, reward, terminated, truncated, info = env.step(action)
            actual_steps += 1
            env.render()

            if terminated:
                if info.get("exfiltrated"):
                    print("\n  >>> EXFILTRATION SUCCESSFUL <<<")
                else:
                    print("\n  >>> DETECTED — Game Over <<<")
                break
            if truncated:
                print("\n  >>> MAX STEPS REACHED <<<")
                break

            # Sleep adjusted by speed multiplier
            speed_mult = controls.speed_multiplier if controls is not None else 1.0
            effective_delay = base_delay / max(speed_mult, 0.01)
            time.sleep(effective_delay)

        print(f"\nEpisode finished  |  total reward: {info['episode_reward']:+.1f}")
        print("Window stays open — close it or press ESC / Ctrl+C to exit.")

        # Keep the window open until the user closes it
        while True:
            if env._renderer is None or not env._renderer.is_open:
                break
            env.render()
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
