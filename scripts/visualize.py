"""Visualize a CyberEnv episode in the Pygame window.

Usage:
    python scripts/visualize.py                                       # random actions, fixed network
    python scripts/visualize.py --seed 123                            # custom seed
    python scripts/visualize.py --speed 3                             # slower (default 10 FPS)
    python scripts/visualize.py --model models/red_agent_final.zip    # trained agent
    python scripts/visualize.py --pcg small                           # random PCG network (small)
    python scripts/visualize.py --pcg medium --model models/...zip    # PCG + trained agent
    python scripts/visualize.py --pcg small --blue-team               # with active Blue Team defender

Controls (keyboard):
    SPACE      pause / resume
    + / UP     speed up
    - / DOWN   slow down
    R          restart episode (new network in PCG mode)
    ESC        quit
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to Python path so `src` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.environment.cyber_env import CyberEnv

logger = logging.getLogger(__name__)

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
    parser.add_argument(
        "--blue-team",
        action="store_true",
        default=False,
        help="Activate the scripted Blue Team defender",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_delay = 1.0 / max(args.speed, 1)

    # Instantiate Blue Team if requested
    blue_team = None
    if args.blue_team:
        from src.agents.blue_scripted import ScriptedBlueTeam
        blue_team = ScriptedBlueTeam(seed=args.seed)

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
            logger.info(
                "[PCG] new %s network — %d nodes, %d subnets, difficulty=%.1f, max_steps=%d",
                args.pcg, meta.n_nodes, meta.n_subnets, meta.difficulty_score,
                meta.recommended_max_steps,
            )
            print(
                f"  [PCG] new {args.pcg} network — {meta.n_nodes} nodes, "
                f"{meta.n_subnets} subnets, difficulty={meta.difficulty_score:.1f}, "
                f"max_steps={meta.recommended_max_steps}"
            )
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
            tmp_env = make_pcg_masked_env(
                size=args.pcg, seed=args.seed, max_steps=max_steps, blue_team=blue_team
            )
        else:
            tmp_env = make_masked_env(seed=args.seed, max_steps=max_steps, blue_team=blue_team)
        model = load_model(args.model, env=tmp_env)
        tmp_env.close()
        print("Model loaded — running trained agent.\n")
    else:
        print("No model provided — running random valid actions.\n")

    blue_label = "blue-team=ON" if args.blue_team else "blue-team=OFF"
    pcg_label = f"pcg={args.pcg}" if args.pcg else "fixed-network"
    agent_label = f"model={Path(args.model).name}" if args.model else "random"

    env = CyberEnv(
        seed=args.seed,
        max_steps=max_steps,
        render_mode="human",
        network_factory=network_factory,
        blue_team=blue_team,
    )

    try:
        # Outer loop: each iteration is one full episode
        while True:
            obs, info = env.reset()
            # First render creates the renderer; controls are owned by it.
            env.render()
            # Clear step-replay history for the new episode
            if env._renderer is not None:
                env._renderer.reset_history()

            seed_label = str(args.seed) if args.seed is not None else "random"
            print(
                f"Episode started  |  seed={seed_label}  |  speed={args.speed} FPS  "
                f"|  {pcg_label}  |  {blue_label}  |  agent={agent_label}"
            )
            print("Controls: SPACE=pause  +/-=speed  ←/→=replay  R=restart  ESC=quit\n")

            actual_steps = 0
            rng = np.random.default_rng(args.seed)  # deterministic if --seed is set
            restart = False

            while actual_steps < max_steps:
                controls = env.renderer_controls

                # Check if user closed the window
                if controls is not None and not env._renderer.is_open:
                    return

                # Check if user pressed R to restart episode
                if controls is not None and controls.restart_requested:
                    controls.restart_requested = False
                    restart = True
                    break

                # If paused OR in replay mode: redraw without stepping
                in_replay = env._renderer is not None and env._renderer.in_replay
                if (controls is not None and controls.paused) or in_replay:
                    env.render()
                    time.sleep(0.05)
                    continue

                mask = env.action_masks()

                if model is not None:
                    action, _ = model.predict(obs, deterministic=True, action_masks=mask)
                    action = int(action)
                else:
                    valid_actions = mask.nonzero()[0]
                    action = int(rng.choice(valid_actions))

                obs, reward, terminated, truncated, info = env.step(action)
                actual_steps += 1
                env.render()
                # Record this real step for replay (pause/post-episode renders are excluded)
                if env._renderer is not None:
                    env._renderer.record_current_step()

                if terminated:
                    if info.get("exfiltrated"):
                        msg, color = "EXFILTRATION SUCCESSFUL", (0, 220, 80)
                        print("\n  >>> EXFILTRATION SUCCESSFUL <<<")
                    else:
                        msg, color = "DETECTED — Game Over", (255, 40, 40)
                        print("\n  >>> DETECTED — Game Over <<<")
                    if env._renderer is not None:
                        env._renderer.set_terminal_message(msg, color)
                    break
                if truncated:
                    msg, color = f"TIME OUT — {actual_steps} steps", (220, 180, 0)
                    print("\n  >>> MAX STEPS REACHED <<<")
                    if env._renderer is not None:
                        env._renderer.set_terminal_message(msg, color)
                    break

                speed_mult = controls.speed_multiplier if controls is not None else 1.0
                effective_delay = base_delay / max(speed_mult, 0.01)
                time.sleep(effective_delay)

            if restart:
                print("\n  [RESTART requested — starting new episode]\n")
                continue

            print(f"\nEpisode finished  |  total reward: {info['episode_reward']:+.1f}")
            print("Window stays open — press R to restart, ←/→ to replay steps, or ESC to exit.")

            # Keep the window open until the user closes it or presses R
            while True:
                if env._renderer is None or not env._renderer.is_open:
                    return
                controls = env.renderer_controls
                if controls is not None and controls.restart_requested:
                    controls.restart_requested = False
                    print("\n  [RESTART requested — starting new episode]\n")
                    break
                env.render()
                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
