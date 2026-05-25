"""Sanity check: roll the env out with random actions and print stats.

Use this first after `pip install -e .` to confirm the env works end-to-end.

    python scripts/sanity_check.py
"""
from __future__ import annotations

import argparse
import logging

import gymnasium as gym
import numpy as np

# Importing the package registers the env id
import upf_rl_env  # noqa: F401
from upf_rl_env.action.space import action_description


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-csv", type=str, default="runs/sanity/metrics.csv")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    env = gym.make("upf-rl/SliceQoS-v0")
    # Configure CSV logging on the underlying env
    env.unwrapped._log_csv_path = args.log_csv  # type: ignore[attr-defined]

    obs, info = env.reset(seed=args.seed)
    print(f"obs shape: {obs.shape}, action_space: {env.action_space}")
    print(f"sample obs[:5]: {obs[:5]}")
    print()

    rng = np.random.default_rng(args.seed)
    rewards = []
    losses_g, losses_s, losses_b = [], [], []
    utils = []
    action_counts = np.zeros(int(env.action_space.n), dtype=np.int64)  # type: ignore[arg-type]

    for t in range(args.steps):
        action = int(rng.integers(0, env.action_space.n))  # type: ignore[arg-type]
        action_counts[action] += 1
        obs, r, terminated, truncated, info = env.step(action)
        rewards.append(r)
        m = info["metrics"]
        losses_g.append(m["loss_gold"])
        losses_s.append(m["loss_silver"])
        losses_b.append(m["loss_bronze"])
        utils.append(m["n6_utilization"])
        if args.verbose and t % 50 == 0:
            hs = info.get("hidden_state", {})
            print(
                f"t={t:4d} {action_description(action):>20s} "
                f"r={r:+.3f}  loss=({m['loss_gold']:.3f},{m['loss_silver']:.3f},{m['loss_bronze']:.3f}) "
                f"util={m['n6_utilization']:.2f}  regime={hs.get('regime', '?')}"
            )
        if terminated or truncated:
            obs, info = env.reset()

    print()
    print("=== Sanity stats ===")
    print(f"steps             : {args.steps}")
    print(f"mean reward       : {np.mean(rewards):+.3f}")
    print(f"mean loss gold    : {np.mean(losses_g):.4f}")
    print(f"mean loss silver  : {np.mean(losses_s):.4f}")
    print(f"mean loss bronze  : {np.mean(losses_b):.4f}")
    print(f"mean n6 util      : {np.mean(utils):.3f}")
    print(f"action usage      :")
    for a, c in enumerate(action_counts):
        print(f"  {action_description(a)}: {c}")

    env.close()


if __name__ == "__main__":
    main()
