#!/usr/bin/env python3
"""Sanity check for replaying real telemetry CSV as SliceQoS-v0 observations."""
from __future__ import annotations

import argparse
import numpy as np
import gymnasium as gym

import upf_rl_env  # noqa: F401, registers env
from upf_rl_env.sources.real_csv import RealCsvMetricSource


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--steps", type=int, default=20)
    args = p.parse_args()

    src = RealCsvMetricSource(args.csv, loop=True, use_action_mbr=True)
    env = gym.make("upf-rl/SliceQoS-v0", metric_source=src)
    obs, info = env.reset()

    print(f"rows: {len(src.rows)}")
    print(f"obs shape: {obs.shape}")
    print(f"action_space: {env.action_space}")
    print(f"initial info: {info}")

    assert obs.shape == (85,), f"expected obs shape (85,), got {obs.shape}"
    assert np.isfinite(obs).all(), "initial obs has NaN/Inf"

    for i in range(args.steps):
        action = int(env.action_space.sample())
        obs, reward, terminated, truncated, info = env.step(action)
        metrics = info["metrics"]
        assert np.isfinite(obs).all(), f"obs has NaN/Inf at step {i}"
        for key in ["loss_gold", "loss_silver", "loss_bronze", "n6_utilization"]:
            v = float(metrics[key])
            assert np.isfinite(v), f"{key} is NaN/Inf at step {i}"
        if i < 5:
            print(
                f"step={i:03d} action={action:02d} reward={reward:+.3f} "
                f"offered=({metrics['offered_gold_mbps']:.1f},"
                f"{metrics['offered_silver_mbps']:.1f},"
                f"{metrics['offered_bronze_mbps']:.1f}) "
                f"rx=({metrics['rx_gold_mbps']:.1f},"
                f"{metrics['rx_silver_mbps']:.1f},"
                f"{metrics['rx_bronze_mbps']:.1f}) "
                f"loss=({metrics['loss_gold']:.3f},"
                f"{metrics['loss_silver']:.3f},"
                f"{metrics['loss_bronze']:.3f}) "
                f"n6={metrics['n6_utilization']:.3f}"
            )
        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    print("real_csv sanity check OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
