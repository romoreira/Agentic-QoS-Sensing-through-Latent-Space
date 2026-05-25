"""Preview the mock traffic/noise process used to challenge the DQN.

Writes a CSV with visible slice offered load, aggregate N6 utilization, and the
hidden/noise slice ground truth. This is for auditing only, the agent does not
receive the hidden/noise value in its observation.

Example:
    python scripts/preview_noise.py --steps 3000 --out runs/noise_preview.csv
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from upf_rl_env.config import EnvConfig, NOMINAL_MBR_KBPS
from upf_rl_env.sources.mock import MockMetricSource


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="runs/noise_preview.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cfg = EnvConfig.default()
    source = MockMetricSource(cfg)
    source.reset(seed=args.seed)
    mbr = np.array(NOMINAL_MBR_KBPS, dtype=np.int64)

    fields = [
        "step",
        "ts",
        "regime",
        "offered_gold_mbps",
        "offered_silver_mbps",
        "offered_bronze_mbps",
        "noise_offered_mbps",
        "n6_capacity_mbps",
        "n6_tx_mbps",
        "n6_utilization",
        "loss_gold",
        "loss_silver",
        "loss_bronze",
    ]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step in range(1, args.steps + 1):
            m = source.tick(mbr)
            hidden = source.hidden_state
            d = m.as_dict()
            writer.writerow(
                {
                    "step": step,
                    "ts": d["ts"],
                    "regime": hidden["regime"],
                    "offered_gold_mbps": d["offered_gold_mbps"],
                    "offered_silver_mbps": d["offered_silver_mbps"],
                    "offered_bronze_mbps": d["offered_bronze_mbps"],
                    "noise_offered_mbps": hidden["hidden_offered_mbps"],
                    "n6_capacity_mbps": hidden["n6_capacity_mbps"],
                    "n6_tx_mbps": d["n6_tx_mbps"],
                    "n6_utilization": d["n6_utilization"],
                    "loss_gold": d["loss_gold"],
                    "loss_silver": d["loss_silver"],
                    "loss_bronze": d["loss_bronze"],
                }
            )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
