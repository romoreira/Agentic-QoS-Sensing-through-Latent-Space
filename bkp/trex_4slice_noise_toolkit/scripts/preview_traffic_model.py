#!/usr/bin/env python3
"""Generate the planned four-slice traffic curves without connecting to TRex."""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from traffic_model import FourSliceTrafficModel, TrafficModelConfig, SLICE_NAMES
from tb_utils import make_writer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=180.0)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--csv", default="runs/traffic_plan.csv")
    p.add_argument("--tb", default="runs/tensorboard/traffic_plan")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = TrafficModelConfig(dt_s=args.dt)
    model = FourSliceTrafficModel(cfg, seed=args.seed)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    writer = make_writer(args.tb)

    fieldnames = ["step", "t_s", "regime"] + [f"{s}_mbps" for s in SLICE_NAMES] + ["visible_total_mbps", "total_mbps"]
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for sample in model.iter_duration(args.duration):
            row = sample.as_csv_row()
            w.writerow(row)
            step = int(row["step"])
            writer.add_scalar("traffic/regime", row["regime"], step)
            for s in SLICE_NAMES:
                writer.add_scalar(f"traffic_target/{s}_mbps", row[f"{s}_mbps"], step)
            writer.add_scalar("traffic_target/visible_total_mbps", row["visible_total_mbps"], step)
            writer.add_scalar("traffic_target/total_mbps", row["total_mbps"], step)
    writer.flush()
    writer.close()
    print(f"Wrote CSV: {args.csv}")
    if args.tb:
        print(f"TensorBoard: tensorboard --logdir {args.tb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
