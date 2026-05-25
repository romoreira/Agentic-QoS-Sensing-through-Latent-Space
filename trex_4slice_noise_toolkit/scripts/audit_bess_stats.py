#!/usr/bin/env python3
"""Loose BESS/PFCP stats sampler.

This does not assume a specific BESS-UPF flavor. It repeatedly runs bessctl and
pfcpctl commands, stores raw outputs, and optionally writes a few parsed numeric
counters to TensorBoard when simple byte/packet fields are visible.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from tb_utils import make_writer

NUM_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_\- ]{0,40}?(?:bytes|packets|pkts|drops|dropped|errors))\s*[:=]\s*(?P<val>[0-9]+)", re.IGNORECASE)


def run_cmd(cmd: List[str]) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return p.stdout + p.stderr
    except FileNotFoundError:
        return f"COMMAND_NOT_FOUND: {cmd[0]}\n"
    except Exception as exc:
        return f"COMMAND_FAILED: {' '.join(cmd)}\n{exc}\n"


def parse_counters(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for m in NUM_RE.finditer(text):
        key = re.sub(r"\s+", "_", m.group("key").strip().lower())
        try:
            out[key] = float(m.group("val"))
        except Exception:
            pass
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=180.0)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--bessctl", default=os.environ.get("BESSCTL", "bessctl"))
    p.add_argument("--pfcpctl", default=os.environ.get("PFCPCTL", "pfcpctl"))
    p.add_argument("--n3-port", default=os.environ.get("N3_PORT_NAME", "n3"))
    p.add_argument("--n6-port", default=os.environ.get("N6_PORT_NAME", "n6"))
    p.add_argument("--module", action="append", default=[], help="BESS module to sample, can be repeated")
    p.add_argument("--csv", default="runs/bess_stats.csv")
    p.add_argument("--raw-dir", default="runs/bess_raw")
    p.add_argument("--tb", default="runs/tensorboard/bess_stats")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    writer = make_writer(args.tb)

    commands = [
        ("pipeline", [args.bessctl, "show", "pipeline"]),
        ("port_n3", [args.bessctl, "show", "port", args.n3_port]),
        ("port_n6", [args.bessctl, "show", "port", args.n6_port]),
        ("pfcp_sessions", [args.pfcpctl, "service", "session", "list"]),
    ]
    for mod in args.module:
        commands.append((f"module_{mod}", [args.bessctl, "show", "module", mod]))

    n = int(args.duration / args.dt)
    fieldnames = ["step", "t_s", "source", "counter", "value"]
    with open(args.csv, "w", newline="") as f_csv:
        w = csv.DictWriter(f_csv, fieldnames=fieldnames)
        w.writeheader()
        t0 = time.monotonic()
        for step in range(n):
            step_dir = raw_dir / f"step_{step:06d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            now = time.monotonic() - t0
            for source, cmd in commands:
                text = run_cmd(cmd)
                (step_dir / f"{source}.txt").write_text(text)
                counters = parse_counters(text)
                for k, v in counters.items():
                    w.writerow({"step": step, "t_s": now, "source": source, "counter": k, "value": v})
                    writer.add_scalar(f"bess/{source}/{k}", v, step)
            time.sleep(args.dt)
    writer.flush()
    writer.close()
    print(f"Wrote parsed CSV: {args.csv}")
    print(f"Wrote raw outputs: {args.raw_dir}")
    if args.tb:
        print(f"TensorBoard: tensorboard --logdir {args.tb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
