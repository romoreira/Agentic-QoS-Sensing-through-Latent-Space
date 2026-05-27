#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from traffic_model import make_traffic_model
from run_trex_dynamic_4slice import import_trex, build_streams

from online_dqn_proxy_train import (
    ACTIONS,
    ACTION_MBR,
    DEFAULT_MBR,
    apply_mbr,
    compute_reward,
    get_total_rx_tx,
    json_safe,
    shared_uplink_proxy,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, choices=["fixed_mbr", "reactive_threshold"])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--sync-port", type=int, default=4501)
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--rx-port", type=int, default=1)
    p.add_argument("--duration", type=float, default=900.0)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--episode-steps", type=int, default=60)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--traffic-model",
        default="sine_gold_safe_overload",
        choices=["stochastic", "sine_abrupt", "sine_stress", "sine_overload", "sine_gold_safe_overload"],
    )
    p.add_argument("--wire-overhead-bytes", type=int, default=20)
    p.add_argument("--uplink-capacity-mbps", type=float, default=200.0)
    p.add_argument("--csv", required=True)
    p.add_argument("--jsonl", required=True)
    p.add_argument("--episodes-csv", required=True)
    p.add_argument("--tb", required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-apply", action="store_true")
    return p.parse_args()


def choose_action(
    policy: str,
    prev_visible: np.ndarray,
    prev_uplink_pressure: float,
    prev_gold_shortage: float,
) -> int:
    if policy == "fixed_mbr":
        return 0

    prev_gold = float(prev_visible[0])

    if prev_gold_shortage > 25.0 or prev_uplink_pressure > 1.35:
        return 3

    if prev_gold_shortage > 0.0 or prev_uplink_pressure > 1.10:
        return 2

    if prev_gold >= 175.0:
        return 4

    if prev_gold < 130.0 and prev_uplink_pressure < 0.95:
        return 0

    if prev_gold < 150.0 and prev_uplink_pressure < 1.00:
        return 7

    return 4


def log_tensorboard(tb_writer: Any, step: int, row: Dict[str, Any], offered: Dict[str, float], pm: Dict[str, float], active_mbr: np.ndarray) -> None:
    if tb_writer is None:
        return

    reward = float(row["reward"])
    tb_writer.add_scalar("reward/step", reward, step)

    tb_writer.add_scalar("gold/shortage_mbps", pm["gold_shortage_mbps"], step)
    tb_writer.add_scalar("gold/offered_mbps", offered.get("gold", 0.0), step)
    tb_writer.add_scalar("gold/rx_proxy_mbps", pm["gold_rx_proxy_mbps"], step)
    tb_writer.add_scalar("gold/available_for_gold_mbps", pm["available_for_gold_mbps"], step)

    tb_writer.add_scalar("uplink/pressure", pm["uplink_pressure"], step)
    tb_writer.add_scalar("uplink/utilization", pm["uplink_utilization"], step)
    tb_writer.add_scalar("uplink/rx_proxy_mbps", pm["uplink_rx_proxy_mbps"], step)

    tb_writer.add_scalar("traffic/offered_gold_mbps", offered.get("gold", 0.0), step)
    tb_writer.add_scalar("traffic/offered_silver_mbps", offered.get("silver", 0.0), step)
    tb_writer.add_scalar("traffic/offered_bronze_mbps", offered.get("bronze", 0.0), step)
    tb_writer.add_scalar("traffic/hidden_burst_mbps", pm["hidden_burst_mbps"], step)

    tb_writer.add_scalar("mbr/gold_kbps", active_mbr[0], step)
    tb_writer.add_scalar("mbr/silver_kbps", active_mbr[1], step)
    tb_writer.add_scalar("mbr/bronze_kbps", active_mbr[2], step)


def main() -> int:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.episodes_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.tb).mkdir(parents=True, exist_ok=True)

    tb_writer = SummaryWriter(args.tb) if SummaryWriter is not None else None

    model = make_traffic_model(args.traffic_model, dt_s=args.dt, seed=args.seed)

    api = import_trex()
    STLClient = api[0]
    client = STLClient(server=args.host, sync_port=args.sync_port)
    client.connect()
    client.acquire(ports=[args.port, args.rx_port], force=args.force)

    active_mbr = DEFAULT_MBR.copy()
    prev_mbr = active_mbr.copy()

    if not args.no_apply:
        rc, out = apply_mbr(active_mbr)
        print(f"Initial MBR apply rc={rc}: {out}")

    prev_visible = np.zeros(3, dtype=np.float32)
    prev_uplink_pressure = 0.0
    prev_gold_shortage = 0.0

    fieldnames = [
        "policy",
        "step", "episode", "episode_step", "t_s", "regime",
        "action_id", "action_name", "epsilon", "reward", "train_loss",
        "offered_gold_mbps", "offered_silver_mbps", "offered_bronze_mbps",
        "hidden_burst_mbps",
        "mbr_active_gold_kbps", "mbr_active_silver_kbps", "mbr_active_bronze_kbps",
        "after_mbr_gold_mbps", "after_mbr_silver_mbps", "after_mbr_bronze_mbps", "after_mbr_hidden_mbps",
        "mbr_drop_gold_mbps", "mbr_drop_silver_mbps", "mbr_drop_bronze_mbps",
        "available_for_gold_mbps", "gold_rx_proxy_mbps", "gold_shortage_mbps",
        "total_after_mbr_mbps", "uplink_rx_proxy_mbps", "uplink_pressure", "uplink_utilization",
        "proxy_loss_gold", "proxy_loss_silver", "proxy_loss_bronze",
        "trex_total_tx_pkts", "trex_total_rx_pkts",
        "pfcp_modify_rc", "pfcp_modify_output",
    ]

    ep_fieldnames = ["policy", "episode", "start_step", "end_step", "reward_sum", "reward_mean", "loss_mean", "gold_shortage_sum"]

    episode_rewards: List[float] = []
    episode_gold_shortage: List[float] = []
    episode_start = 0

    try:
        with open(args.csv, "w", newline="") as f_csv, open(args.jsonl, "w") as f_jsonl, open(args.episodes_csv, "w", newline="") as f_ep:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()

            ep_writer = csv.DictWriter(f_ep, fieldnames=ep_fieldnames, extrasaction="ignore")
            ep_writer.writeheader()

            for sample in model.iter_duration(args.duration):
                step = int(sample.step)
                episode = step // max(args.episode_steps, 1)
                episode_step = step % max(args.episode_steps, 1)

                action = choose_action(args.policy, prev_visible, prev_uplink_pressure, prev_gold_shortage)

                prev_mbr = active_mbr.copy()
                active_mbr = ACTION_MBR[action].copy()

                rc = 0
                out = "unchanged"
                if not args.no_apply:
                    if step == 0 or not np.array_equal(active_mbr, prev_mbr):
                        rc, out = apply_mbr(active_mbr)

                offered = {k: float(v) for k, v in sample.mbps.items()}

                client.stop(ports=[args.port])
                client.reset(ports=[args.port])
                streams, _, _ = build_streams(api, offered, args.wire_overhead_bytes)
                client.add_streams(streams=streams, ports=[args.port])
                client.clear_stats()
                client.start(ports=[args.port], duration=args.dt)
                time.sleep(max(args.dt, 0.05))

                stats = client.get_stats()
                total_tx_pkts, total_rx_pkts = get_total_rx_tx(stats)

                pm = shared_uplink_proxy(offered, active_mbr, args.uplink_capacity_mbps)
                reward = compute_reward(pm, active_mbr, prev_mbr)

                current_visible = np.array(
                    [
                        float(offered.get("gold", 0.0)),
                        float(offered.get("silver", 0.0)),
                        float(offered.get("bronze", 0.0)),
                    ],
                    dtype=np.float32,
                )

                row = {
                    "policy": args.policy,
                    "step": step,
                    "episode": episode,
                    "episode_step": episode_step,
                    "t_s": float(sample.t_s),
                    "regime": sample.regime,
                    "action_id": action,
                    "action_name": ACTIONS[action][0],
                    "epsilon": math.nan,
                    "reward": reward,
                    "train_loss": math.nan,
                    "offered_gold_mbps": offered.get("gold", 0.0),
                    "offered_silver_mbps": offered.get("silver", 0.0),
                    "offered_bronze_mbps": offered.get("bronze", 0.0),
                    "mbr_active_gold_kbps": active_mbr[0],
                    "mbr_active_silver_kbps": active_mbr[1],
                    "mbr_active_bronze_kbps": active_mbr[2],
                    "trex_total_tx_pkts": total_tx_pkts,
                    "trex_total_rx_pkts": total_rx_pkts,
                    "pfcp_modify_rc": rc,
                    "pfcp_modify_output": out,
                    **pm,
                }

                writer.writerow(json_safe(row))
                f_jsonl.write(json.dumps(json_safe(row)) + "\n")
                f_csv.flush()
                f_jsonl.flush()

                log_tensorboard(tb_writer, step, row, offered, pm, active_mbr)

                episode_rewards.append(reward)
                episode_gold_shortage.append(pm["gold_shortage_mbps"])

                if episode_step == args.episode_steps - 1:
                    ep_row = {
                        "policy": args.policy,
                        "episode": episode,
                        "start_step": episode_start,
                        "end_step": step,
                        "reward_sum": float(np.sum(episode_rewards)),
                        "reward_mean": float(np.mean(episode_rewards)),
                        "loss_mean": math.nan,
                        "gold_shortage_sum": float(np.sum(episode_gold_shortage)),
                    }
                    ep_writer.writerow(json_safe(ep_row))
                    f_ep.flush()

                    episode_rewards.clear()
                    episode_gold_shortage.clear()
                    episode_start = step + 1

                print(
                    f"step={step:04d} ep={episode:03d} policy={args.policy} "
                    f"action={action:02d} {ACTIONS[action][0]} "
                    f"mbr=({active_mbr[0]:.0f},{active_mbr[1]:.0f},{active_mbr[2]:.0f}) "
                    f"offered=({offered.get('gold',0.0):.1f},{offered.get('silver',0.0):.1f},{offered.get('bronze',0.0):.1f}) "
                    f"hidden={pm['hidden_burst_mbps']:.1f} "
                    f"gold_shortage={pm['gold_shortage_mbps']:.1f} "
                    f"pressure={pm['uplink_pressure']:.2f} "
                    f"reward={reward:.2f} "
                    f"pfcp_rc={rc}"
                )

                prev_visible = current_visible.copy()
                prev_uplink_pressure = pm["uplink_pressure"]
                prev_gold_shortage = pm["gold_shortage_mbps"]

    finally:
        try:
            client.stop(ports=[args.port])
        except Exception:
            pass
        client.disconnect()
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()

    print(f"Wrote CSV: {args.csv}")
    print(f"Wrote JSONL: {args.jsonl}")
    print(f"Wrote episodes CSV: {args.episodes_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
