#!/usr/bin/env python3
"""Online DQN dry-run against the real TRex + BESS-UPF telemetry loop.

This script is intentionally safe:
- it generates real TRex traffic,
- it reads real RX PG-ID counters from TRex,
- it reads real UPF Prometheus counters,
- it builds the same 85-dim stacked observation used by the DQN,
- it loads a trained DQN checkpoint and chooses an action,
- it logs the intended MBR decision,
- it DOES NOT call pfcpctl and DOES NOT modify BESS/PFCP sessions.

Use this before enabling any real action applier.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from traffic_model import make_traffic_model, SLICE_NAMES
from run_trex_dynamic_4slice import import_trex, build_streams
from real_telemetry_probe import (
    CONTROLLED,
    fetch_prometheus,
    prom_get,
    safe_delta,
    mbps_from_bytes_delta,
    read_pgid_snapshot,
    pg_tx_pkts,
    pg_rx_pkts,
    pg_tx_bytes,
    pg_rx_bytes,
)
from tb_utils import make_writer

try:
    from upf_rl_agents.dqn import DQNAgent
    from upf_rl_env.action.space import action_to_mbr_kbps, action_description, N_ACTIONS
    from upf_rl_env.config import DQNConfig, EnvConfig, NOMINAL_MBR_KBPS
    from upf_rl_env.features import metrics_to_features
    from upf_rl_env.reward import compute_reward
    from upf_rl_env.sources.base import WindowMetrics
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Could not import upf-rl-env. Run this first:\n"
        "  cd ~/Agentic-QoS-Sensing-through-Latent-Space/upf-rl-env\n"
        "  python3 -m pip install -e .\n"
        f"Original error: {exc}"
    )


VISIBLE = ("gold", "silver", "bronze")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--sync-port", type=int, default=4501)
    p.add_argument("--port", type=int, default=0, help="TRex TX uplink port")
    p.add_argument("--rx-port", type=int, default=1, help="TRex RX/N6 port for PG-ID correlation")
    p.add_argument("--duration", type=float, default=300.0)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--traffic-model", default="sine_abrupt", choices=["stochastic", "sine_abrupt"])
    p.add_argument("--upf-metrics-url", default="http://192.168.90.1:8080/metrics")
    p.add_argument("--n6-capacity-mbps", type=float, default=1000.0)
    p.add_argument("--checkpoint", required=True, help="DQN checkpoint, e.g., ../upf-rl-env/checkpoints/dqn_real_csv/final.pt")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    p.add_argument("--eval-mode", action="store_true", default=True, help="Use greedy action selection. Default true.")
    p.add_argument("--epsilon", type=float, default=None, help="Optional manual epsilon for non-greedy dry run. Default uses eval_mode greedy.")
    p.add_argument("--mbr-gold-kbps", type=float, default=200000.0, help="Actual current Gold MBR, not changed by this script")
    p.add_argument("--mbr-silver-kbps", type=float, default=100000.0, help="Actual current Silver MBR, not changed by this script")
    p.add_argument("--mbr-bronze-kbps", type=float, default=50000.0, help="Actual current Bronze MBR, not changed by this script")
    p.add_argument("--wire-overhead-bytes", type=int, default=20)
    p.add_argument("--rx-packet-bytes", type=float, default=174.0)
    p.add_argument("--csv", default="runs/online_dqn_dryrun/online_dqn_dryrun.csv")
    p.add_argument("--jsonl", default="runs/online_dqn_dryrun/online_dqn_dryrun.jsonl")
    p.add_argument("--tb", default="runs/tensorboard/online_dqn_dryrun")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is False. Use --device cpu or --device auto.")
    return name


def build_metrics(row: Dict[str, Any], actual_mbr_kbps: np.ndarray, n6_capacity_mbps: float) -> WindowMetrics:
    offered = np.array([float(row[f"offered_{s}_mbps"]) for s in VISIBLE], dtype=np.float32)
    rx = np.array([float(row[f"rx_{s}_mbps"]) for s in VISIBLE], dtype=np.float32)
    loss = np.array([float(row[f"loss_{s}"]) for s in VISIBLE], dtype=np.float32)
    qer = np.array([float(row[f"qer_red_{s}"]) for s in VISIBLE], dtype=np.float32)
    n6 = float(row["n6_total_mbps"])
    util = float(row.get("n6_utilization", n6 / max(n6_capacity_mbps, 1e-9)))
    return WindowMetrics(
        offered_mbps=offered,
        rx_mbps=rx,
        loss_rate=np.clip(loss, 0.0, 1.0),
        qer_red=np.clip(qer, 0.0, None),
        n6_tx_mbps=n6,
        n6_utilization=util,
        mbr_kbps=actual_mbr_kbps.astype(np.int64),
        timestamp_s=float(row.get("t_s", row.get("step", 0.0))),
    )


def row_from_window(
    sample: Any,
    target_mbps: Dict[str, float],
    pps_by_name: Dict[str, float],
    packet_len_by_name: Dict[str, float],
    pg_before: Dict[str, Dict[str, float]],
    pg_after: Dict[str, Dict[str, float]],
    prom_before: Dict[Any, float],
    prom_after: Dict[Any, float],
    dt: float,
    n6_capacity_mbps: float,
    actual_mbr_kbps: np.ndarray,
    rx_packet_bytes: float,
) -> Dict[str, Any]:
    upf_core_tx_bytes_before = prom_get(prom_before, "upf_bytes_count", dir="tx", iface="Core")
    upf_core_tx_bytes_after = prom_get(prom_after, "upf_bytes_count", dir="tx", iface="Core")
    upf_access_rx_bytes_before = prom_get(prom_before, "upf_bytes_count", dir="rx", iface="Access")
    upf_access_rx_bytes_after = prom_get(prom_after, "upf_bytes_count", dir="rx", iface="Access")
    upf_access_drop_before = prom_get(prom_before, "upf_dropped_count", dir="rx", iface="Access")
    upf_access_drop_after = prom_get(prom_after, "upf_dropped_count", dir="rx", iface="Access")
    pfcp_sessions = prom_get(prom_after, "pfcp_sessions", node_id="192.168.90.2")

    n6_total_mbps = mbps_from_bytes_delta(safe_delta(upf_core_tx_bytes_after, upf_core_tx_bytes_before), dt)
    upf_access_rx_mbps = mbps_from_bytes_delta(safe_delta(upf_access_rx_bytes_after, upf_access_rx_bytes_before), dt)
    upf_access_drop_delta = safe_delta(upf_access_drop_after, upf_access_drop_before)

    row: Dict[str, Any] = {
        "step": sample.step,
        "t_s": sample.t_s,
        "regime": sample.regime,
        "n6_total_mbps": n6_total_mbps,
        "n6_utilization": n6_total_mbps / max(n6_capacity_mbps, 1e-9),
        "pfcp_sessions": pfcp_sessions,
        "upf_access_rx_mbps": upf_access_rx_mbps,
        "upf_access_drop_delta": upf_access_drop_delta,
    }

    total_tx_pkts = 0.0
    total_rx_pkts = 0.0
    for s in SLICE_NAMES:
        b = pg_before.get(s, {})
        a = pg_after.get(s, {})
        tx_pkts = safe_delta(pg_tx_pkts(a), pg_tx_pkts(b))
        rx_pkts = safe_delta(pg_rx_pkts(a), pg_rx_pkts(b))
        tx_bytes = safe_delta(pg_tx_bytes(a), pg_tx_bytes(b))
        rx_bytes = safe_delta(pg_rx_bytes(a), pg_rx_bytes(b))
        if tx_bytes <= 0.0 and tx_pkts > 0.0:
            tx_bytes = tx_pkts * float(packet_len_by_name.get(s, 210))
        if rx_bytes <= 0.0 and rx_pkts > 0.0:
            rx_bytes = rx_pkts * float(rx_packet_bytes)

        rx_mbps = mbps_from_bytes_delta(rx_bytes, dt)
        loss = max(tx_pkts - rx_pkts, 0.0) / max(tx_pkts, 1.0)
        total_tx_pkts += tx_pkts
        total_rx_pkts += rx_pkts

        if s in VISIBLE:
            row[f"offered_{s}_mbps"] = float(target_mbps[s])
            row[f"rx_{s}_mbps"] = float(rx_mbps)
            row[f"loss_{s}"] = float(loss)
            row[f"qer_red_{s}"] = float(loss)
        else:
            row["noise_offered_mbps"] = float(target_mbps[s])
            row["noise_rx_mbps"] = float(rx_mbps)
            row["noise_loss"] = float(loss)

        row[f"traffic_target_{s}_pps"] = float(pps_by_name.get(s, 0.0))

    row["mbr_gold_kbps"] = float(actual_mbr_kbps[0])
    row["mbr_silver_kbps"] = float(actual_mbr_kbps[1])
    row["mbr_bronze_kbps"] = float(actual_mbr_kbps[2])
    row["trex_total_tx_pkts"] = total_tx_pkts
    row["trex_total_rx_pkts"] = total_rx_pkts
    return row


def log_tb(writer: Any, step: int, row: Dict[str, Any], reward: float, action: int, intended_mbr: np.ndarray, q_values: List[float]) -> None:
    for s in VISIBLE:
        writer.add_scalar(f"live_metrics/offered_{s}_mbps", float(row[f"offered_{s}_mbps"]), step)
        writer.add_scalar(f"live_metrics/rx_{s}_mbps", float(row[f"rx_{s}_mbps"]), step)
        writer.add_scalar(f"live_metrics/loss_{s}", float(row[f"loss_{s}"]), step)
        writer.add_scalar(f"live_metrics/qer_red_{s}", float(row[f"qer_red_{s}"]), step)
        writer.add_scalar(f"actual_mbr/{s}_kbps", float(row[f"mbr_{s}_kbps"]), step)
    writer.add_scalar("live_metrics/n6_total_mbps", float(row["n6_total_mbps"]), step)
    writer.add_scalar("live_metrics/n6_utilization", float(row["n6_utilization"]), step)
    writer.add_scalar("audit/noise_offered_mbps", float(row.get("noise_offered_mbps", 0.0)), step)
    writer.add_scalar("audit/noise_rx_mbps", float(row.get("noise_rx_mbps", 0.0)), step)
    writer.add_scalar("audit/noise_loss", float(row.get("noise_loss", 0.0)), step)
    writer.add_scalar("audit/pfcp_sessions", float(row.get("pfcp_sessions", 0.0)), step)
    writer.add_scalar("audit/trex_total_tx_pkts", float(row.get("trex_total_tx_pkts", 0.0)), step)
    writer.add_scalar("audit/trex_total_rx_pkts", float(row.get("trex_total_rx_pkts", 0.0)), step)

    writer.add_scalar("dryrun/action_id", int(action), step)
    writer.add_scalar("dryrun/reward_diagnostic", float(reward), step)
    writer.add_scalar("dryrun/intended_mbr_gold_kbps", float(intended_mbr[0]), step)
    writer.add_scalar("dryrun/intended_mbr_silver_kbps", float(intended_mbr[1]), step)
    writer.add_scalar("dryrun/intended_mbr_bronze_kbps", float(intended_mbr[2]), step)
    if q_values:
        writer.add_scalar("dryrun/q_max", float(np.max(q_values)), step)
        writer.add_scalar("dryrun/q_mean", float(np.mean(q_values)), step)
        writer.add_scalar("dryrun/q_min", float(np.min(q_values)), step)
        for i, q in enumerate(q_values):
            writer.add_scalar(f"dryrun/q_action_{i:02d}", float(q), step)


def main() -> int:
    args = parse_args()
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.jsonl).parent.mkdir(parents=True, exist_ok=True)
    writer = make_writer(args.tb)

    device = resolve_device(args.device)
    cfg_env = EnvConfig.default()
    cfg_dqn = DQNConfig.default()
    cfg_dqn.device = device

    obs_dim = cfg_env.frame_stack_k * cfg_env.features_per_window
    agent = DQNAgent(obs_dim=obs_dim, n_actions=N_ACTIONS, cfg=cfg_dqn)
    agent.load(args.checkpoint, map_location=device)
    agent.online.eval()
    print(f"Loaded DQN checkpoint: {args.checkpoint}")
    print("DRY RUN MODE: actions are logged only. No pfcpctl call. No BESS/PFCP modification.")

    actual_mbr = np.array([args.mbr_gold_kbps, args.mbr_silver_kbps, args.mbr_bronze_kbps], dtype=np.float32)
    frame_buffer: deque[np.ndarray] = deque(maxlen=cfg_env.frame_stack_k)
    prev_action: int | None = None
    last_action = 0

    model = make_traffic_model(args.traffic_model, dt_s=args.dt, seed=args.seed)
    api = import_trex()
    STLClient = api[0]
    client = STLClient(server=args.host, sync_port=args.sync_port)
    client.connect()
    acquire_ports = [args.port]
    if args.rx_port != args.port:
        acquire_ports.append(args.rx_port)
    client.acquire(ports=acquire_ports, force=args.force)
    if args.rx_port != args.port:
        try:
            client.set_service_mode(ports=[args.rx_port], enabled=True)
            print(f"Enabled service mode on RX port={args.rx_port}")
        except Exception as exc:
            print(f"[WARN] could not enable service mode on rx-port {args.rx_port}: {exc}")
    print(f"Connected to TRex {args.host}:{args.sync_port}, tx-port={args.port}, rx-port={args.rx_port}")
    print(f"Reading UPF metrics from {args.upf_metrics_url}")

    fieldnames = [
        "step", "t_s", "regime",
        "offered_gold_mbps", "offered_silver_mbps", "offered_bronze_mbps",
        "rx_gold_mbps", "rx_silver_mbps", "rx_bronze_mbps",
        "loss_gold", "loss_silver", "loss_bronze",
        "qer_red_gold", "qer_red_silver", "qer_red_bronze",
        "n6_total_mbps", "n6_utilization",
        "mbr_gold_kbps", "mbr_silver_kbps", "mbr_bronze_kbps",
        "noise_offered_mbps", "noise_rx_mbps", "noise_loss",
        "pfcp_sessions", "upf_access_rx_mbps", "upf_access_drop_delta",
        "trex_total_tx_pkts", "trex_total_rx_pkts",
        "dryrun_action_id", "dryrun_reward_diagnostic",
        "dryrun_intended_mbr_gold_kbps", "dryrun_intended_mbr_silver_kbps", "dryrun_intended_mbr_bronze_kbps",
        "dryrun_action_description",
    ]

    try:
        with open(args.csv, "w", newline="") as f_csv, open(args.jsonl, "w") as f_jsonl:
            csv_writer = csv.DictWriter(f_csv, fieldnames=fieldnames, extrasaction='ignore')
            csv_writer.writeheader()
            t0 = time.monotonic()

            for sample in model.iter_duration(args.duration):
                step = sample.step
                target_mbps = sample.mbps
                streams, pps_by_name, packet_len_by_name = build_streams(api, target_mbps, args.wire_overhead_bytes)

                try:
                    prom_before = fetch_prometheus(args.upf_metrics_url)
                except Exception as exc:
                    print(f"[WARN] failed to read Prometheus before step {step}: {exc}")
                    prom_before = {}
                pg_before = read_pgid_snapshot(client)

                client.stop(ports=[args.port])
                client.reset(ports=[args.port])
                client.add_streams(streams=streams, ports=[args.port])
                client.start(ports=[args.port], duration=args.dt)

                elapsed = time.monotonic() - t0
                target_elapsed = (step + 1) * args.dt
                time.sleep(max(0.0, target_elapsed - elapsed) + 0.20)

                pg_after = read_pgid_snapshot(client)
                try:
                    prom_after = fetch_prometheus(args.upf_metrics_url)
                except Exception as exc:
                    print(f"[WARN] failed to read Prometheus after step {step}: {exc}")
                    prom_after = {}

                row = row_from_window(
                    sample=sample,
                    target_mbps=target_mbps,
                    pps_by_name=pps_by_name,
                    packet_len_by_name=packet_len_by_name,
                    pg_before=pg_before,
                    pg_after=pg_after,
                    prom_before=prom_before,
                    prom_after=prom_after,
                    dt=args.dt,
                    n6_capacity_mbps=args.n6_capacity_mbps,
                    actual_mbr_kbps=actual_mbr,
                    rx_packet_bytes=args.rx_packet_bytes,
                )

                metrics = build_metrics(row, actual_mbr, args.n6_capacity_mbps)
                feat = metrics_to_features(metrics, cfg_env)
                if len(frame_buffer) == 0:
                    for _ in range(cfg_env.frame_stack_k):
                        frame_buffer.append(feat.copy())
                else:
                    frame_buffer.append(feat.copy())
                obs = np.concatenate(list(frame_buffer), axis=0).astype(np.float32)

                if args.epsilon is not None:
                    # Explicit epsilon dry run. Usually not needed. Greedy mode is safer to inspect.
                    old_steps = agent._env_steps
                    agent._env_steps = int((1.0 - max(0.0, min(1.0, args.epsilon))) * cfg_dqn.eps_decay_steps)
                    action = agent.act(obs, eval_mode=False)
                    agent._env_steps = old_steps
                else:
                    action = agent.act(obs, eval_mode=True)

                with torch.no_grad():
                    x = torch.from_numpy(obs).unsqueeze(0).to(agent.device)
                    q_values = agent.online(x).squeeze(0).detach().cpu().numpy().astype(float).tolist()

                intended_mbr = action_to_mbr_kbps(int(action), NOMINAL_MBR_KBPS)
                reward, _components = compute_reward(metrics, int(action), prev_action, cfg_env)
                prev_action = int(action)
                last_action = int(action)

                row["dryrun_action_id"] = int(action)
                row["dryrun_reward_diagnostic"] = float(reward)
                row["dryrun_intended_mbr_gold_kbps"] = float(intended_mbr[0])
                row["dryrun_intended_mbr_silver_kbps"] = float(intended_mbr[1])
                row["dryrun_intended_mbr_bronze_kbps"] = float(intended_mbr[2])
                row["dryrun_action_description"] = action_description(int(action))

                for col in fieldnames:
                    row.setdefault(col, "")
                csv_writer.writerow(row)
                f_jsonl.write(json.dumps(row) + "\n")
                f_jsonl.flush()
                f_csv.flush()

                log_tb(writer, step, row, reward, int(action), intended_mbr, q_values)
                writer.flush()

                print(
                    f"step={step:04d} "
                    f"loss=({row['loss_gold']:.3f},{row['loss_silver']:.3f},{row['loss_bronze']:.3f}) "
                    f"n6={row['n6_total_mbps']:.1f} pfcp={row['pfcp_sessions']:.0f} "
                    f"action={action:02d} intended_mbr=({intended_mbr[0]:.0f},{intended_mbr[1]:.0f},{intended_mbr[2]:.0f}) "
                    "DRY_RUN_NO_BESS"
                )
    finally:
        writer.flush()
        writer.close()
        try:
            client.stop(ports=[args.port])
        except Exception:
            pass
        try:
            release_ports = [args.port]
            if args.rx_port != args.port:
                release_ports.append(args.rx_port)
            client.release(ports=release_ports)
        except Exception:
            pass
        client.disconnect()

    print(f"Wrote CSV: {args.csv}")
    print(f"Wrote JSONL: {args.jsonl}")
    if args.tb:
        print(f"TensorBoard: tensorboard --logdir {args.tb}")
    print(f"Last dry-run action: {last_action}. No BESS/PFCP modification was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
