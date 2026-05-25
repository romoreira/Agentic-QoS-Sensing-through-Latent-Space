#!/usr/bin/env python3
"""Run a dynamic four-slice traffic pattern on one TRex uplink.

This is an audit tool, intentionally independent from the Gym/DQN package.
It loads four GTP-U streams on one physical TRex port. At each window, it
computes target Mbps from the traffic model and converts them to per-stream pps.

Implementation note
-------------------
TRex STL can always run fixed continuous streams. Per-stream rate changes are
less portable across deployments. This tool therefore uses a conservative
windowed reload mode: each window runs continuous streams for --dt seconds,
then reloads the same four streams with the next rates. Use --dt 1.0 or larger
for live BESS auditing. Use scripts/preview_traffic_model.py first when you only
want the mathematical curves.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from traffic_model import (
    FourSliceTrafficModel,
    TrafficModelConfig,
    SLICE_NAMES,
    SLICE_TEIDS,
    SLICE_UE_IPS,
    SLICE_SRC_PORTS,
    SLICE_PG_IDS,
)
from tb_utils import make_writer

GNB_IP = "192.168.70.2"
UPF_N3_IP = "192.168.70.1"
DN_IP = "192.168.80.2"
PAYLOAD_BYTES = 128


def import_trex():
    try:
        from trex_stl_lib.api import (  # type: ignore
            STLClient,
            Ether,
            IP,
            UDP,
            STLPktBuilder,
            STLStream,
            STLTXCont,
            STLFlowLatencyStats,
        )
        from scapy.contrib.gtp import GTP_U_Header  # type: ignore
        return STLClient, Ether, IP, UDP, STLPktBuilder, STLStream, STLTXCont, STLFlowLatencyStats, GTP_U_Header
    except Exception as exc:
        raise SystemExit(
            "Could not import TRex STL libraries. Run on the TRex host and export PYTHONPATH, for example:\n"
            "  export PYTHONPATH=/opt/trex/v3.08/automation/trex_control_plane/interactive:$PYTHONPATH\n"
            f"Original error: {exc}"
        )


def mbps_to_pps(mbps: float, packet_bytes: int, wire_overhead_bytes: int) -> int:
    wire_bytes = packet_bytes + wire_overhead_bytes
    return max(1, int(round((mbps * 1_000_000.0) / (8.0 * wire_bytes))))


def build_packet(api, idx: int):
    _, Ether, IP, UDP, _, _, _, _, GTP_U_Header = api
    return (
        Ether()
        / IP(src=GNB_IP, dst=UPF_N3_IP)
        / UDP(sport=2152, dport=2152, chksum=0)
        / GTP_U_Header(teid=SLICE_TEIDS[idx])
        / IP(src=SLICE_UE_IPS[idx], dst=DN_IP)
        / UDP(sport=SLICE_SRC_PORTS[idx], dport=5555, chksum=0)
        / ("X" * PAYLOAD_BYTES)
    )


def build_streams(api, target_mbps: Dict[str, float], wire_overhead_bytes: int):
    _, _, _, _, STLPktBuilder, STLStream, STLTXCont, STLFlowLatencyStats, _ = api
    streams = []
    packet_len_by_name = {}
    pps_by_name = {}
    for idx, name in enumerate(SLICE_NAMES):
        pkt = build_packet(api, idx)
        packet_bytes = len(pkt)
        pps = mbps_to_pps(target_mbps[name], packet_bytes, wire_overhead_bytes)
        packet_len_by_name[name] = packet_bytes
        pps_by_name[name] = pps
        streams.append(
            STLStream(
                name=f"exp4_dynamic_{name}_teid_{SLICE_TEIDS[idx]}",
                packet=STLPktBuilder(pkt=pkt),
                mode=STLTXCont(pps=pps),
                flow_stats=STLFlowLatencyStats(pg_id=SLICE_PG_IDS[idx]),
            )
        )
    return streams, pps_by_name, packet_len_by_name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--sync-port", type=int, default=4501)
    p.add_argument("--port", type=int, default=0, help="single TRex TX uplink port")
    p.add_argument("--duration", type=float, default=180.0)
    p.add_argument("--dt", type=float, default=1.0, help="rate update window in seconds")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--wire-overhead-bytes", type=int, default=20)
    p.add_argument("--csv", default="runs/trex_dynamic_targets.csv")
    p.add_argument("--jsonl", default="runs/trex_dynamic_targets.jsonl")
    p.add_argument("--tb", default="runs/tensorboard/trex_dynamic")
    p.add_argument("--plan-only", action="store_true", help="do not connect to TRex, only write target curves")
    p.add_argument("--force", action="store_true", help="force acquire port")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = TrafficModelConfig(dt_s=args.dt)
    model = FourSliceTrafficModel(cfg, seed=args.seed)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.jsonl).parent.mkdir(parents=True, exist_ok=True)
    writer = make_writer(args.tb)

    api = None
    client = None
    if not args.plan_only:
        api = import_trex()
        STLClient = api[0]
        client = STLClient(server=args.host, sync_port=args.sync_port)
        client.connect()
        client.acquire(ports=[args.port], force=args.force)
        print(f"Connected to TRex {args.host}:{args.sync_port}, using one uplink port={args.port}")

    fieldnames = ["step", "t_s", "regime"]
    for s in SLICE_NAMES:
        fieldnames += [f"{s}_target_mbps", f"{s}_target_pps", f"{s}_packet_bytes"]
    fieldnames += ["visible_total_mbps", "total_mbps"]

    try:
        with open(args.csv, "w", newline="") as f_csv, open(args.jsonl, "w") as f_jsonl:
            w = csv.DictWriter(f_csv, fieldnames=fieldnames)
            w.writeheader()
            t0 = time.monotonic()
            for sample in model.iter_duration(args.duration):
                target_mbps = sample.mbps
                pps_by_name = {s: 0 for s in SLICE_NAMES}
                packet_len_by_name = {s: 0 for s in SLICE_NAMES}

                if client is not None and api is not None:
                    streams, pps_by_name, packet_len_by_name = build_streams(api, target_mbps, args.wire_overhead_bytes)
                    # Conservative per-window reload. This is for audit, not final closed-loop RL.
                    client.stop(ports=[args.port])
                    client.reset(ports=[args.port])
                    client.add_streams(streams=streams, ports=[args.port])
                    client.start(ports=[args.port], duration=args.dt)
                else:
                    # Estimate pps using the same packet size approximation when not on TRex.
                    packet_len_by_name = {s: 206 for s in SLICE_NAMES}
                    pps_by_name = {
                        s: mbps_to_pps(target_mbps[s], packet_len_by_name[s], args.wire_overhead_bytes)
                        for s in SLICE_NAMES
                    }

                row = {"step": sample.step, "t_s": sample.t_s, "regime": sample.regime}
                for s in SLICE_NAMES:
                    row[f"{s}_target_mbps"] = target_mbps[s]
                    row[f"{s}_target_pps"] = pps_by_name[s]
                    row[f"{s}_packet_bytes"] = packet_len_by_name[s]
                row["visible_total_mbps"] = target_mbps["gold"] + target_mbps["silver"] + target_mbps["bronze"]
                row["total_mbps"] = row["visible_total_mbps"] + target_mbps["noise"]
                w.writerow(row)
                f_jsonl.write(json.dumps(row) + "\n")

                step = sample.step
                writer.add_scalar("traffic/regime", sample.regime, step)
                for s in SLICE_NAMES:
                    writer.add_scalar(f"traffic_target/{s}_mbps", target_mbps[s], step)
                    writer.add_scalar(f"traffic_target/{s}_pps", pps_by_name[s], step)
                writer.add_scalar("traffic_target/visible_total_mbps", row["visible_total_mbps"], step)
                writer.add_scalar("traffic_target/total_mbps", row["total_mbps"], step)

                if args.plan_only:
                    continue
                # Pace the control loop to wall-clock. The start(duration=dt) call already runs traffic,
                # so this sleep just avoids immediate reconfiguration.
                elapsed = time.monotonic() - t0
                target_elapsed = (sample.step + 1) * args.dt
                sleep_s = max(0.0, target_elapsed - elapsed)
                time.sleep(sleep_s)
    finally:
        writer.flush()
        writer.close()
        if client is not None:
            try:
                client.stop(ports=[args.port])
            except Exception:
                pass
            try:
                client.release(ports=[args.port])
            except Exception:
                pass
            try:
                client.disconnect()
            except Exception:
                pass

    print(f"Wrote target CSV: {args.csv}")
    print(f"Wrote target JSONL: {args.jsonl}")
    if args.tb:
        print(f"TensorBoard: tensorboard --logdir {args.tb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
