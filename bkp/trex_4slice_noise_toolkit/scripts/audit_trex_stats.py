#!/usr/bin/env python3
"""Poll TRex stats and write CSV, JSONL, and optional TensorBoard.

This script is intentionally tolerant of TRex dictionary differences. It logs
raw stats as JSONL and extracts the most common port and pg_id counters when
available.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from traffic_model import SLICE_NAMES, SLICE_PG_IDS
from tb_utils import make_writer

PGID_TO_SLICE = {str(pg): name for pg, name in zip(SLICE_PG_IDS, SLICE_NAMES)}
PGID_TO_SLICE_INT = {pg: name for pg, name in zip(SLICE_PG_IDS, SLICE_NAMES)}


def import_trex_client():
    try:
        from trex_stl_lib.api import STLClient  # type: ignore
        return STLClient
    except Exception as exc:
        raise SystemExit(
            "Could not import TRex STL libraries. Run on the TRex host and export PYTHONPATH, for example:\n"
            "  export PYTHONPATH=/opt/trex/v3.08/automation/trex_control_plane/interactive:$PYTHONPATH\n"
            f"Original error: {exc}"
        )


def as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def first_number(d: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
    for k in keys:
        if k in d:
            v = as_float(d[k])
            if v is not None:
                return v
    return None


def extract_port_stats(stats: Dict[str, Any], port: int) -> Dict[str, float]:
    src = stats.get(port) or stats.get(str(port)) or {}
    if not isinstance(src, dict):
        return {}
    out: Dict[str, float] = {}
    mapping = {
        "tx_bps": ("tx_bps", "tx_bps_L2", "tx_bps_l2"),
        "tx_bps_l1": ("tx_bps_L1", "tx_bps_l1"),
        "rx_bps": ("rx_bps", "rx_bps_L2", "rx_bps_l2"),
        "tx_pps": ("tx_pps",),
        "rx_pps": ("rx_pps",),
        "opackets": ("opackets", "tx_pkts", "tx_packets"),
        "ipackets": ("ipackets", "rx_pkts", "rx_packets"),
        "obytes": ("obytes", "tx_bytes"),
        "ibytes": ("ibytes", "rx_bytes"),
        "oerrors": ("oerrors", "tx_errors"),
        "ierrors": ("ierrors", "rx_errors"),
    }
    for out_key, candidates in mapping.items():
        v = first_number(src, candidates)
        if v is not None:
            out[out_key] = v
    return out


def extract_pgid_stats(pgid_stats: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Extract per-slice counters from client.get_pgid_stats(), if available."""
    result: Dict[str, Dict[str, float]] = {name: {} for name in SLICE_NAMES}
    if not isinstance(pgid_stats, dict):
        return result

    # TRex often nests pg_id flow counters under 'flow_stats' and latency under 'latency'.
    for section_name in ("flow_stats", "latency"):
        section = pgid_stats.get(section_name, {})
        if not isinstance(section, dict):
            continue
        for pgid_key, value in section.items():
            slice_name = PGID_TO_SLICE.get(str(pgid_key))
            if slice_name is None and isinstance(pgid_key, int):
                slice_name = PGID_TO_SLICE_INT.get(pgid_key)
            if slice_name is None or not isinstance(value, dict):
                continue
            flat = result[slice_name]
            for k, v in value.items():
                # Some fields are per-port dictionaries. Keep scalar fields and simple sums.
                if isinstance(v, dict):
                    nums = [as_float(x) for x in v.values()]
                    nums = [x for x in nums if x is not None]
                    if nums:
                        flat[f"{section_name}_{k}_sum"] = float(sum(nums))
                else:
                    num = as_float(v)
                    if num is not None:
                        flat[f"{section_name}_{k}"] = num
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--sync-port", type=int, default=4501)
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--duration", type=float, default=180.0)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--csv", default="runs/trex_stats.csv")
    p.add_argument("--jsonl", default="runs/trex_stats_raw.jsonl")
    p.add_argument("--tb", default="runs/tensorboard/trex_stats")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.jsonl).parent.mkdir(parents=True, exist_ok=True)
    writer = make_writer(args.tb)

    STLClient = import_trex_client()
    client = STLClient(server=args.host, sync_port=args.sync_port)
    client.connect()
    print(f"Connected to TRex {args.host}:{args.sync_port}, polling port={args.port}")

    fieldnames = [
        "step", "t_s",
        "port_tx_mbps", "port_tx_mbps_l1", "port_rx_mbps", "port_tx_pps", "port_rx_pps",
        "port_opackets", "port_ipackets", "port_obytes", "port_ibytes", "port_oerrors", "port_ierrors",
    ]
    for s in SLICE_NAMES:
        fieldnames += [
            f"{s}_flow_tx_pkts", f"{s}_flow_rx_pkts", f"{s}_flow_tx_bytes", f"{s}_flow_rx_bytes",
            f"{s}_lat_avg", f"{s}_lat_max", f"{s}_lat_min",
        ]

    try:
        with open(args.csv, "w", newline="") as f_csv, open(args.jsonl, "w") as f_jsonl:
            w = csv.DictWriter(f_csv, fieldnames=fieldnames)
            w.writeheader()
            n = int(args.duration / args.dt)
            t0 = time.monotonic()
            for step in range(n):
                stats = client.get_stats(ports=[args.port])
                try:
                    pgid_stats = client.get_pgid_stats()
                except Exception:
                    pgid_stats = {}

                now = time.monotonic() - t0
                raw = {"step": step, "t_s": now, "stats": stats, "pgid_stats": pgid_stats}
                f_jsonl.write(json.dumps(raw, default=str) + "\n")

                ps = extract_port_stats(stats, args.port)
                pg = extract_pgid_stats(pgid_stats)
                row: Dict[str, Any] = {"step": step, "t_s": now}
                row["port_tx_mbps"] = ps.get("tx_bps", 0.0) / 1e6
                row["port_tx_mbps_l1"] = ps.get("tx_bps_l1", 0.0) / 1e6
                row["port_rx_mbps"] = ps.get("rx_bps", 0.0) / 1e6
                row["port_tx_pps"] = ps.get("tx_pps", 0.0)
                row["port_rx_pps"] = ps.get("rx_pps", 0.0)
                row["port_opackets"] = ps.get("opackets", 0.0)
                row["port_ipackets"] = ps.get("ipackets", 0.0)
                row["port_obytes"] = ps.get("obytes", 0.0)
                row["port_ibytes"] = ps.get("ibytes", 0.0)
                row["port_oerrors"] = ps.get("oerrors", 0.0)
                row["port_ierrors"] = ps.get("ierrors", 0.0)

                for s in SLICE_NAMES:
                    # Multiple TRex versions use different key names. Fill what exists, leave zero otherwise.
                    sd = pg.get(s, {})
                    row[f"{s}_flow_tx_pkts"] = first_number(sd, ("flow_stats_tx_pkts", "flow_stats_tx_pkts_sum", "flow_stats_opackets", "flow_stats_opackets_sum")) or 0.0
                    row[f"{s}_flow_rx_pkts"] = first_number(sd, ("flow_stats_rx_pkts", "flow_stats_rx_pkts_sum", "flow_stats_ipackets", "flow_stats_ipackets_sum")) or 0.0
                    row[f"{s}_flow_tx_bytes"] = first_number(sd, ("flow_stats_tx_bytes", "flow_stats_tx_bytes_sum", "flow_stats_obytes", "flow_stats_obytes_sum")) or 0.0
                    row[f"{s}_flow_rx_bytes"] = first_number(sd, ("flow_stats_rx_bytes", "flow_stats_rx_bytes_sum", "flow_stats_ibytes", "flow_stats_ibytes_sum")) or 0.0
                    row[f"{s}_lat_avg"] = first_number(sd, ("latency_average", "latency_avg", "latency_avg_delay")) or 0.0
                    row[f"{s}_lat_max"] = first_number(sd, ("latency_total_max", "latency_max", "latency_max_delay")) or 0.0
                    row[f"{s}_lat_min"] = first_number(sd, ("latency_total_min", "latency_min", "latency_min_delay")) or 0.0

                w.writerow(row)
                writer.add_scalar("trex/port_tx_mbps", row["port_tx_mbps"], step)
                writer.add_scalar("trex/port_tx_mbps_l1", row["port_tx_mbps_l1"], step)
                writer.add_scalar("trex/port_rx_mbps", row["port_rx_mbps"], step)
                writer.add_scalar("trex/port_tx_pps", row["port_tx_pps"], step)
                writer.add_scalar("trex/port_rx_pps", row["port_rx_pps"], step)
                for s in SLICE_NAMES:
                    for suffix in ("flow_tx_pkts", "flow_rx_pkts", "lat_avg", "lat_max"):
                        writer.add_scalar(f"trex_pgid/{s}_{suffix}", row[f"{s}_{suffix}"], step)

                time.sleep(args.dt)
    finally:
        writer.flush()
        writer.close()
        client.disconnect()

    print(f"Wrote stats CSV: {args.csv}")
    print(f"Wrote raw JSONL: {args.jsonl}")
    if args.tb:
        print(f"TensorBoard: tensorboard --logdir {args.tb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
