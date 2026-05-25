#!/usr/bin/env python3
"""Run one synchronized real-telemetry probe for the four-slice TRex/BESS testbed.

This is intentionally separate from the Gym/DQN package.
It generates four GTP-U streams on one TRex uplink and, for each window, logs
exactly the 17 features expected by the future DQN plus extra audit columns.

DQN observation columns, one row per window:
  offered_gold_mbps, offered_silver_mbps, offered_bronze_mbps,
  rx_gold_mbps, rx_silver_mbps, rx_bronze_mbps,
  loss_gold, loss_silver, loss_bronze,
  qer_red_gold, qer_red_silver, qer_red_bronze,
  n6_total_mbps, n6_utilization,
  mbr_gold_kbps, mbr_silver_kbps, mbr_bronze_kbps

Notes:
- offered_* comes from the traffic target actually applied to TRex in that window.
- rx_* and loss_* come from TRex PG-ID counters when available.
- qer_red_* is a proxy equal to per-PG-ID loss, because BESS exposes appQER RED
  as a gate aggregate, not per TEID/QER in Prometheus.
- n6_total_mbps comes from UPF Prometheus upf_bytes_count{dir="tx",iface="Core"}.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from traffic_model import (
    FourSliceTrafficModel,
    TrafficModelConfig,
    make_traffic_model,
    SLICE_NAMES,
    SLICE_PG_IDS,
)
from run_trex_dynamic_4slice import import_trex, build_streams
from tb_utils import make_writer

CONTROLLED = ("gold", "silver", "bronze")
AUDIT_ONLY = ("noise",)
PGID_TO_SLICE = {str(pg): name for pg, name in zip(SLICE_PG_IDS, SLICE_NAMES)}
PGID_TO_SLICE_INT = {pg: name for pg, name in zip(SLICE_PG_IDS, SLICE_NAMES)}

PROM_RE = re.compile(r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)\{(?P<labels>[^}]*)\}\s+(?P<value>[-+0-9.eE]+)')
LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"')


def as_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def first_number(d: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
    for k in keys:
        if k in d:
            val = as_float(d[k])
            if val is not None:
                return val
    return None


def flatten_numbers(prefix: str, obj: Any, out: Dict[str, float]) -> None:
    """Flatten scalar numbers from nested TRex dictionaries."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten_numbers(f"{prefix}_{k}" if prefix else str(k), v, out)
    elif isinstance(obj, (int, float)):
        out[prefix] = float(obj)


def extract_pgid_counters(pgid_stats: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Extract per-PG counters from several TRex STL dictionary variants."""
    result: Dict[str, Dict[str, float]] = {name: {} for name in SLICE_NAMES}
    if not isinstance(pgid_stats, dict):
        return result

    # Most useful sections are flow_stats and latency. Different TRex versions
    # place TX/RX counters either directly in a PG-ID object or inside nested
    # dictionaries. We flatten everything and try multiple key names later.
    for section_name in ("flow_stats", "latency"):
        section = pgid_stats.get(section_name, {})
        if not isinstance(section, dict):
            continue
        for pgid_key, value in section.items():
            slice_name = PGID_TO_SLICE.get(str(pgid_key))
            if slice_name is None and isinstance(pgid_key, int):
                slice_name = PGID_TO_SLICE_INT.get(pgid_key)
            if slice_name is None:
                continue
            flat: Dict[str, float] = {}
            flatten_numbers(section_name, value, flat)
            result[slice_name].update(flat)

    return result


def get_counter(d: Dict[str, float], candidates: Tuple[str, ...]) -> float:
    # Exact first. Candidates must prefer *_total before per-port keys.
    for k in candidates:
        if k in d:
            return float(d[k])

    lowered = {k.lower(): k for k in d}

    # Prefer aggregate keys produced by TRex, e.g., flow_stats_rx_pkts_total.
    for cand in candidates:
        c = cand.lower()
        for suffix in ("_total", "_1", "_0"):
            key = c + suffix
            if key in lowered:
                return float(d[lowered[key]])

    # Conservative fallback. Avoid returning *_0 before *_total when both exist.
    for cand in candidates:
        c = cand.lower()
        matches = [orig for lk, orig in lowered.items() if lk.endswith(c) or c in lk]
        if matches:
            matches.sort(key=lambda x: (0 if x.lower().endswith("_total") else 1 if x.lower().endswith("_1") else 2))
            return float(d[matches[0]])

    return 0.0


def read_pgid_snapshot(client: Any) -> Dict[str, Dict[str, float]]:
    try:
        raw = client.get_pgid_stats()
    except Exception:
        raw = {}
    return extract_pgid_counters(raw)


def pg_tx_pkts(sd: Dict[str, float]) -> float:
    return get_counter(sd, (
        "flow_stats_tx_pkts_total", "tx_pkts_total",
        "flow_stats_tx_pkts", "flow_stats_opackets", "latency_tx_pkts", "latency_tx",
        "tx_pkts", "opackets",
    ))


def pg_rx_pkts(sd: Dict[str, float]) -> float:
    return get_counter(sd, (
        "flow_stats_rx_pkts_total", "rx_pkts_total",
        "flow_stats_rx_pkts_1", "rx_pkts_1",
        "flow_stats_rx_pkts", "flow_stats_ipackets", "latency_rx_pkts", "latency_rx",
        "rx_pkts", "ipackets",
    ))


def pg_tx_bytes(sd: Dict[str, float]) -> float:
    return get_counter(sd, (
        "flow_stats_tx_bytes_total", "tx_bytes_total",
        "flow_stats_tx_bytes", "flow_stats_obytes", "latency_tx_bytes", "tx_bytes", "obytes",
    ))


def pg_rx_bytes(sd: Dict[str, float]) -> float:
    return get_counter(sd, (
        "flow_stats_rx_bytes_total", "rx_bytes_total",
        "flow_stats_rx_bytes_1", "rx_bytes_1",
        "flow_stats_rx_bytes", "flow_stats_ibytes", "latency_rx_bytes", "rx_bytes", "ibytes",
    ))


def parse_prometheus(text: str) -> Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float]:
    out: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float] = {}
    for line in text.splitlines():
        m = PROM_RE.match(line.strip())
        if not m:
            continue
        labels = tuple(sorted(LABEL_RE.findall(m.group("labels"))))
        out[(m.group("name"), labels)] = float(m.group("value"))
    return out


def labels_match(labels: Tuple[Tuple[str, str], ...], required: Dict[str, str]) -> bool:
    d = dict(labels)
    return all(d.get(k) == v for k, v in required.items())


def prom_get(metrics: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float], name: str, **labels: str) -> float:
    for (metric_name, metric_labels), value in metrics.items():
        if metric_name == name and labels_match(metric_labels, labels):
            return float(value)
    return 0.0


def fetch_prometheus(url: str, timeout: float = 3.0) -> Dict[Tuple[str, Tuple[Tuple[str, str], ...]], float]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    return parse_prometheus(text)


def safe_delta(after: float, before: float) -> float:
    # Counter reset guard.
    if after < before:
        return max(after, 0.0)
    return after - before


def mbps_from_bytes_delta(delta_bytes: float, dt_s: float) -> float:
    return 8.0 * max(delta_bytes, 0.0) / max(dt_s, 1e-9) / 1_000_000.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--sync-port", type=int, default=4501)
    p.add_argument("--port", type=int, default=0, help="TRex TX uplink port")
    p.add_argument("--rx-port", type=int, default=1, help="TRex RX/N6 port used for latency PG-ID correlation")
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--traffic-model", default="stochastic", choices=["stochastic", "sine_abrupt"], help="traffic generator: stochastic AR/burst regime or clean sinusoids + abrupt noise")
    p.add_argument("--upf-metrics-url", default="http://192.168.90.1:8080/metrics")
    p.add_argument("--n6-capacity-mbps", type=float, default=1000.0)
    p.add_argument("--mbr-gold-kbps", type=float, default=200000.0)
    p.add_argument("--mbr-silver-kbps", type=float, default=100000.0)
    p.add_argument("--mbr-bronze-kbps", type=float, default=50000.0)
    p.add_argument("--wire-overhead-bytes", type=int, default=20)
    p.add_argument("--rx-packet-bytes", type=float, default=174.0, help="fallback RX bytes per decapped packet if PGID bytes are unavailable")
    p.add_argument("--csv", default="runs/real_telemetry_probe.csv")
    p.add_argument("--jsonl", default="runs/real_telemetry_probe.jsonl")
    p.add_argument("--tb", default="runs/tensorboard/real_telemetry_probe")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.jsonl).parent.mkdir(parents=True, exist_ok=True)
    writer = make_writer(args.tb)

    model = make_traffic_model(args.traffic_model, dt_s=args.dt, seed=args.seed)

    api = import_trex()
    STLClient = api[0]
    client = STLClient(server=args.host, sync_port=args.sync_port)
    client.connect()
    acquire_ports = [args.port]
    if args.rx_port != args.port:
        acquire_ports.append(args.rx_port)
    client.acquire(ports=acquire_ports, force=args.force)
    # TRex latency PG-ID RX correlation for GTP-U decap requires the RX port in service mode,
    # the same as the manual console sequence: service --port 1.
    # Without this, the UPF can forward traffic and N6 Prometheus can increase,
    # while get_pgid_stats() still reports rx_pkts=0.
    if args.rx_port != args.port:
        try:
            client.set_service_mode(ports=[args.rx_port], enabled=True)
            print(f"Enabled service mode on RX port={args.rx_port}")
        except Exception as exc:
            print(f"[WARN] could not enable service mode on rx-port {args.rx_port}: {exc}")
    print(f"Connected to TRex {args.host}:{args.sync_port}, tx-port={args.port}, rx-port={args.rx_port}")
    print(f"Reading UPF metrics from {args.upf_metrics_url}")

    mbr = {
        "gold": args.mbr_gold_kbps,
        "silver": args.mbr_silver_kbps,
        "bronze": args.mbr_bronze_kbps,
    }

    feature_cols = [
        "offered_gold_mbps", "offered_silver_mbps", "offered_bronze_mbps",
        "rx_gold_mbps", "rx_silver_mbps", "rx_bronze_mbps",
        "loss_gold", "loss_silver", "loss_bronze",
        "qer_red_gold", "qer_red_silver", "qer_red_bronze",
        "n6_total_mbps", "n6_utilization",
        "mbr_gold_kbps", "mbr_silver_kbps", "mbr_bronze_kbps",
    ]
    audit_cols = [
        "noise_offered_mbps", "noise_rx_mbps", "noise_loss",
        "pfcp_sessions",
        "upf_access_rx_mbps", "upf_access_drop_delta",
        "trex_total_tx_pkts", "trex_total_rx_pkts",
    ]
    fieldnames = ["step", "t_s", "regime"] + feature_cols + audit_cols

    try:
        with open(args.csv, "w", newline="") as f_csv, open(args.jsonl, "w") as f_jsonl:
            w = csv.DictWriter(f_csv, fieldnames=fieldnames)
            w.writeheader()
            t0 = time.monotonic()

            for sample in model.iter_duration(args.duration):
                step = sample.step
                target_mbps = sample.mbps
                streams, pps_by_name, packet_len_by_name = build_streams(api, target_mbps, args.wire_overhead_bytes)

                # Snapshot before the window.
                try:
                    prom_before = fetch_prometheus(args.upf_metrics_url)
                except Exception as exc:
                    print(f"[WARN] failed to read Prometheus before step {step}: {exc}")
                    prom_before = {}
                pg_before = read_pgid_snapshot(client)

                # Apply one-window traffic. This matches the audit generator, not final low-latency control.
                client.stop(ports=[args.port])
                client.reset(ports=[args.port])
                client.add_streams(streams=streams, ports=[args.port])
                client.start(ports=[args.port], duration=args.dt)

                # Pace to the end of the window.
                elapsed = time.monotonic() - t0
                target_elapsed = (step + 1) * args.dt
                time.sleep(max(0.0, target_elapsed - elapsed) + 0.20)

                pg_after = read_pgid_snapshot(client)
                try:
                    prom_after = fetch_prometheus(args.upf_metrics_url)
                except Exception as exc:
                    print(f"[WARN] failed to read Prometheus after step {step}: {exc}")
                    prom_after = {}

                upf_core_tx_bytes_before = prom_get(prom_before, "upf_bytes_count", dir="tx", iface="Core")
                upf_core_tx_bytes_after = prom_get(prom_after, "upf_bytes_count", dir="tx", iface="Core")
                upf_access_rx_bytes_before = prom_get(prom_before, "upf_bytes_count", dir="rx", iface="Access")
                upf_access_rx_bytes_after = prom_get(prom_after, "upf_bytes_count", dir="rx", iface="Access")
                upf_access_drop_before = prom_get(prom_before, "upf_dropped_count", dir="rx", iface="Access")
                upf_access_drop_after = prom_get(prom_after, "upf_dropped_count", dir="rx", iface="Access")
                pfcp_sessions = prom_get(prom_after, "pfcp_sessions", node_id="192.168.90.2")

                n6_total_mbps = mbps_from_bytes_delta(safe_delta(upf_core_tx_bytes_after, upf_core_tx_bytes_before), args.dt)
                upf_access_rx_mbps = mbps_from_bytes_delta(safe_delta(upf_access_rx_bytes_after, upf_access_rx_bytes_before), args.dt)
                upf_access_drop_delta = safe_delta(upf_access_drop_after, upf_access_drop_before)

                row: Dict[str, Any] = {"step": step, "t_s": sample.t_s, "regime": sample.regime}
                total_tx_pkts = 0.0
                total_rx_pkts = 0.0

                for s in SLICE_NAMES:
                    b = pg_before.get(s, {})
                    a = pg_after.get(s, {})
                    tx_pkts = safe_delta(pg_tx_pkts(a), pg_tx_pkts(b))
                    rx_pkts = safe_delta(pg_rx_pkts(a), pg_rx_pkts(b))
                    tx_bytes = safe_delta(pg_tx_bytes(a), pg_tx_bytes(b))
                    rx_bytes = safe_delta(pg_rx_bytes(a), pg_rx_bytes(b))

                    # Fallbacks for latency PG stats that expose packets but not bytes.
                    if tx_bytes <= 0.0 and tx_pkts > 0.0:
                        tx_bytes = tx_pkts * float(packet_len_by_name.get(s, 210))
                    if rx_bytes <= 0.0 and rx_pkts > 0.0:
                        rx_bytes = rx_pkts * float(args.rx_packet_bytes)

                    offered_mbps = float(target_mbps[s])
                    rx_mbps = mbps_from_bytes_delta(rx_bytes, args.dt)
                    loss = max(tx_pkts - rx_pkts, 0.0) / max(tx_pkts, 1.0)

                    total_tx_pkts += tx_pkts
                    total_rx_pkts += rx_pkts

                    if s in CONTROLLED:
                        row[f"offered_{s}_mbps"] = offered_mbps
                        row[f"rx_{s}_mbps"] = rx_mbps
                        row[f"loss_{s}"] = loss
                        row[f"qer_red_{s}"] = loss
                        row[f"mbr_{s}_kbps"] = mbr[s]
                    else:
                        row["noise_offered_mbps"] = offered_mbps
                        row["noise_rx_mbps"] = rx_mbps
                        row["noise_loss"] = loss

                row["n6_total_mbps"] = n6_total_mbps
                row["n6_utilization"] = n6_total_mbps / max(args.n6_capacity_mbps, 1e-9)
                row["pfcp_sessions"] = pfcp_sessions
                row["upf_access_rx_mbps"] = upf_access_rx_mbps
                row["upf_access_drop_delta"] = upf_access_drop_delta
                row["trex_total_tx_pkts"] = total_tx_pkts
                row["trex_total_rx_pkts"] = total_rx_pkts

                # Ensure all expected columns exist.
                for col in feature_cols + audit_cols:
                    row.setdefault(col, 0.0)

                w.writerow(row)
                f_jsonl.write(json.dumps(row) + "\n")

                # TensorBoard, DQN features.
                for col in feature_cols:
                    writer.add_scalar(f"dqn_features/{col}", float(row[col]), step)
                # TensorBoard, audit.
                for col in audit_cols:
                    writer.add_scalar(f"audit/{col}", float(row[col]), step)
                for s in SLICE_NAMES:
                    writer.add_scalar(f"traffic_target/{s}_mbps", float(target_mbps[s]), step)
                    writer.add_scalar(f"traffic_target/{s}_pps", float(pps_by_name[s]), step)

                print(
                    f"step={step:04d} "
                    f"offered=({row['offered_gold_mbps']:.1f},{row['offered_silver_mbps']:.1f},{row['offered_bronze_mbps']:.1f}) Mbps "
                    f"rx=({row['rx_gold_mbps']:.1f},{row['rx_silver_mbps']:.1f},{row['rx_bronze_mbps']:.1f}) Mbps "
                    f"loss=({row['loss_gold']:.3f},{row['loss_silver']:.3f},{row['loss_bronze']:.3f}) "
                    f"n6={row['n6_total_mbps']:.1f} Mbps pfcp={row['pfcp_sessions']:.0f}"
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
