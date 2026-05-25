"""Run the single-uplink TRex traffic profile.

This matches the validated exp2 pattern: one TX port, several GTP-U streams,
one TEID per slice. It does not create one TRex port per slice.

Example:
    python scripts/run_trex_traffic.py --port 0 --duration 600
    python scripts/run_trex_traffic.py --port 0 --no-noise
    python scripts/run_trex_traffic.py --port 0 --pps-scale 0.75
"""
from __future__ import annotations

import argparse
import logging

from upf_rl_traffic.trex_conductor import ConductorConfig, TrexConductor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1", help="TRex daemon host")
    p.add_argument("--sync-port", type=int, default=4501)
    p.add_argument("--port", type=int, default=0, help="single TRex TX uplink port")
    p.add_argument("--duration", type=float, default=None, help="seconds to run, default until Ctrl-C")
    p.add_argument("--pps-scale", type=float, default=1.0, help="scale every stream pps by this factor")
    p.add_argument("--no-noise", action="store_true", help="disable the fourth noise slice")
    p.add_argument("--no-hidden", action="store_true", help="deprecated alias for --no-noise")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = ConductorConfig(
        trex_host=args.host,
        trex_sync_port=args.sync_port,
        tx_port=args.port,
        duration_s=args.duration,
        include_noise=not (args.no_noise or args.no_hidden),
        pps_scale=args.pps_scale,
    )
    TrexConductor(cfg=cfg).run()


if __name__ == "__main__":
    main()
